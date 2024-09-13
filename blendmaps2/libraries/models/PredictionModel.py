import os, json, math
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from PIL import Image

from .visiontransformer import MultiStainViTStudent
from .foundationmodel import FusionMIL

def get_prediction_model(args):
    """
    args:
        args (Namespace): main script arguments
    """
    if args.task == "subtyping":
        return SubtypingModel()
    elif args.task == "treatment":
        return TreatmentResponseModel(args.foundation_model)
    else:
        return None

class PredictionModel:
    """
        Parent class for prediction models
    """
    
    def __init__(self):
        # store dictionnary of class indexes and names
        self.classes = None

    def apply(self, img):
        """
        args:
            img (PIL.Image): image to apply model on

        return:
            results as a dictionary containing prediction scores for each class
            attention scores as a list of region coordinates and score (return None if not applicable)
        """
        pass

    def post_process(self, prediction):
        """
        args:
            prediction (torch.Tensor): prediction of model
        """
        pass

    def patching(self, img, patch_size, zero_padding=True):
        """
        args:
            img (PIL.Image): image of mode RGB
            patch_size (int): dimension size (both width and height) of a patch
            zero_padding (bool): wether to apply zero padding if the image size is not propotional to patch_size

        return list of patches (List[PIL.Image])
        """

        # fill borders with zeros to have size proportional to patch size
        if zero_padding:
            w, h = img.size
            w = w if (w % patch_size) == 0 else math.ceil(w / patch_size) * patch_size
            h = h if (h % patch_size) == 0 else math.ceil(h / patch_size) * patch_size
            tmp = Image.new(img.mode, (w, h), (0,0,0))
            tmp.paste(img, (0,0))
            img = tmp

        # patching
        #TODO: filter patches without tissue (based on bright values)
        patches = []
        for y in range(0, img.size[1] - patch_size + 1, patch_size):
            for x in range(0, img.size[0] - patch_size + 1, patch_size):
                patches.append(img.crop((x, y, x + patch_size, y + patch_size)))

        return patches
        

class SubtypingModel(PredictionModel):
    def __init__(self):
        super().__init__()

        # properties
        self.classes = {0: "GCB", 1: "ABC"}
        self.patch_size = 32

        # load prediction model
        with open(os.path.join(os.path.split(__file__)[0], "checkpoints", "subtyping", "student_100_1", "config.json"), "r") as json_f:
            vit_config = json.load(json_f)["vit_config"]
        self.model = MultiStainViTStudent(vit_config, num_classes=2)
        state_dict = torch.load(os.path.join(os.path.split(__file__)[0], "checkpoints", "subtyping", "student_100_1", "ckpt.pth"), map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        # normalization statistics
        self.mean = torch.tensor((0.5, 0.5, 0.5), dtype=torch.float32)
        self.std = torch.tensor((0.5, 0.5, 0.5), dtype=torch.float32)

    def apply(self, img):
        """
        args:
            img (PIL.Image) input image

        return result and attention scores
        """
        
        # patching
        x = super().patching(img, self.patch_size, zero_padding=True)

        # convert to torch.Tensor
        x = list(map(torchvision.transforms.functional.pil_to_tensor, x))
        x = torch.stack(x, dim=0).to(dtype=torch.float32)

        # normalize
        x = torch.moveaxis(x, 1, -1)
        x = (x / 255 - self.mean) / self.std
        x = torch.moveaxis(x, -1, 1)

        # model prediction
        x = x.unsqueeze(dim=0)
        with torch.no_grad():
            pred, attn_scores = self.model(x)
        
        return self.post_process(pred, attn_scores, img)
    
    def post_process(self, prediction, attn_scores, img):
        """
        args:
            prediction (torch.Tensor): prediction of model
            attn_scores (torch.Tensor): list of attention scores
            img (PIL.Image) input image
        """
        
        # transform scores into probability
        prediction = prediction[0]   # remove batch dimension
        prediction = F.softmax(prediction, dim=-1)

        # retrieve class with highest probability
        idx = int(torch.argmax(prediction))
        k = self.classes[idx]
        prediction = {k: prediction[idx].item()}

        # transform attentions scores
        attn_scores = attn_scores[0]    # remove batch dimension
        attn_scores = attn_scores.mean(dim=0)    # average heads attention scores
        attn_scores = attn_scores[0,1:]    # keep only attention scores of the cls token towards the patches

        # construct dict of coordinates with attention score
        scores = []
        for i, s in enumerate(attn_scores):
            # top-left coordinates of patch within input image
            x = math.floor((((i-1) * self.patch_size) % img.size[0]) / self.patch_size)
            y = math.floor(((i-1) * self.patch_size) / img.size[1])
            
            # save box coordinates of patch and attention score
            scores.append({"x0": x, "y0": y, "x1": x + self.patch_size, "y1": y + self.patch_size,
                           "score": s.item()})

        return prediction, scores

class TreatmentResponseModel(PredictionModel):
    def __init__(self, foundation_model):
        """
        args:
            foundation_model (str): type of foundation model (single or double)
        """
        super().__init__()
        
        # properties
        self.classes = {0: "NEGATIVE", 1: "POSITIVE"}
        self.patch_size = 256

        # load prediction model
        self.model = FusionMIL(foundation_model, num_cls=1)
        if foundation_model == "single":
            state_path = os.path.join(os.path.split(__file__)[0], "checkpoints", "treatment", "single_hipt_None_abmil", "fold_0", "best.pth")
        else:
            state_path = os.path.join(os.path.split(__file__)[0], "checkpoints", "treatment", "double_conch_hipt_concat_abmil", "fold_1", "best.pth")
        
        state_dict = torch.load(state_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def apply(self, img):
        """
        args:
            img (PIL.Image) input image

        return result and attention scores
        """

        # patching
        x = super().patching(img, self.patch_size, zero_padding=True)

        # model prediction
        with torch.no_grad():
            pred, attn_scores = self.model(x)

        # return pred, attn_scores
        return self.post_process(pred, attn_scores, img)
    
    def post_process(self, prediction, attn_scores, img):
        """
        args:
            prediction (torch.Tensor): prediction of model
            attn_scores (torch.Tensor): list of attention scores
            img (PIL.Image) input image
        """
        
        # transform scores into probability
        prediction = prediction[0]   # remove batch dimension
        prediction = F.sigmoid(prediction).item()

        # retrieve class based on thresholding
        if prediction > 0.5:
            prediction = {self.classes[1]: prediction}
        else:
            prediction = {self.classes[0]: 1 - prediction}
        
        # remove batch dimension
        attn_scores = attn_scores[0]

        # construct dict of coordinates with attention score
        scores = []
        for i, s in enumerate(attn_scores):
            # top-left coordinates of patch within input image
            x = math.floor((((i-1) * self.patch_size) % img.size[0]) / self.patch_size)
            y = math.floor(((i-1) * self.patch_size) / img.size[1])

            # save box coordinates of patch and attention score
            scores.append({"x0": x, "y0": y, "x1": x + self.patch_size, "y1": y + self.patch_size, 
                           "score": s.item()})

        return prediction, scores
