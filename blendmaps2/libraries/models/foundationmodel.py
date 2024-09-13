import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from conch.open_clip_custom import create_model_from_pretrained
from . import HIPT

def conch():
    """
    Perform `pip install git+https://github.com/Mahmoodlab/CONCH.git` before
    """

    model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", 
                                                     hf_auth_token="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")
    return model, preprocess

def hipt():
    model = HIPT.get_vit256(os.path.join(__file__, "checkpoints", "HIPT", "vit256_small_dino.pth"))

    # for normalization values see https://github.com/mahmoodlab/HIPT/issues/6
    # or also https://github.com/mahmoodlab/HIPT/blob/780fafaed2e5b112bc1ed6e78852af1fe6714342/HIPT_4K/hipt_model_utils.py#L111
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    return model, preprocess

class Foundation(nn.Module):
    def __init__(self, foundation_model):
        """
        args:
            foundation_model (string): type of foundation models single or double
        """
        super().__init__()

        if foundation_model == "single":
            # preserve the order of foundation models during training
            self.order = ["hipt"]
            
            ftext, preprocess = hipt()
            self.models = [
                {"name": "hipt", "ftext": ftext, "preprocess": preprocess}
            ]
        else:
            # preserve the order of foundation models during training
            self.order = ["conch", "hipt"]
            
            conch_ftext, conch_preprocess = conch()
            hipt_ftext, hipt_preprocess = hipt()            
            self.models = [
                {"name": "conch", "ftext": conch_ftext, "preprocess": conch_preprocess},
                {"name": "hipt", "ftext": hipt_ftext, "preprocess": hipt_preprocess},
            ]

    def forward(self, x):
        """
        args:
            x (List[PIL.Image]) list of PNG images
        
            return list of torch.Tensor features
        """

        out = []
        for img in x:
            features = {}
            for model in self.models:
                processed_img = model["preprocess"](img).unsqueeze(dim=0)
                if model["name"] == "conch":
                    fts = model["ftext"].encode_image(processed_img)
                else:
                    fts = model["ftext"](processed_img)
                features.update({model["name"]: fts})
            
            features = [features[k] for k in self.order]
            features = torch.cat(features, dim=-1)
            out.append(features)
        
        return out


class ABMIL(nn.Module):
    def __init__(self, in_chn, num_cls):
        super().__init__()

        # default values from https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(in_chn, self.M),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Linear(self.M*self.ATTENTION_BRANCHES, num_cls)

    def forward(self, x):
        """
        args:
            x (torch.Tensor): a bag of feature vectors of shape (1,N,C) or (N,C)
        """
        
        if len(x.shape) == 3:
            x = x.squeeze(dim=0)

        H = self.feature_extractor_part2(x)  # KxM

        A_V = self.attention_V(H)  # KxL
        A_U = self.attention_U(H)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        logits = self.classifier(Z)

        return logits, A

class FusionMIL(nn.Module):
    def __init__(self, foundation_model, num_cls):
        super().__init__()
        
        EXTRACTOR_OUT_DIM = {"single": 384, "double": 512+384}

        # feature extractor(s)
        self.FM = Foundation(foundation_model)
        
        # MIL aggregator
        out_dim = EXTRACTOR_OUT_DIM[foundation_model]
        self.mil_aggregator = ABMIL(out_dim, num_cls)

    def forward(self, x):
        """
        args:
            x (List[PIL.Image]) list of PNG images

        return logits and attetnion scores
        """

        # return list of feature vectors
        x = self.FM(x) # [(1,C), (1,C), ...]

        x = torch.stack(x, dim=1) # (1,N,C)
        out, attn = self.mil_aggregator(x)
        return out, attn
