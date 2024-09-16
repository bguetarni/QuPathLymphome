import os

############################  OPENSLIDE  #################################
# The path can also be read from a config file, etc.
OPENSLIDE_PATH = r"C:\Users\bilel.guetarni\openslide-win64-20230414\bin"

if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide
##########################################################################

import argparse, logging, time, json, glob
import numpy as np
import pandas
import webbrowser

from libraries.layout_dashboard import dashboard_layout
from libraries.models import PredictionModel
from libraries import utils

# for each task, specify which magnification level to extract the area
# depend on the model training
MAGNIFICATION_LEVEL = {"subtyping": 1, "treatment": 1}

def gather_data(cwd, logger):
    """
    Gather the data of all annotations

    args:
        cwd (str): current working directory
        logger: to log information
    """
    df = []
    for annotation_path in glob.glob(os.path.join(cwd, "data", "*", "*", "*")):

        base, task = os.path.split(annotation_path)
        base, annotation = os.path.split(base)
        base, wsi = os.path.split(base)

        logger.info("check image exists")
        if os.path.exists(os.path.join(annotation_path, "image.png")):
            image_path = os.path.join(annotation_path, "image.png")
        else:
            continue

        logger.info("check prediction file exists")
        if os.path.exists(os.path.join(annotation_path, "predictions.json")):
            with open(os.path.join(annotation_path, "predictions.json"), "r") as json_file:
                predictions = json.load(json_file)
                class_, pb = list(predictions.items())[0]
        else:
            continue

        logger.info("check attention score file exists")
        if os.path.exists(os.path.join(annotation_path, "attention_scores.json")):
            with open(os.path.join(annotation_path, "attention_scores.json"), "r") as json_file:
                attention_scores = json.load(json_file)
        else:
            continue

        logger.info("gather data in dictionary")
        df.append({"wsi": wsi, "annotation": annotation, "task": task, "image": image_path, 
                   "class": class_, "probability": pb, "attention_scores": attention_scores})

    return pandas.DataFrame(df)


if __name__ == '__main__':
    # set path of the project
    cwd = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="127.0.0.1", help="URL to run dashapp")
    parser.add_argument("--port", type=int, default=8050, help="port to run dashapp")
    parser.add_argument("--wsiPath", type=str, required=True, help="absolute path to the WSI file")
    parser.add_argument("--outputPath", type=str, required=True, help="absolute path to the annotation folder (to store the results)")
    parser.add_argument("--annotationPath", type=str, required=True, help="absolute path to the annotation JSON file")
    parser.add_argument("--task", type=str, required=True, choices=["subtyping", "treatment"], help="task to perform")
    parser.add_argument("--foundation_model", type=str, default="double", choices=["single", "double"], help="which model to use for treatment response prediction \
                         single (HIPT) or double (CONCH+HIPT)")
    args = parser.parse_args()

    # logger
    handler = logging.FileHandler(filename=os.path.join(args.outputPath, "log"))
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    logger.addHandler(handler)

    # log arguments
    logger.info(time.strftime("%x %X"))
    for k, v in vars(args).items():
        logger.info('{} : {}'.format(k,v))

    # load annotation
    try:
        with open(args.annotationPath, "r") as json_file:
            features = json.load(json_file)

        cntrs = features['geometry']['coordinates']
        cntrs = np.array(cntrs).reshape((-1, 2))
    except json.JSONDecodeError as e:
        exit(5)

    x, y = cntrs[0]
    x1, y1 = cntrs[2]

    # read region
    logger.info("extracting RGB region from wsi file")
    wsi = openslide.OpenSlide(args.wsiPath)
    size = utils.calculate_region_size(wsi, (x,y), (x1, y1), target_level=MAGNIFICATION_LEVEL[args.task])
    img = wsi.read_region(location=(x,y), level=MAGNIFICATION_LEVEL[args.task], size=size).convert("RGB")

    # save image to annotation folder
    logger.info("saving RGB region in png file")
    img.save(os.path.join(args.outputPath, "image.png"))

    # apply prediction model
    logger.info("loading prediction model")
    pmodel = PredictionModel.get_prediction_model(args)
    logger.info("applying prediction model")
    results, attn_scores = pmodel.apply(img)

    # save results
    logger.info("saving predictions in json file")
    with open(os.path.join(args.outputPath, "predictions.json"), "w") as json_file:
        json.dump(results, json_file)

    # save attention scores
    if attn_scores is not None:
        logger.info("saving attention scores in json file")
        with open(os.path.join(args.outputPath, "attention_scores.json"), "w") as json_file:
            json.dump(attn_scores, json_file)

    # df = get_data_from_json(cwd)
    logger.info("creating pandas.DataFrame for dash")
    df = gather_data(cwd, logger)

    # launch dash app
    logger.info("extract wsi, annotation and task names")
    base, _ = os.path.split(args.outputPath)
    base, annotation = os.path.split(base)
    base, wsi = os.path.split(base)
    logger.info("opening web browser")
    webbrowser.open_new('http://{}:{}/'.format(args.url, args.port))
    app = dashboard_layout(df, wsi, annotation, args.task)
    logger.info("launch dash app")
    app.run_server(host=args.url, port=args.port, debug=True, use_reloader=False)
    exit(0)
