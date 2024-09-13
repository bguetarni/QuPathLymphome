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

def gather_data(cwd):
    """
    Gather the data of all annotations

    args:
        cwd (str): current working directory
    """
    df = []
    for file_ in glob.glob(os.path.join(cwd, "data", "*", "*", "*", "*.json")):
        base, name = os.path.split(file_)
        name = os.path.splitext(name)[0]

        base, task = os.path.split(base)
        base, annotation = os.path.split(base)
        base, wsi = os.path.split(base)

        if os.path.exists(file_):
            with open(file_, "r") as json_file:
                data = json.load(json_file)
        
        df.append({"wsi": wsi, "annotation": annotation, "task": task, "type": name, "data": data})

    return pandas.DataFrame(df)


if __name__ == '__main__':
    # set path of the project
    cwd = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="127.0.0.1", help="URL to run dashapp")
    parser.add_argument("--port", type=int, default=8050, help="port to run dashapp")
    parser.add_argument("--wsiPath", type=str, required=True, help="absolute path to the WSI file")
    parser.add_argument("--outputPath", type=str, required=True, help="absolute path to the annotation folder (to store the results)")
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
    logger.info("loading annotation")
    annotation_files = glob.glob(os.path.join(args.outputPath, "*.json"))
    if len(annotation_files) == 0:
        logger.info("no JSON annotation found in {}".format(args.outputPath))
        exit(3)
    elif len(annotation_files) > 1:
        logger.info("more than 1 JSON annotation found in {}".format(args.outputPath))
        exit(4)
    
    try:
        with open(annotation_files[0], "r") as json_file:
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
    logger.info("creaeting pandas.DataFrame for dash")
    df = gather_data(cwd)

    # # launch dash app
    # base, task = os.path.split(args.outputPath)
    # base, annotation = os.path.split(base)
    # base, wsi = os.path.split(base)
    # webbrowser.open_new('http://{}:{}}/'.format(args.url, args.port))
    # app = dashboard_layout(df, wsi, annotation, task)
    # app.run_server(host=args.url, port=args.port, debug=True, use_reloader=False)

    exit(0)
