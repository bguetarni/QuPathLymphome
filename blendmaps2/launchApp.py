import os, argparse, logging, time, datetime
import webbrowser

from libraries.get_datas import get_data_from_json
from libraries.layout_dashboard import dashboard_layout
from libraries.models.generateDatas import create_proba, create_heatmap_png

# set path of the project
cwd = os.path.dirname(__file__)

TILE_SIZE = 50

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="127.0.0.1", help="URL to run dashapp")
    parser.add_argument("--port", type=int, default=8050, help="port to run dashapp")
    parser.add_argument("--wsiPath", type=str, required=True, help="absolute path to the WSI file")
    parser.add_argument("--outputPath", type=str, required=True, help="absolute path to the ouptut folder to store the results")
    parser.add_argument("--task", type=str, required=True, choices=["subtyping", "treatment"], help="task to perform")
    parser.add_argument("--model", type=str, help="which model to use")
    args = parser.parse_args()

    # logger
    now = datetime.datetime.now()
    handler = logging.FileHandler(filename=os.path.join(cwd, "logs", now.strftime("%Y-%m-%d-%H%M%S")))
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    logger.addHandler(handler)

    logger.info(time.strftime("%x %X"))
    for k, v in vars(args).items():
        logger.info('{} : {}'.format(k,v))
        
    create_proba(cwd, TILE_SIZE)
    create_heatmap_png(cwd)

    df = get_data_from_json(cwd)

    webbrowser.open_new('http://{}:{}}/'.format(args.url, args.port))
    app = dashboard_layout(df, cwd)
    app.run_server(host=args.url, port=args.port, debug=True, use_reloader=False)
