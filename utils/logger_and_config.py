import yaml
import datetime
import argparse
import logging

def setup_logger(name):
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    logging.basicConfig(filename=f'./logging/{name}_{ts}.log', level=logging.WARNING)

def get_config():
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="./configs/mujoco/start.yaml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)
    
    if not cfg['id']:
        raise ValueError('"id" should not be none in config yaml')

    return cfg

def setup_logger_and_config():
    cfg = get_config()
    setup_logger(cfg['id'])

    return cfg
