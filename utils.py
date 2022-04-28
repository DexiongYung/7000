import yaml
import torch
import datetime
import argparse
import logging


def get_logger(name: str, seed: int, add_date: bool):
    logger = logging.getLogger(name)

    if add_date:
        ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
        ts = ts.replace(":", "_").replace("-", "_")
        ts = "_" + ts
    else:
        ts = ""

    file_path = f"./logging/{name}_{seed}{ts}.log"
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


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
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    if not cfg["id"]:
        raise ValueError('"id" should not be none in config yaml')

    return cfg


def process_obs(obs: torch.tensor):
    if not isinstance(obs, torch.Tensor):
        try:
            obs = torch.tensor(obs)
        except ValueError as e:
            obs = torch.tensor(obs.copy())

    err_sz = "obs should be of dimension Batch x Num Channel x Height x Width."
    if len(obs.shape) == 3:
        obs = torch.unsqueeze(obs, 0)
    elif len(obs.shape) != 4:
        raise ValueError(f"{err_sz} Got obs of shape length = {len(obs.shape)}.")

    if obs.shape[1] != 3:
        if obs.shape[3] == 3:
            obs = torch.transpose(obs, 1, 3)
        else:
            raise ValueError("None of the expected dimensions of: [1,3] were of size 3")

    return obs


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class AddBias(torch.nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = torch.nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
