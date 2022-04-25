import torch
import logging
from models.PPO_model import PPO_model
from utils.logger_and_config import get_config, setup_logger
from envs.SB3MuJoCo import SB3MuJoCo
from stable_baselines3.common.env_util import make_vec_env

def train(cfg):
    seed = cfg['train']['seed']
    num_mini_batch = cfg['train']['num_mini_batch']
    num_workers = cfg['train']['num_workers']
    batch_sz = num_workers * cfg['train']['worker_steps']
    mini_batch_sz = batch_sz // num_mini_batch
    assert batch_sz % num_mini_batch == 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_vec_env(SB3MuJoCo, n_envs=num_workers, seed=seed, env_kwargs={'cfg':cfg})


if __name__ == '__main__':
    cfg = get_config()
    # !!!TODO: Add back in logger
    # setup_logger(cfg['id'])
    train(cfg)