import os
import json
import time
from turtle import st
import torch
import numpy as np
from storage import RolloutStorage
from model import Policy
from env import get_env
from utils import get_config, get_logger
import algorithm as algo
from torch.utils.tensorboard import SummaryWriter


def train(cfg):
    seed = cfg['train']['seed']
    # Setup logger
    logger = get_logger(
        name=cfg["id"],
        seed=seed,
        add_date=cfg["train"]["logs"]["add_date"],
    )
    logger.info(cfg)
    writer = SummaryWriter(log_dir=os.path.join("./tb_logs", cfg["algorithm"], cfg["id"]))

    # Parse hyperparams
    num_workers = cfg["train"]["num_workers"]
    num_steps = cfg["train"]["num_steps"]
    algo_params = cfg["train"]["algorithm_params"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tb_log_interval = cfg["train"]["tb_log_interval"]
    save_interval = cfg["train"]["save_interval"]
    save_path = os.path.join("./checkpoints", cfg["algorithm"], cfg["id"], str(seed))

    envs = get_env(cfg=cfg, num_workers=num_workers, device=device)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space).to(
        device=device
    )

    agent = algo.PPO(actor_critic=actor_critic, **algo_params)

    rollouts = RolloutStorage(
        num_steps=num_steps,
        num_processes=num_workers,
        obs_shape=envs.observation_space.shape,
        action_space=envs.action_space,
        recurrent_hidden_state_size=actor_critic.recurrent_hidden_state_size,
    )

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    start = time.time()
    num_updates = int(cfg["train"]["num_env_steps"]) // num_steps // num_workers
    logger.info(f"Number of updates is set to: {num_updates}")
    logger.info(f"Training Begins!")

    for j in range(num_updates):
        for step in range(num_steps):
            # Sample actions with current policy
            with torch.no_grad():
                (
                    value,
                    action,
                    action_log_prob,
                    recurrent_hidden_states,
                ) = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step],
                )

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(actions=action)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
            )
            rollouts.insert(
                obs=obs,
                recurrent_hidden_states=recurrent_hidden_states,
                actions=action,
                action_log_probs=action_log_prob,
                value_preds=value,
                rewards=reward,
                masks=masks,
                bad_masks=bad_masks,
            )

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1],
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1],
            ).detach()

        rollouts.compute_returns(
            next_value=next_value, **cfg["train"]["compute_returns"]
        )

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if j % save_interval == 0 or j == num_updates - 1:
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            if j == 0:
                with open(os.path.join(save_path, "config.json"), "w") as output:
                    logger.info(f"num update: {j}. Saving config.yaml...")
                    json.dump(cfg, output)

            logger.info(f"num update: {j}. Saving checkpoint...")
            torch.save(
                {
                    "epoch": j,
                    "model_state_dict": agent.actor_critic.state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),
                },
                os.path.join(save_path, "checkpoint.pt")
            )

        if j % tb_log_interval == 0 or j == num_updates - 1:
            total_num_steps = (j + 1) * num_workers * num_steps
            end = time.time()
            mean_reward = torch.mean(rollouts.rewards)
            median_reward = torch.median(rollouts.rewards)
            writer.add_scalar(tag='FPS', scalar_value=int(total_num_steps / (end - start)), global_step=total_num_steps)
            writer.add_scalar(tag="Mean Reward Of Last Rollout", scalar_value=mean_reward, global_step=total_num_steps)
            writer.add_scalar(tag="Median Reward Of Last Rollout", scalar_value=median_reward, global_step=total_num_steps)
            writer.add_scalar(tag="Distribution Entropy At Num Step", scalar_value=dist_entropy, global_step=total_num_steps)
            writer.add_scalar(tag="Value Loss At Num Step", scalar_value=value_loss, global_step=total_num_steps)
            writer.add_scalar(tag="Actionn Loss At Num Step", scalar_value=action_loss, global_step=total_num_steps)
            logger.info(f'Step:{total_num_steps}/{num_updates*num_workers*num_steps}, mean reward: {mean_reward}, median reward: {median_reward}')
    
    logger.info(f'Total Time To Complete: {end - start}')
    writer.close()


if __name__ == "__main__":
    cfg = get_config()

    if cfg['device_id'] is not None:
        with torch.cuda.device(cfg['device_id']):
            train(cfg)
    else:
        train(cfg)
