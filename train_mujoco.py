import os
import json
import time
import torch
import numpy as np
from storage import RolloutStorage
from model import Policy
from env import get_env
from utils import get_config, get_logger
import algorithm as algo
from collections import deque


def train(cfg):
    # Setup logger
    logger = get_logger(
        name=cfg["id"],
        seed=cfg["train"]["seed"],
        add_date=cfg["train"]["logs"]["add_date"],
    )
    logger.info(cfg)

    # Parse hyperparams
    num_workers = cfg["train"]["num_workers"]
    num_steps = cfg["train"]["num_steps"]
    algo_params = cfg["train"]["algorithm_params"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_interval = cfg["train"]["log_interval"]
    save_interval = cfg["train"]["save_interval"]
    save_path = os.path.join("./checkpoints", cfg["algorithm"], cfg["id"])

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

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(cfg["train"]["num_env_steps"]) // num_steps // num_workers
    logger.info(f"Number of updates is: {num_updates}")

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

            for info in infos:
                if "episode" in info.keys():
                    episode_rewards.append(info["episode"]["r"])

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
                    logger.info(f"Saving config.yaml...")
                    json.dump(cfg, output)

            logger.info(f"num update: {j}. Saving checkpoint...")
            torch.save(
                {
                    "epoch": j,
                    "model_state_dict": agent.actor_critic.state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),
                },
                os.path.join(save_path, "checkpoint.pt"),
            )

        if j % log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * num_workers * algo_params.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(
                    j,
                    total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards),
                    np.mean(episode_rewards),
                    np.median(episode_rewards),
                    np.min(episode_rewards),
                    np.max(episode_rewards),
                    dist_entropy,
                    value_loss,
                    action_loss,
                )
            )


if __name__ == "__main__":
    cfg = get_config()
    train(cfg)
