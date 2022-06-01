from pathlib import Path

import torch.nn as nn

from hand_env_utils.arg_utils import *
from hand_env_utils.teleop_env import create_relocate_env
from hand_env_utils.wandb_callback import WandbCallback, setup_wandb
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ep', type=int, default=20)
    parser.add_argument('--bs', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--object_name', type=str)

    args = parser.parse_args()
    object_name = args.object_name
    exp_keywords = ["ppo", object_name, args.exp, str(args.seed)]
    env_iter = args.iter * 500 * args.n

    config = {
        'n_env_horizon': args.n,
        'object_name': args.object_name,
        'update_iteration': args.iter,
        'total_step': env_iter,
    }

    exp_name = "-".join(exp_keywords)
    result_path = Path("./results") / exp_name
    result_path.mkdir(exist_ok=True, parents=True)
    wandb_run = setup_wandb(config, exp_name, tags=["state", "relocate", object_name])


    def create_env_fn():
        environment = create_relocate_env(object_name, use_visual_obs=False)
        return environment


    env = SubprocVecEnv([create_env_fn] * args.workers, "spawn")

    print(env.observation_space, env.action_space)

    model = PPO("MlpPolicy", env, verbose=1,
                n_epochs=args.ep,
                n_steps=(args.n // args.workers) * 500,
                learning_rate=args.lr,
                batch_size=args.bs,
                seed=args.seed,
                policy_kwargs={'activation_fn': nn.ReLU},
                tensorboard_log=str(result_path / "log"),
                min_lr=args.lr,
                max_lr=args.lr,
                adaptive_kl=0.02,
                target_kl=0.2,
                )

    model.learn(
        total_timesteps=int(env_iter),
        callback=WandbCallback(
            model_save_freq=50,
            model_save_path=str(result_path / "model"),
        ),
    )
    wandb_run.finish()
