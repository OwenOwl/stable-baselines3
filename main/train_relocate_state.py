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
    parser.add_argument('--ep', type=int, default=10)
    parser.add_argument('--bs', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument('--randomness', type=float, default=1.0)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--object_name', type=str)
    parser.add_argument('--object_cat', default="YCB", type=str)

    args = parser.parse_args()
    object_name = args.object_name
    object_cat = args.object_cat
    randomness = args.randomness
    exp_keywords = ["ppo", object_name, args.exp, str(args.seed)]
    horizon = 200
    env_iter = args.iter * horizon * args.n

    config = {
        'n_env_horizon': args.n,
        'object_name': object_name,
        'object_category': object_cat,
        'update_iteration': args.iter,
        'total_step': env_iter,
        'randomness': randomness,
    }

    exp_name = "-".join(exp_keywords)
    result_path = Path("./results") / exp_name
    result_path.mkdir(exist_ok=True, parents=True)
    wandb_run = setup_wandb(config, exp_name, tags=["state", "relocate", object_name])


    def create_env_fn():
        environment = create_relocate_env(object_name, use_visual_obs=False, object_category=object_cat,
                                          randomness_scale=randomness)
        return environment


    def create_eval_env_fn():
        environment = create_relocate_env(object_name, use_visual_obs=False, is_eval=True, object_category=object_cat,
                                          randomness_scale=randomness)
        return environment


    env = SubprocVecEnv([create_env_fn] * args.workers, "spawn")

    print(env.observation_space, env.action_space)

    model = PPO("MlpPolicy", env, verbose=1,
                n_epochs=args.ep,
                n_steps=(args.n // args.workers) * horizon,
                learning_rate=args.lr,
                batch_size=args.bs,
                seed=args.seed,
                policy_kwargs={'activation_fn': nn.ReLU},
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
            eval_env_fn=create_eval_env_fn,
            eval_freq=50,
            eval_cam_names=["relocate_viz"],
        ),
    )
    wandb_run.finish()
