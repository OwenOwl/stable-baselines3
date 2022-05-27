from pathlib import Path

import torch.nn as nn
import wandb

from hand_env_utils.arg_utils import *
from hand_env_utils.teleop_env import create_relocate_env
from hand_env_utils.wandb_callback import WandbCallback
from hand_teleop.real_world import task_setting
from stable_baselines3.common.torch_layers import PointNetImaginationExtractor
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO


def setup_wandb(parser_config, exp_name):
    run = wandb.init(
        project="hand_teleop_ppo",
        name=exp_name,
        config=parser_config,
        monitor_gym=True,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  # optional
    )
    return run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ep', type=int, default=5)
    parser.add_argument('--bs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--iter', type=int, default=2000)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--object_name', type=str)
    parser.add_argument('--use_bn', type=bool, default=True)
    parser.add_argument('--img_type',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=['robot', 'goal_robot', 'goal'], )

    args = parser.parse_args()
    object_name = args.object_name
    exp_keywords = ["ppo_imagination", args.img_type, object_name, args.exp, str(args.seed)]
    env_iter = args.iter * 500 * args.n

    config = {
        'n_env_horizon': args.n,
        'object_name': args.object_name,
        'update_iteration': args.iter,
        'total_step': env_iter,
        'use_bn': args.use_bn,
        'img_type': args.img_type,
    }

    exp_name = "-".join(exp_keywords)
    result_path = Path("./results") / exp_name
    result_path.mkdir(exist_ok=True, parents=True)

    wandb_run = setup_wandb(config, exp_name)


    def create_env_fn():
        environment = create_relocate_env(object_name, use_visual_obs=True)
        if args.img_type == "robot":
            environment.setup_imagination_config(task_setting.IMG_CONFIG["relocate_robot_only"])
        elif args.img_type == "goal":
            environment.setup_imagination_config(task_setting.IMG_CONFIG["relocate_goal_only"])
        elif args.img_type == "goal_robot":
            environment.setup_imagination_config(task_setting.IMG_CONFIG["relocate_goal_robot"])
        return environment


    env = SubprocVecEnv([create_env_fn] * args.workers, "spawn")

    print(env.observation_space, env.action_space)

    feature_extractor_class = PointNetImaginationExtractor
    feature_extractor_kwargs = {
        "pc_key": "relocate-point_cloud",
        "local_channels": (64, 128, 256),
        "global_channels": (256,),
        "imagination_keys": ("imagination_goal",),
        "use_bn": args.use_bn,
    }
    policy_kwargs = {
        "features_extractor_class": feature_extractor_class,
        "features_extractor_kwargs": feature_extractor_kwargs,
        "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
        "activation_fn": nn.Tanh,
    }

    model = PPO("PointCloudPolicy", env, verbose=1,
                n_epochs=args.ep,
                n_steps=(args.n // args.workers) * 500,
                learning_rate=args.lr,
                batch_size=args.bs,
                seed=args.seed,
                policy_kwargs=policy_kwargs,
                tensorboard_log=str(result_path / "log"),
                min_lr=1e-4,
                max_lr=args.lr,
                target_kl=0.02,
                )

    model.learn(
        total_timesteps=int(env_iter),
        callback=WandbCallback(
            model_save_freq=10,
            model_save_path=str(result_path / "model"),
        ),
    )
    wandb_run.finish()
