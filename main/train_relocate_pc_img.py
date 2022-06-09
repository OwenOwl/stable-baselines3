from pathlib import Path

import torch.nn as nn

from hand_env_utils.arg_utils import *
from hand_env_utils.teleop_env import create_relocate_env
from hand_env_utils.wandb_callback import WandbCallback, setup_wandb
from hand_teleop.real_world import task_setting
from stable_baselines3.common.torch_layers import PointNetImaginationExtractor
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ep', type=int, default=10)
    parser.add_argument('--bs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument('--randomness', type=float, default=1.0)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--object_name', type=str)
    parser.add_argument('--object_cat', default="YCB", type=str)
    parser.add_argument('--use_bn', type=bool, default=True)
    parser.add_argument('--img_type',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=['robot', 'goal_robot', 'goal'], )

    args = parser.parse_args()
    object_name = args.object_name
    object_cat = args.object_cat
    randomness = args.randomness
    exp_keywords = ["ppo_img", args.img_type, object_name, args.exp, str(args.seed)]
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
    wandb_run = setup_wandb(config, exp_name, tags=["imagination", "relocate", object_name])

    if args.img_type == "robot":
        img_config = task_setting.IMG_CONFIG["relocate_robot_only"]
        imagination_keys = ("imagination_robot",)
    elif args.img_type == "goal":
        img_config = task_setting.IMG_CONFIG["relocate_goal_only"]
        imagination_keys = ("imagination_goal",)
    elif args.img_type == "goal_robot":
        img_config = task_setting.IMG_CONFIG["relocate_goal_robot"]
        imagination_keys = ("imagination_goal", "imagination_robot")
    else:
        raise NotImplementedError


    def create_env_fn():
        environment = create_relocate_env(object_name, use_visual_obs=True, object_category=object_cat,
                                          randomness_scale=randomness)
        environment.setup_imagination_config(img_config)
        return environment


    def create_eval_env_fn():
        environment = create_relocate_env(object_name, use_visual_obs=True, is_eval=True, object_category=object_cat,
                                          randomness_scale=randomness)
        environment.setup_imagination_config(img_config)
        return environment


    env = SubprocVecEnv([create_env_fn] * args.workers, "spawn")

    print(env.observation_space, env.action_space)

    feature_extractor_class = PointNetImaginationExtractor
    feature_extractor_kwargs = {
        "pc_key": "relocate-point_cloud",
        "local_channels": (64, 128, 256),
        "global_channels": (256, ),
        "imagination_keys": imagination_keys,
        "use_bn": args.use_bn,
    }
    policy_kwargs = {
        "features_extractor_class": feature_extractor_class,
        "features_extractor_kwargs": feature_extractor_kwargs,
        "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
        "activation_fn": nn.ReLU,
    }

    config = {'n_env_horizon': args.n, 'object_name': args.object_name, 'update_iteration': args.iter,
              'total_step': env_iter, "use_bn": args.use_bn, "policy_kwargs": policy_kwargs}

    model = PPO("PointCloudPolicy", env, verbose=1,
                n_epochs=args.ep,
                n_steps=(args.n // args.workers) * horizon,
                learning_rate=args.lr,
                batch_size=args.bs,
                seed=args.seed,
                policy_kwargs=policy_kwargs,
                min_lr=args.lr,
                max_lr=args.lr,
                adaptive_kl=0.02,
                target_kl=0.1,
                )

    model.learn(
        total_timesteps=int(env_iter),
        callback=WandbCallback(
            model_save_freq=50,
            model_save_path=str(result_path / "model"),
            eval_env_fn=create_eval_env_fn,
            eval_freq=50,
            eval_cam_names=["relocate_viz"],
            viz_point_cloud=True,
        ),
    )
    wandb_run.finish()
