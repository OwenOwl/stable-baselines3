from pathlib import Path

import torch.nn as nn
import numpy as np

from hand_env_utils.arg_utils import *
from hand_env_utils.wandb_callback import WandbCallback, setup_wandb
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO

from datetime import datetime

def create_env(use_visual_obs, use_gui=False, is_eval=False,
               reward_args=np.zeros(5), randomness_scale=1, pc_noise=True):
    import os
    from hand_teleop.env.rl_env.imitation_env import ImitationEnv
    from hand_teleop.real_world import task_setting
    from hand_teleop.env.sim_env.constructor import add_default_scene_light
    frame_skip = 1
    env_params = dict(reward_args=reward_args, use_gui=use_gui, frame_skip=frame_skip, no_rgb=True)
    if is_eval:
        env_params["no_rgb"] = False 
        env_params["need_offscreen_render"] = True

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    env = ImitationEnv(**env_params)

    if use_visual_obs:
        raise NotImplementedError
    
    if is_eval:
        env.setup_camera_from_config(task_setting.CAMERA_CONFIG["viz_only"])
        add_default_scene_light(env.scene, env.renderer)
    
    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ep', type=int, default=10)
    parser.add_argument('--bs', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--iter', type=int, default=5000)
    parser.add_argument('--randomness', type=float, default=1.0)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--reward', type=float, nargs="+", default=[10,0.5,0.1,10,0.5])

    args = parser.parse_args()
    randomness = args.randomness
    now = datetime.now()
    exp_keywords = ["ppo", args.exp, ",".join(str(i) for i in args.reward)]
    horizon = 200
    env_iter = args.iter * horizon * args.n
    reward_args = args.reward
    assert(len(reward_args) >= 5)

    config = {
        'n_env_horizon': args.n,
        'update_iteration': args.iter,
        'total_step': env_iter,
        'randomness': randomness,
    }

    exp_name = "-".join(exp_keywords)
    result_path = Path("./results") / exp_name
    result_path.mkdir(exist_ok=True, parents=True)
    wandb_run = setup_wandb(config, "-".join([exp_name, now.strftime("(%Y/%m/%d,%H:%M)")]), tags=["state", "imitation"])

    def create_env_fn():
        environment = create_env(use_visual_obs=False, reward_args=reward_args, randomness_scale=randomness)
        return environment
    
    def create_eval_env_fn():
        environment = create_env(use_visual_obs=False, reward_args=reward_args, is_eval=True, randomness_scale=randomness)
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
            eval_freq=100,
            eval_cam_names=["relocate_viz"],
        ),
    )
    wandb_run.finish()