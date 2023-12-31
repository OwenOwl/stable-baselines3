from pathlib import Path

import torch.nn as nn
import numpy as np

from hand_env_utils.arg_utils import *
from hand_env_utils.wandb_callback import WandbCallback, setup_wandb
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO

from datetime import datetime

def create_env(use_visual_obs, use_gui=False, is_eval=False, obj_scale=1.0, obj_name="tomato_soup_can",
               reward_args=np.zeros(3), randomness_scale=1, object_pc_sample=0, pc_noise=True):
    import os
    from hand_teleop.env.rl_env.free_safe_env import FreeSafeEnv
    from hand_teleop.real_world import task_setting
    from hand_teleop.env.sim_env.constructor import add_default_scene_light
    frame_skip = 5
    env_params = dict(object_scale=obj_scale, object_name=obj_name, use_gui=use_gui, object_pc_sample=object_pc_sample,
                      frame_skip=frame_skip, no_rgb=True)
    if is_eval:
        env_params["no_rgb"] = False 
        env_params["need_offscreen_render"] = True

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    env = FreeSafeEnv(**env_params)

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
    parser.add_argument('--objscale', type=float, default=1.0)
    parser.add_argument('--objcat', type=str, default="random")
    parser.add_argument('--objname', type=str, default="random")
    parser.add_argument('--objpc', type=int, default=100)
    parser.add_argument('--noise_pc', type=bool, default=True)

    args = parser.parse_args()
    randomness = args.randomness
    now = datetime.now()
    exp_keywords = [args.exp]
    horizon = 200
    env_iter = args.iter * horizon * args.n
    obj_scale = args.objscale
    obj_name = (args.objcat, args.objname)
    obj_pc_smp = args.objpc

    config = {
        'n_env_horizon': args.n,
        'update_iteration': args.iter,
        'total_step': env_iter,
        'randomness': randomness,
    }

    exp_name = "-".join(exp_keywords)
    result_path = Path("./results") / exp_name
    result_path.mkdir(exist_ok=True, parents=True)
    wandb_run = setup_wandb(config, "-".join([exp_name, now.strftime("(%Y/%m/%d,%H:%M)")]), tags=["state", "dapg"])

    def create_env_fn():
        environment = create_env(use_visual_obs=False, obj_scale=obj_scale, obj_name=obj_name, object_pc_sample=obj_pc_smp,
                                 randomness_scale=randomness)
        return environment
    
    def create_eval_env_fn():
        environment = create_env(use_visual_obs=False, obj_scale=obj_scale, obj_name=obj_name, object_pc_sample=obj_pc_smp,
                                 is_eval=True, randomness_scale=randomness)
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