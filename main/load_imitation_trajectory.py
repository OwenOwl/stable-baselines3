from pathlib import Path

import torch.nn as nn
import numpy as np

from hand_env_utils.arg_utils import *
from hand_env_utils.wandb_callback import WandbCallback, setup_wandb
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO

from datetime import datetime

def create_env(use_visual_obs, use_gui=False, is_eval=False, obj_scale=1.0, obj_name="tomato_soup_can",
               reward_args=np.zeros(3), data_id=0, randomness_scale=1, pc_noise=True):
    import os
    from hand_teleop.env.rl_env.imitation_pick_env import ImitationPickEnv
    from hand_teleop.real_world import task_setting
    from hand_teleop.env.sim_env.constructor import add_default_scene_light
    frame_skip = 1
    env_params = dict(reward_args=reward_args, object_scale=obj_scale, object_name=obj_name, data_id=data_id,
                      use_gui=use_gui, frame_skip=frame_skip, no_rgb=True)
    if is_eval:
        env_params["no_rgb"] = False 
        env_params["need_offscreen_render"] = True

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    env = ImitationPickEnv(**env_params)

    if use_visual_obs:
        raise NotImplementedError
    
    if is_eval:
        env.setup_camera_from_config(task_setting.CAMERA_CONFIG["viz_only"])
        add_default_scene_light(env.scene, env.renderer)
    
    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--randomness', type=float, default=1.0)
    parser.add_argument('--reward', type=float, nargs="+", default=[1, 0.05, 0.01])
    parser.add_argument('--objscale', type=float, default=1.0)
    parser.add_argument('--objname', type=str, default="tomato_soup_can")
    parser.add_argument('--dataid', type=int, default=0)

    args = parser.parse_args()
    randomness = args.randomness
    horizon = 200
    reward_args = args.reward
    data_id = args.dataid
    assert(len(reward_args) >= 3)
    obj_scale = args.objscale
    obj_name = args.objname

    env = create_env(use_visual_obs=False, obj_scale=obj_scale, obj_name=obj_name,
                     data_id=data_id, randomness_scale=randomness)
    
    model_path = "/home/lixing/results/result-test/model_test.zip"

    model = PPO.load(path=model_path, env=None)

    obs = env.reset()
    for i in range(env.horizon):
        action = model.policy.predict(observation=obs, deterministic=True)[0]
        obs, reward, done, _ = env.step(action)
        print(reward)