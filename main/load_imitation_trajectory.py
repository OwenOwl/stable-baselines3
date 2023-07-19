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


import os, tqdm, pickle
if __name__ == '__main__':
    model_list_path = "/home/lixing/results/result-0718"
    model_list = os.listdir(model_list_path)

    data = []
    
    for model_exp in tqdm.tqdm(model_list):
        model_args = model_exp.split("-")
        data_id = int(model_args[1][3:])
        obj_scale = 1.0
        obj_name = model_args[2]
        randomness = 1.0

        env = create_env(use_visual_obs=False, obj_scale=obj_scale, obj_name=obj_name,
                         data_id=data_id, randomness_scale=randomness)

        model_path = os.path.join(model_list_path, model_exp, "model/model_1000.zip")

        model = PPO.load(path=model_path, env=None)

        observations, actions = [], []
        obs = env.reset()
        for i in range(env.horizon):
            action = model.policy.predict(observation=obs, deterministic=True)[0]
            observations.append(obs)
            actions.append(action)
            obs, reward, done, _ = env.step(action)
        observations = np.stack(observations, axis=0)
        observations = np.delete(observations, 39, 1) # Delete timestamp observation
        actions = np.stack(actions, axis=0)
        trajectory = {"observations" : observations, "actions" : actions}
        data.append(trajectory)
    
    save_file = open(os.path.join(model_list_path, "data-0718.pkl"), "wb")
    pickle.dump(data, save_file)
    save_file.close()