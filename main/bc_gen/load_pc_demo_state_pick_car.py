from pathlib import Path

import torch.nn as nn
import numpy as np
import transforms3d

from hand_env_utils.arg_utils import *
from hand_env_utils.wandb_callback import WandbCallback, setup_wandb
from hand_teleop.env.rl_env.base import compute_inverse_kinematics
from hand_teleop.kinematics.kinematics_helper import PartialKinematicModel
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO

import random
from datetime import datetime

def create_env(use_visual_obs=True, use_gui=False, obj_scale=1.0, obj_name=None,
               randomness_scale=1, pc_noise=False, is_eval=False):
    import os
    from hand_teleop.env.rl_env.free_pick_car_env import FreePickCarEnv
    from hand_teleop.real_world import task_setting
    from hand_teleop.env.sim_env.constructor import add_default_scene_light
    frame_skip = 5
    env_params = dict(object_scale=obj_scale, object_name=obj_name, use_gui=use_gui, frame_skip=frame_skip, no_rgb=True,
                      object_pc_sample=100, use_visual_obs=use_visual_obs)

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    env = FreePickCarEnv(**env_params)

    # Setup visual
    env.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate"])
    if pc_noise:
        env.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate_noise_pick"])
    else:
        env.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate"])

    if is_eval:
        env.setup_camera_from_config(task_setting.CAMERA_CONFIG["viz_only"])
        add_default_scene_light(env.scene, env.renderer)
    
    return env


import os, tqdm, pickle
from hoi4d_data.hoi4d_config import hoi4d_config
from hand_teleop.utils.hoi4d_object_utils import sample_hoi4d_object_pc
from hand_teleop.utils.munet import load_pretrained_munet

if __name__ == '__main__':
    model_path = "/data/lixing/results/state_toycar_f20-0.0/model/model_4950.zip"
    model = PPO.load(path=model_path, env=None)

    env = create_env(use_visual_obs=True, obj_scale=1.0, obj_name=("random", "random"), pc_noise=True)

    env.set_seed(0)

    data = []

    for iters in tqdm.tqdm(range(1000)):
        observations, actions = {"relocate-point_cloud": [], "state": []}, []
        obs = env.reset()

        # from sapien.utils import Viewer
        # from hand_teleop.env.sim_env.constructor import add_default_scene_light
        # viewer = Viewer(env.renderer)
        # viewer.set_scene(env.scene)
        # add_default_scene_light(env.scene, env.renderer)
        # env.viewer = viewer
        # viewer.toggle_pause(True)
        # env.render()

        for i in range(env.horizon):
            oracle_state = env.get_oracle_state()
            pc_obs = obs["relocate-point_cloud"]
            state_obs = obs["state"]
            action = model.policy.predict(observation=oracle_state, deterministic=True)[0]
            obs, _, _, _ = env.step(action)
            
            observations["relocate-point_cloud"].append(pc_obs)
            observations["state"].append(state_obs)
            actions.append(action)

            for _ in range(5):
                pass
                # env.render()

            dist = np.linalg.norm(env.target_in_object)
            if dist <= 0.05:
                observations["relocate-point_cloud"] = np.stack(observations["relocate-point_cloud"], axis=0)
                observations["state"] = np.stack(observations["state"], axis=0)
                actions = np.stack(actions, axis=0)
                trajectory = {"observations" : observations, "actions" : actions}
                data.append(trajectory)
                break
    
    save_file = open("/data/lixing/data/data-state-pick-car-f20.pkl", "wb")
    pickle.dump(data, save_file)
    save_file.close()
    print(len(data))