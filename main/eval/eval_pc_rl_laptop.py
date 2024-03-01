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
from hand_teleop.utils.hoi4d_object_utils import HOI4D_OBJECT_LIST

import random
from datetime import datetime

def create_lab_env(use_visual_obs, use_gui=False, obj_scale=1.0, friction=1, obj_name="tomato_soup_can",
                   randomness_scale=1, pc_noise=True):
    import os
    from hand_teleop.env.rl_env.free_laptop_env import FreeLaptopEnv
    from hand_teleop.real_world import task_setting
    from hand_teleop.env.sim_env.constructor import add_default_scene_light
    frame_skip = 5
    env_params = dict(object_scale=obj_scale, object_name=obj_name, friction=friction,
                      use_gui=use_gui, frame_skip=frame_skip, no_rgb=True, use_visual_obs=use_visual_obs, object_pc_sample=100)

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    env = FreeLaptopEnv(**env_params)

    # Setup visual
    env.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate"])

    if pc_noise:
        env.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate_noise"])
    else:
        env.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate"])
    
    if use_gui:
        config = task_setting.CAMERA_CONFIG["viz_only"].copy()
        config.update(task_setting.CAMERA_CONFIG["relocate"])
        env.setup_camera_from_config(config)
        add_default_scene_light(env.scene, env.renderer)
    
    return env


import os, tqdm, pickle
from hoi4d_data.hoi4d_config import hoi4d_config
from hand_teleop.utils.hoi4d_object_utils import sample_hoi4d_object_pc
from hand_teleop.utils.munet import load_pretrained_munet
from hand_teleop.utils.camera_utils import fetch_texture
import cv2

if __name__ == '__main__':
    f = open("results/eval/pc_rl_laptop.txt", "w")

    model_path = "/home/lixing/results/pc_rl_laptop/model/model_0.zip"
    
    object_list = HOI4D_OBJECT_LIST['laptop']

    pointnet = load_pretrained_munet()

    succeed = 0
    seed = 0
    
    for friction in [1, 0.7, 0.5, 0.2]:
        for scale in [0.5, 0.75, 1, 1.25, 1.5]:
            for ITERS in range(5):
                succeed = 0
                for (object_cat, object_name) in tqdm.tqdm(object_list):
                    randomness = 1.0

                    lab_env = create_lab_env(use_visual_obs=True, obj_scale=scale, friction=friction, obj_name=(object_cat, object_name),
                                            randomness_scale=randomness)
                    
                    lab_env.rl_step = lab_env.ability_arm_sim_step

                    seed += 1

                    lab_env.set_seed(seed)

                    lab_obs = lab_env.reset()

                    from sapien.utils import Viewer
                    from hand_teleop.env.sim_env.constructor import add_default_scene_light

                    # viewer = Viewer(lab_env.renderer)
                    # viewer.set_scene(lab_env.scene)
                    # add_default_scene_light(lab_env.scene, lab_env.renderer)
                    # lab_env.viewer = viewer
                    # viewer.toggle_pause(True)

                    model = PPO.load(path=model_path, env=None)
                    
                    # lab_env.render()

                    for i in range(lab_env.horizon):
                        lab_action = model.policy.predict(lab_obs, deterministic=True)[0]
                        lab_obs, lab_reward, _, _ = lab_env.step(lab_action)

                        for _ in range(5):
                            pass
                            # lab_env.render()

                        if lab_env.object.get_qpos()[0] >= np.pi / 12 * 5:
                            succeed += 1
                            break
                
                print(succeed, " / ", len(object_list))
                f.write("%.3f " % (succeed / len(object_list)))
            f.write("\n")

    f.close()