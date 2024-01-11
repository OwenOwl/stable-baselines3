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

def create_lab_env(use_visual_obs, use_gui=True, obj_scale=1.0, obj_name="tomato_soup_can",
                   obj_init_orientation=np.array([1, 0, 0, 0]), randomness_scale=1, pc_noise=True):
    import os
    from hand_teleop.env.rl_env.free_laptop_env import FreeLaptopEnv
    from hand_teleop.real_world import task_setting
    from hand_teleop.env.sim_env.constructor import add_default_scene_light
    frame_skip = 5
    env_params = dict(object_scale=obj_scale, object_name=obj_name, obj_init_orientation=obj_init_orientation,
                      use_gui=use_gui, frame_skip=frame_skip, no_rgb=True)

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    env = FreeLaptopEnv(**env_params)

    if use_visual_obs:
        raise NotImplementedError
    
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
    SAMPLE_OBJECT_PC_NUM = 100
    EMB_DIM = 32
    # model_path = "/home/lixing/results/state_laptop-0.002/model/model_500.zip"
    model_path = "/home/lixing/results/rl_laptop/model/model_500.zip"
    object_list = HOI4D_OBJECT_LIST['laptop']

    pointnet = load_pretrained_munet()

    succeed = 0
    seed = 0
    
    for (object_cat, object_name) in tqdm.tqdm(object_list[:1]):
        randomness = 1.0

        object_pc = sample_hoi4d_object_pc((object_cat, object_name), SAMPLE_OBJECT_PC_NUM)

        lab_env = create_lab_env(use_visual_obs=False, obj_scale=1.0, obj_name=(object_cat, object_name),
                                obj_init_orientation=np.array([0, 0, 0, 1]), randomness_scale=randomness)
        
        object_emb = pointnet.get_embedding(object_pc)
        
        lab_env.rl_step = lab_env.ability_arm_sim_step

        seed += 1

        lab_env.set_seed(seed)

        lab_obs = lab_env.reset()

        from sapien.utils import Viewer
        from hand_teleop.env.sim_env.constructor import add_default_scene_light

        viewer = Viewer(lab_env.renderer)
        viewer.set_scene(lab_env.scene)
        add_default_scene_light(lab_env.scene, lab_env.renderer)
        lab_env.viewer = viewer
        viewer.toggle_pause(True)

        model = PPO.load(path=model_path, env=None)
        
        lab_env.render()

        for i in range(lab_env.horizon):
            lab_action = model.policy.predict(observation=np.concatenate([lab_obs, object_emb]), deterministic=True)[0]
            lab_obs, lab_reward, _, _ = lab_env.step(lab_action)

            for _ in range(5):
                pass
                lab_env.render()
            
            if i % 10 == 0:
                cam = lab_env.cameras["relocate_viz"]
                cam.take_picture()
                img = fetch_texture(cam, "Color", return_torch=False)
                cv2.imwrite("temp_pics/"+model_path.split('/')[4]+'_'+object_name+'_'+str(i)+".png", img*255)
    
    print(succeed, len(object_list))