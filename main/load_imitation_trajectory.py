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

from datetime import datetime

def create_env(use_visual_obs, use_gui=True, obj_scale=1.0, obj_name="tomato_soup_can",
               data_id=0, randomness_scale=1, pc_noise=True):
    import os
    from hand_teleop.env.rl_env.imitation_pick_env import ImitationPickEnv
    from hand_teleop.real_world import task_setting
    from hand_teleop.env.sim_env.constructor import add_default_scene_light
    frame_skip = 5
    env_params = dict(reward_args=np.zeros(3), object_scale=obj_scale, object_name=obj_name, data_id=data_id,
                      use_gui=use_gui, frame_skip=frame_skip, no_rgb=True)

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    env = ImitationPickEnv(**env_params)

    if use_visual_obs:
        raise NotImplementedError
    
    return env

def create_lab_env(use_visual_obs, use_gui=True, obj_scale=1.0, obj_name="tomato_soup_can",
                   obj_init_orientation=np.array([1, 0, 0, 0]), randomness_scale=1, pc_noise=True):
    import os
    from hand_teleop.env.rl_env.free_pick_env import FreePickEnv
    from hand_teleop.real_world import task_setting
    from hand_teleop.env.sim_env.constructor import add_default_scene_light
    frame_skip = 5
    env_params = dict(object_scale=obj_scale, object_name=obj_name, obj_init_orientation=obj_init_orientation,
                      use_gui=use_gui, frame_skip=frame_skip, no_rgb=True)

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    env = FreePickEnv(**env_params)

    if use_visual_obs:
        raise NotImplementedError
    
    return env


import os, tqdm, pickle
from hoi4d_data.hoi4d_config import hoi4d_config
from hand_teleop.utils.hoi4d_object_utils import sample_hoi4d_object_pc
from hand_teleop.utils.object_embedding import MuNet

if __name__ == '__main__':
    SAMPLE_OBJECT_PC_NUM = 100
    EMB_DIM = 32
    model_list_path = "/home/lixing/results/result-1018"
    model_list = os.listdir(model_list_path)

    pointnet = MuNet(obs_dim=2, act_dim=2, pc_dim=SAMPLE_OBJECT_PC_NUM, emb_dim=EMB_DIM)

    data = []
    
    for model_exp in tqdm.tqdm(model_list[:5]):
        model_args = model_exp.split("-")
        data_id = int(model_args[1])
        randomness = 1.0

        object_cat = hoi4d_config.data_list[data_id]["obj_cat"]
        object_name = hoi4d_config.data_list[data_id]["seq_path"].split('/')[3].replace('N', '0')

        object_pc = sample_hoi4d_object_pc((object_cat, object_name), SAMPLE_OBJECT_PC_NUM)
        object_emb = np.zeros(EMB_DIM) ### TODO

        env = create_env(use_visual_obs=False, obj_scale=1.0, obj_name=(object_cat, object_name),
                         data_id=data_id, randomness_scale=randomness)
        lab_env = create_lab_env(use_visual_obs=False, obj_scale=1.0, obj_name=(object_cat, object_name),
                                 obj_init_orientation=env.init_orientation, randomness_scale=randomness)
        
        env.rl_step = env.ability_sim_step_deterministic
        lab_env.rl_step = lab_env.ability_arm_sim_step

        env.set_seed(1)
        lab_env.set_seed(1)

        observations, actions = [], []
        obs = env.reset()
        lab_obs = lab_env.reset()

        from sapien.utils import Viewer
        from hand_teleop.env.sim_env.constructor import add_default_scene_light

        # viewer = Viewer(lab_env.renderer)
        # viewer.set_scene(lab_env.scene)
        # add_default_scene_light(lab_env.scene, lab_env.renderer)
        # lab_env.viewer = viewer
        # viewer.toggle_pause(True)

        # viewer = Viewer(env.renderer)
        # viewer.set_scene(env.scene)
        # add_default_scene_light(env.scene, env.renderer)
        # env.viewer = viewer
        # viewer.toggle_pause(True)

        model_path = os.path.join(model_list_path, model_exp, "model/model_1950.zip")
        model = PPO.load(path=model_path, env=None)

        # IK Initial xarm pose by pinocchio
        lab_IK_model = lab_env.robot.create_pinocchio_model()
        lab_PK_model = PartialKinematicModel(lab_env.robot, 'joint1', 'joint6')
        link_name2id = {lab_env.robot.get_links()[i].get_name(): i for i in range(len(lab_env.robot.get_links()))}
        ee_link_id = link_name2id[lab_env.robot_info.palm_name]
        lab_pose_inv = lab_env.robot.get_pose().inv()
        palm_pose = lab_pose_inv * env.palm_link.get_pose()
        lab_init_qpos = np.zeros(lab_env.robot.dof)
        xarm_init_qpos = lab_env.robot_info.arm_init_qpos
        lab_init_qpos[:lab_env.arm_dof] = xarm_init_qpos
        arm_qmask = np.concatenate([np.ones(lab_env.arm_dof), np.zeros(lab_env.robot.dof-lab_env.arm_dof)])
        lab_qpos = lab_IK_model.compute_inverse_kinematics(ee_link_id, palm_pose, lab_init_qpos, arm_qmask)
        lab_env.robot.set_qpos(np.concatenate([lab_qpos[0][:lab_env.arm_dof], env.robot.get_qpos()[6:]]))
        lab_env.robot.set_drive_target(lab_env.robot.get_qpos())
        
        # lab_env.render()
        # env.render()

        for i in range(env.horizon):
            action = model.policy.predict(observation=obs, deterministic=True)[0]
            obs, reward, done, _ = env.step(action)
            # import ipdb; ipdb.set_trace()
            
            palm_next_pose = lab_pose_inv * env.palm_link.get_pose()
            palm_delta_pose = palm_pose.inv() * palm_next_pose
            delta_axis, delta_angle = transforms3d.quaternions.quat2axangle(palm_delta_pose.q)
            if delta_angle > np.pi:
                delta_angle = 2 * np.pi - delta_angle
                delta_axis = -delta_axis
            delta_axis_world = palm_pose.to_transformation_matrix()[:3, :3] @ delta_axis
            delta_pose = np.concatenate([palm_next_pose.p - palm_pose.p, delta_axis_world * delta_angle])
            # print(delta_pose)

            hand_qpos_action = action[6:]
            lab_action = np.concatenate([delta_pose / env.scene.get_timestep() / env.frame_skip, hand_qpos_action])

            observations.append(np.concatenate([lab_obs, object_emb]))
            actions.append(lab_action)
            
            pose1 = lab_env.palm_link.get_pose()
            lab_obs, lab_reward, _, _ = lab_env.step(lab_action)
            pose2 = lab_env.palm_link.get_pose()

            # print(env.robot.get_qpos()[6:] - lab_env.robot.get_qpos()[6:])
            # print((pose2.p - pose1.p) - delta_pose[:3])
            # print(env.palm_link.get_pose().inv() * lab_env.palm_link.get_pose())

            for _ in range(5):
                pass
                # lab_env.render()
                # env.render()
            
            palm_pose = lab_pose_inv * lab_env.palm_link.get_pose()
        
        observations = np.stack(observations, axis=0)
        actions = np.stack(actions, axis=0)
        trajectory = {"observations" : observations, "actions" : actions}
        if (lab_reward > 2):
            data.append(trajectory)
    
    save_file = open(os.path.join(model_list_path, "data.pkl"), "wb")
    pickle.dump(data, save_file)
    save_file.close()
    print(len(data))