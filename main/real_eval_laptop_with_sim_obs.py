import pickle
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

from hand_env_utils.arg_utils import *

from hand_env_utils.teleop_env import create_relocate_env
from hand_teleop.real_world.task_setting import IMG_CONFIG
from stable_baselines3.dapg import DAPG
from stable_baselines3.ppo import PPO

from hand_teleop.utils.camera_utils import fetch_texture

import cv2
import pathlib
import open3d as o3d

def create_env(
        use_gui=False, is_eval=False, obj_scale=1.0, obj_name="tomato_soup_can",
        object_pc_sample=0, pc_noise=False, **renderer_kwargs
):
    import os

    from hand_teleop.real_dexmv2.real_laptop import RealAbilityXArmLaptopEnv
    from hand_teleop.real_world import task_setting
    from hand_teleop.env.sim_env.constructor import add_default_scene_light
    frame_skip = 5
    env_params = dict(
        object_scale=obj_scale, object_name=obj_name, 
        use_gui=use_gui, object_pc_sample=object_pc_sample,
        frame_skip=frame_skip, no_rgb=True, use_visual_obs=True
    )
    env_params.update(renderer_kwargs)

    if is_eval:
        env_params["no_rgb"] = False
        env_params["need_offscreen_render"] = True

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"

    env_params["xarm_ip"] = "192.168.1.242"
    env_params["hand_address"] = 0x50
    env_params["hand_smooth"] = 0.9975
    env_params["hand_port"] = "/dev/ttyUSB0"
    env = RealAbilityXArmLaptopEnv(**env_params)

    # Setup visual
    if not is_eval:
        env.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate"])

    if is_eval:
        config = task_setting.CAMERA_CONFIG["viz_only"].copy()
        config.update(task_setting.CAMERA_CONFIG["relocate"])
        env.setup_camera_from_config(config)
        add_default_scene_light(env.scene, env.renderer)

    if pc_noise:
        env.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate_noise"])
    else:
        env.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate"])

    return env


def viz_pc(point_cloud: torch.Tensor):
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(point_cloud.cpu().numpy())
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pc, coord])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ep', type=int, default=10)
    parser.add_argument('--bs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--iter', type=int, default=1000)
    parser.add_argument('--randomness', type=float, default=1.0)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--objscale', type=float, default=1.0)
    parser.add_argument('--objcat', type=str, default="toycar")
    parser.add_argument('--objname', type=str, default="035")
    parser.add_argument('--objpc', type=int, default=100)
    parser.add_argument('--use_bn', type=bool, default=True)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--noise_pc', type=bool, default=True)

    args = parser.parse_args()
    randomness = args.randomness
    now = datetime.now()
    exp_keywords = [args.exp, str(args.objcat), str(args.objname)]
    horizon = 200
    env_iter = args.iter * horizon * args.n
    obj_scale = args.objscale


    from hand_teleop.utils.hoi4d_object_utils import HOI4D_OBJECT_LIST
    object_list = HOI4D_OBJECT_LIST['laptop']
    obj_name = object_list[0]
    obj_pc_smp = args.objpc

    config = {
        'n_env_horizon': args.n,
        'update_iteration': args.iter,
        'total_step': env_iter,
        'randomness': randomness,
    }

    checkpoint_path = args.checkpoint_path
    use_visual_obs = True
    # if "imagination" not in checkpoint_path:
    #     object_name = checkpoint_path.split("/")[-3].split("-")[1]
    # else:
    #     object_name = checkpoint_path.split("/")[-3].split("-")[2]

    # algorithm_name = checkpoint_path.split("/")[-3].split("-")[0]
    env = create_env(
        obj_scale=obj_scale, obj_name=obj_name, 
        object_pc_sample=obj_pc_smp,
        is_eval=True, use_gui=True, pc_noise=False
    )

    env.reset()
    # if use_visual_obs:
    #     if "imagination-goal_robot" in checkpoint_path:
    #         img_type = "goal_robot"
    #         env.setup_imagination_config(IMG_CONFIG["relocate_goal_robot"])
    #     elif "imagination-goal" in checkpoint_path:
    #         img_type = "goal"
    #         env.setup_imagination_config(IMG_CONFIG["relocate_goal"])
    #     elif "imagination-robot" in checkpoint_path:
    #         img_type = "robot"
    #         env.setup_imagination_config(IMG_CONFIG["relocate_robot"])
    #     else:
    #         img_type = None

    device = "cuda:0"
    # if "ppo" in algorithm_name:
    #     policy = PPO.load(checkpoint_path, env, device)
    # elif "dapg" in algorithm_name:
    policy = DAPG.load(checkpoint_path, env, device)
    # else:
    #     raise NotImplementedError
    # print(policy)
    # print(policy.policy)
    # exit()

    print(env.observation_space)
    # viewer = env.render(mode="human")


    from sapien.utils import Viewer
    from hand_teleop.env.sim_env.constructor import add_default_scene_light

    traj_root = "sim_trajs_0314_laptop/001"
    traj_idx = 5


    with open(f"{traj_root}/{traj_idx}/action_traj.pkl", "rb") as f:
        traj = pickle.load(f)

    with open(f"{traj_root}/{traj_idx}/obs_traj.pkl", "rb") as f:
        obs_traj = pickle.load(f)

    # viewer = env.render()
    env.reset()

    done = False
    manual_action = False
    action = np.zeros(22)
    # while True:
    reward_sum = 0
    obs_real = env.reset()

    print(obs_real.keys())


    root = "real_trajs_0311_laptop"

    obs_sequence = []
    action_sequence = []

    traj_idx = 4
    pathlib.Path(f"{root}/{traj_idx}/pcs").mkdir(parents=True, exist_ok=True)

    time_idx = -1
    # pathlib.Path(f"temp_pics/{traj_idx}").mkdir(parents=True, exist_ok=True)
    for action, obs in zip(traj, obs_traj):
        time_idx += 1
        # print("Obs", obs)
        with open(f"{root}/{traj_idx}/pcs/{time_idx}.pkl", "wb") as f:
            pickle.dump(obs_real, f)

        # obs_real["relocate-point_cloud"] = obs["relocate-point_cloud"]
        # print(obs_real["state"].shape)

        # fake_hand_state = np.array([
        #     0.27, 0.27, 0.27, 0.27, -0.27, 0.27, 
        # ])
        print(obs_real["state"][7:13])

        obs_real["state"] = np.concatenate([
                # obs["state"][:7],
                obs_real["state"][:7],
                # obs["state"][7:13],
                obs_real["state"][7:13],
                # fake_hand_state,
                obs_real["state"][13:]
        ])
        if False:
            new_points_for_visual = obs_real["relocate-point_cloud"][:,:3]
            # new_points_for_visual = new_points_for_visual[new_points_for_visual[:, 2] > 0.2]
            # print("-" * 100)
            # for p_i in range(512):
            #     print(new_points_for_visual[p_i])
            camera_obs = o3d.geometry.PointCloud()
            camera_obs.points = o3d.utility.Vector3dVector(new_points_for_visual)
            # camera_obs.paint_uniform_color([0, 0, 1])
            
            # img_pc_for_visual = img_pc[:,:3]
            # img = o3d.geometry.PointCloud()
            # img.points = o3d.utility.Vector3dVector(img_pc_for_visual)
            # img.paint_uniform_color([1, 0, 0])
            coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])

            o3d.visualization.draw_geometries([camera_obs, coordinate])
        if manual_action:
            action = np.concatenate([np.array([1, 1, 1, 0, 0, 0]), action[6:]])
        else:
            action = policy.predict(observation=obs_real, deterministic=True)[0]
        print("action:", action)
        obs_real, reward, done, _ = env.step(action)
        # print("Robot State Sim:", obs["state"])
        # print("Robot State Real:", obs_real["state"])
        # print("Hand State Sim:", obs["state"][7:13])
        # print("Hand State Real:", obs_real["state"][7:13])
        print("-" * 100)
        # print(obs.keys())
        # for k,v in obs.items():
        #     print(k, v.shape)
        # print(action.shape)
        action_sequence.append(action)
        obs_sequence.append(obs_real)
        reward_sum += reward

        # for _ in range(5):
        #     pass
        # env.render()

        # if env.viewer.window.key_down("enter"):
        #     manual_action = True
        # elif env.viewer.window.key_down("p"):
        #     manual_action = False

        
        # if i % 10 == 0:
        #     cam = env.cameras["relocate_viz"]
        #     cam.take_picture()
        #     img = fetch_texture(cam, "Color", return_torch=False)
        #     cv2.imwrite(f"temp_pics/{traj_idx}/{str(i)}.png", img*255)
        # with open(f"{root}/{traj_idx}/pcs/{time_idx}.pkl", "wb") as f:
        #     pickle.dump(obs_real, f)

    print(f"Reward: {reward_sum}")
    with open(f"{root}/{traj_idx}/action_traj.pkl", "wb") as f:
        pickle.dump(action_sequence, f)
    with open(f"{root}/{traj_idx}/obs_traj.pkl", "wb") as f:
        pickle.dump(obs_sequence, f)

    env.stop()