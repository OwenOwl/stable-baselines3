import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import imageio
from hand_env_utils.arg_utils import *

from hand_env_utils.teleop_env import create_relocate_env
from hand_teleop.real_world.task_setting import IMG_CONFIG
from stable_baselines3.dapg import DAPG
from stable_baselines3.ppo import PPO

from hand_teleop.utils.camera_utils import fetch_texture

import cv2
import pathlib
import pickle

def create_env(use_gui=False, is_eval=False, obj_scale=1.0, obj_name="tomato_soup_can",
               object_pc_sample=0, pc_noise=False, **renderer_kwargs):
    import os
    from hand_teleop.env.rl_env.free_pick_bot_env import FreePickBotEnv
    from hand_teleop.real_world import task_setting
    from hand_teleop.env.sim_env.constructor import add_default_scene_light
    frame_skip = 5
    env_params = dict(object_scale=obj_scale, object_name=obj_name, use_gui=use_gui, object_pc_sample=object_pc_sample,
                      frame_skip=frame_skip, no_rgb=True, use_visual_obs=True)
    env_params.update(renderer_kwargs)

    if is_eval:
        env_params["no_rgb"] = False
        env_params["need_offscreen_render"] = True

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    env = FreePickBotEnv(**env_params)

    # Setup visual
    if not is_eval:
        env.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate"])

    if is_eval:
        config = task_setting.CAMERA_CONFIG["viz_only"].copy()
        config.update(task_setting.CAMERA_CONFIG["relocate"])
        env.setup_camera_from_config(config)
        add_default_scene_light(env.scene, env.renderer)

    if pc_noise:
        env.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate_noise_pick"])
    else:
        env.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate"])
    print(pc_noise)
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
    parser.add_argument('--exp', type=str, default="exp")
    parser.add_argument('--objscale', type=float, default=1.0)
    parser.add_argument('--objcat', type=str, default="random")
    parser.add_argument('--objname', type=str, default="random")
    parser.add_argument('--objpc', type=int, default=100)
    parser.add_argument('--use_bn', type=bool, default=True)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--noise_pc', type=bool, default=True)

    args = parser.parse_args()
    randomness = args.randomness
    now = datetime.now()
    exp_keywords = [args.exp, str(args.objcat), str(args.objname)]
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
        is_eval=True, use_gui=True, pc_noise=True
    )

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
    print(policy)
    print(policy.policy)
    # exit()

    print(env.observation_space)
    # viewer = env.render(mode="human")


    from sapien.utils import Viewer
    from hand_teleop.env.sim_env.constructor import add_default_scene_light

    # viewer = Viewer(env.renderer)
    # viewer.set_scene(env.scene)
    # add_default_scene_light(env.scene, env.renderer)
    # env.viewer = viewer
    # viewer.toggle_pause(True)

    viewer = env.render()

    root = "sim_trajs_flx"

    done = False
    manual_action = False
    action = np.zeros(22)
    traj_idx = 0
    # while not viewer.closed:
    for i in range(5):
        reward_sum = 0
        obs = env.reset()
        traj_idx += 1

        pathlib.Path(f"{root}/{traj_idx}/imgs").mkdir(parents=True, exist_ok=True)
        action_sequence = []
        obs_sequence = []
        frames = []
        for i in range(env.horizon):
            # print("Obs:", obs)
            if manual_action:
                action = np.concatenate([np.array([0, 0, 0.1, 0, 0, 0]), action[6:]])
            else:
                action = policy.predict(observation=obs, deterministic=True)[0]
            obs, reward, done, _ = env.step(action)
            action_sequence.append(action)
            obs_sequence.append(obs)
            # print(obs.keys())
            # for k,v in obs.items():
            #     print(k, v.shape)
            # print(action.shape)
            reward_sum += reward

            # for _ in range(5):
            #     pass
            env.render()

            print("Time Step:", env.control_time_step, env.scene.get_timestep())

            if env.viewer.window.key_down("enter"):
                manual_action = True
            elif env.viewer.window.key_down("p"):
                manual_action = False

            cam = env.cameras["relocate_viz"]
            cam.take_picture()
            img = fetch_texture(cam, "Color", return_torch=False)
            frames.append(img)
            if i % 10 == 0:
                cv2.imwrite(f"{root}/{traj_idx}/imgs/{str(i)}.png", img*255)
        with open(f"{root}/{traj_idx}/action_traj.pkl", "wb") as f:
            pickle.dump(action_sequence, f)
        with open(f"{root}/{traj_idx}/obs_traj.pkl", "wb") as f:
            pickle.dump(obs_sequence, f)

        output_filename = f"{root}/{traj_idx}/sim_traj.mp4"
        print(output_filename)
        imageio.mimsave(output_filename, frames, fps=20)
        print(f"Reward: {reward_sum}")