from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from hand_env_utils.arg_utils import *
from stable_baselines3.common.vec_env.hand_teleop_vec_env import HandTeleopVecEnv


def create_env(use_gui=False, is_eval=False, obj_scale=1.0, obj_name="tomato_soup_can",
               object_pc_sample=0, pc_noise=False, **renderer_kwargs):
    import os
    from hand_teleop.env.rl_env.free_pick_env import FreePickEnv
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
    env = FreePickEnv(**env_params)

    # Setup visual
    env.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate"])
    if pc_noise:
        env.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate_noise"])
    else:
        env.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate"])

    if is_eval:
        env.setup_camera_from_config(task_setting.CAMERA_CONFIG["viz_only"])
        add_default_scene_light(env.scene, env.renderer)

    return env


def viz_pc(point_cloud: torch.Tensor):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(point_cloud.cpu().numpy())
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pc, coord])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ep', type=int, default=10)
    parser.add_argument('--bs', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--iter', type=int, default=5000)
    parser.add_argument('--randomness', type=float, default=1.0)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--objscale', type=float, default=1.0)
    parser.add_argument('--objcat', type=str, default="toycar")
    parser.add_argument('--objname', type=str, default="035")
    parser.add_argument('--objpc', type=int, default=100)
    parser.add_argument('--dataset_path', type=str)
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

    exp_name = "-".join(exp_keywords)
    result_path = Path("./results") / exp_name
    result_path.mkdir(exist_ok=True, parents=True)


    def create_eval_env_fn():
        environment = create_env(use_visual_obs=False, obj_scale=obj_scale, obj_name=obj_name,
                                 object_pc_sample=obj_pc_smp, is_eval=True)
        return environment


    make_env_fn = partial(create_env, obj_scale=obj_scale, obj_name=obj_name, object_pc_sample=obj_pc_smp)
    env = HandTeleopVecEnv([make_env_fn] * args.workers)
    print(env.observation_space, env.action_space)

    obs = env.reset()
    viz_pc(obs["relocate-point_cloud"][1, :, :])
    for key, value in obs.items():
        print(key, value.shape)
    for _ in range(100):
        action = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        actions = np.tile(action, [args.workers, 1])
        obs, reward, done, info = env.step(actions)
    viz_pc(obs["relocate-point_cloud"][1, :, :])
