from datetime import datetime
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn

from hand_env_utils.arg_utils import *
from hand_env_utils.wandb_callback import WandbCallback, setup_wandb
from stable_baselines3.common.torch_layers import PointNetStateExtractor
from stable_baselines3.common.vec_env.hand_teleop_vec_env import HandTeleopVecEnv
from stable_baselines3.dapg import Dagger
from stable_baselines3.ppo import PPO


def create_env(use_gui=False, is_eval=False, obj_scale=1.0, obj_name="tomato_soup_can",
               object_pc_sample=0, pc_noise=True, **renderer_kwargs):
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
    parser.add_argument('--objcat', type=str, default="random")
    parser.add_argument('--objname', type=str, default="random")
    parser.add_argument('--objpc', type=int, default=100)
    parser.add_argument('--use_bn', type=bool, default=True)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--model_path', type=str)
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
    wandb_run = setup_wandb(config, "-".join([exp_name, now.strftime("(%Y/%m/%d,%H:%M)")]),
                            tags=["point_cloud", "dapg"])

    make_env_fn = partial(create_env, obj_scale=obj_scale, obj_name=obj_name, object_pc_sample=obj_pc_smp)
    make_env_fn_eval = partial(create_env, obj_scale=obj_scale, obj_name=obj_name, object_pc_sample=obj_pc_smp,
                               is_eval=True)
    env = HandTeleopVecEnv([make_env_fn] * args.workers)
    print(env.observation_space, env.action_space)

    feature_extractor_class = PointNetStateExtractor
    feature_extractor_kwargs = {
        "pc_key": "relocate-point_cloud",
        "local_channels": (64, 128, 256),
        "global_channels": (256,),
        "use_bn": args.use_bn,
        "state_mlp_size": (64, 64),
    }
    policy_kwargs = {
        "features_extractor_class": feature_extractor_class,
        "features_extractor_kwargs": feature_extractor_kwargs,
        "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
        "activation_fn": nn.ReLU,
    }

    state_model = PPO.load(path=args.model_path, env=None)

    model = Dagger("PointCloudPolicy", env, verbose=1,
                 dataset_path=args.dataset_path,
                 state_model=state_model,
                 bc_coef=0.002,
                 bc_decay=1,
                 bc_batch_size=500,
                 n_epochs=args.ep,
                 n_steps=(args.n // args.workers) * horizon,
                 learning_rate=args.lr,
                 batch_size=args.bs,
                 seed=args.seed,
                 policy_kwargs=policy_kwargs,
                 tensorboard_log=str(result_path / "log"),
                 min_lr=args.lr,
                 max_lr=args.lr,
                 adaptive_kl=0.02,
                 target_kl=0.5,
                 )

    model.learn(
        total_timesteps=int(env_iter),
        bc_init_epoch=100,
        bc_init_batch_size=500,
        callback=WandbCallback(
            model_save_freq=50,
            model_save_path=str(result_path / "model"),
            eval_env_fn=make_env_fn_eval,
            eval_freq=10,
            eval_cam_names=["relocate_viz"],
        ),
    )
    wandb_run.finish()
