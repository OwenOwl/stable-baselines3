import os

from hand_teleop.env.rl_env.open_laptop_env import OpenLaptopRLEnv
from hand_teleop.real_world import task_setting
from hand_teleop.env.sim_env.constructor import add_default_scene_light


def create_open_laptop_env(object_name, use_visual_obs, object_category="HOI4D", use_gui=False, is_eval=False,
                           randomness_scale=1, pc_noise=True):
    if object_name == "laptop":
        robot_name = "allegro_hand_xarm6_wrist_mounted_face_front"
    else:
        print(object_name)
        raise NotImplementedError
    frame_skip = 10
    env_params = dict(robot_name=robot_name, use_gui=use_gui, frame_skip=frame_skip, no_rgb=True)
    if is_eval:
        env_params["no_rgb"] = False
        env_params["need_offscreen_render"] = True

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    env = OpenLaptopRLEnv(**env_params)

    if use_visual_obs:
        raise NotImplementedError
    
    if is_eval:
        env.setup_camera_from_config(task_setting.CAMERA_CONFIG["viz_only"])
        add_default_scene_light(env.scene, env.renderer)
    
    return env
