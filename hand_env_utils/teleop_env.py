import os

from hand_teleop.env.rl_env.relocate_env import LabArmAllegroRelocateRLEnv
from hand_teleop.real_world import task_setting
from hand_teleop.env.sim_env.constructor import add_default_scene_light


def create_relocate_env(object_name, use_visual_obs, object_category="YCB", use_gui=False, is_eval=False,
                        randomness_scale=1, pc_noise=True):
    if object_name == "mustard_bottle":
        robot_name = "allegro_hand_xarm6_wrist_mounted_face_front"
    elif object_name in ["tomato_soup_can", "potted_meat_can"]:
        robot_name = "allegro_hand_xarm6_wrist_mounted_face_down"
    elif object_category == "egad":
        robot_name = "allegro_hand_xarm6_wrist_mounted_face_down"
    elif object_category.isnumeric() and object_category == "02876657":
        robot_name = "allegro_hand_xarm6_wrist_mounted_face_front"
    elif object_category.isnumeric() and object_category == "02946921":
        robot_name = "allegro_hand_xarm6_wrist_mounted_face_down"
    else:
        print(object_name)
        raise NotImplementedError
    rotation_reward_weight = 1
    frame_skip = 10
    env_params = dict(object_name=object_name, robot_name=robot_name, rotation_reward_weight=rotation_reward_weight,
                      randomness_scale=randomness_scale, use_visual_obs=use_visual_obs, use_gui=use_gui, no_rgb=True,
                      object_category=object_category, frame_skip=frame_skip)
    if is_eval:
        env_params["no_rgb"] = False
        env_params["need_offscreen_render"] = True

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    env = LabArmAllegroRelocateRLEnv(**env_params)

    if use_visual_obs:
        # Create camera and setup visual modality
        env.setup_camera_from_config(task_setting.CAMERA_CONFIG["relocate"])
        if pc_noise:
            env.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate_noise"])
        else:
            env.setup_visual_obs_config(task_setting.OBS_CONFIG["relocate"])

    if is_eval:
        env.setup_camera_from_config(task_setting.CAMERA_CONFIG["viz_only"])
        add_default_scene_light(env.scene, env.renderer)

    return env
