import numpy as np
import sapien.core as sapien

from hand_env_utils.teleop_env import create_relocate_env
from hand_teleop.real_world import task_setting


def main():
    object_name = "mustard_bottle"
    img_type = "goal_robot"

    if img_type == "robot":
        img_config = task_setting.IMG_CONFIG["relocate_robot_only"]
        imagination_keys = ("imagination_robot",)
    elif img_type == "goal":
        img_config = task_setting.IMG_CONFIG["relocate_goal_only"]
        imagination_keys = ("imagination_goal",)
    elif img_type == "goal_robot":
        img_config = task_setting.IMG_CONFIG["relocate_goal_robot"]
        imagination_keys = ("imagination_goal", "imagination_robot")
    else:
        raise NotImplementedError

    env = create_relocate_env(object_name, use_visual_obs=True, use_gui=True, is_eval=True)
    env.setup_imagination_config(img_config)

    goal_pose = np.random.rand(7)
    goal_pose_in_robot = sapien.Pose(goal_pose[:3], goal_pose[3:7])
    goal_pose_world = env.robot.get_pose() * goal_pose_in_robot
    env.target_pose = goal_pose_world
    env.target_object.set_pose(goal_pose_world)
    env.update_imagination(reset_goal=True)

    for i in range(200):
        qpos = np.random.rand(22)
        env.robot.set_qpos(qpos)
        env.update_cached_state()
        env.update_imagination(reset_goal=False)
        obs = env.get_observation()
        points = np.random.rand(256, 3)
        obs_to_network = {"relocate-point_cloud": points}
        for key in imagination_keys:
            obs_to_network[key] = obs[key]


if __name__ == '__main__':
    pass
