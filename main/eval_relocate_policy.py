import numpy as np

from hand_env_utils.teleop_env import create_relocate_env
from hand_teleop.real_world.task_setting import IMG_CONFIG
from stable_baselines3.dapg import DAPG
from stable_baselines3.ppo import PPO

if __name__ == '__main__':
    checkpoint_path = "results/ppo-mustard_bottle-pc_3e4_bs_1000-100/model/model_500.zip"
    img_type = None  # "robot", "goal_robot", "goal"
    use_visual_obs = True
    object_name = checkpoint_path.split("/")[1].split("-")[1]
    algorithm_name = checkpoint_path.split("/")[1].split("-")[0]
    env = create_relocate_env(object_name, use_visual_obs=use_visual_obs, use_gui=True)

    if img_type == "robot":
        env.setup_imagination_config(IMG_CONFIG["relocate_robot"])
    elif img_type == "goal":
        env.setup_imagination_config(IMG_CONFIG["relocate_goal"])
    elif img_type == "goal_robot":
        env.setup_imagination_config(IMG_CONFIG["relocate_goal_robot"])

    device = "cuda:0"
    if algorithm_name == "ppo":
        policy = PPO.load(checkpoint_path, env, device)
    elif algorithm_name == "dapg":
        policy = DAPG.load(checkpoint_path, env, device)
    else:
        raise NotImplementedError

    viewer = env.render(mode="human")

    done = False
    manual_action = False
    action = np.zeros(22)
    while not viewer.closed:
        reward_sum = 0
        obs = env.reset()
        for i in range(250):
            if manual_action:
                action = np.concatenate([np.array([0, 0, 0.1, 0, 0, 0]), action[6:]])
            else:
                action = policy.predict(observation=obs, deterministic=True)[0]
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
            env.render()
            if env.viewer.window.key_down("enter"):
                manual_action = True
            elif env.viewer.window.key_down("p"):
                manual_action = False

        print(f"Reward: {reward_sum}")
