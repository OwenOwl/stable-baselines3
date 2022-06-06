import numpy as np

from hand_env_utils.teleop_env import create_relocate_env
from hand_teleop.real_world.task_setting import IMG_CONFIG
from stable_baselines3.dapg import DAPG
from stable_baselines3.ppo import PPO

if __name__ == '__main__':
    checkpoint_path = "results/dapg_pc-mustard_bottle-test_state_pc-200/model/model_0.zip"
    use_visual_obs = True
    if "imagination" not in checkpoint_path:
        object_name = checkpoint_path.split("/")[-3].split("-")[1]
    else:
        object_name = checkpoint_path.split("/")[-3].split("-")[2]

    algorithm_name = checkpoint_path.split("/")[-3].split("-")[0]
    env = create_relocate_env(object_name, use_visual_obs=use_visual_obs, use_gui=True)

    if use_visual_obs:
        if "imagination-goal_robot" in checkpoint_path:
            img_type = "goal_robot"
            env.setup_imagination_config(IMG_CONFIG["relocate_goal_robot"])
        elif "imagination-goal" in checkpoint_path:
            img_type = "goal"
            env.setup_imagination_config(IMG_CONFIG["relocate_goal"])
        elif "imagination-robot" in checkpoint_path:
            img_type = "robot"
            env.setup_imagination_config(IMG_CONFIG["relocate_robot"])
        else:
            img_type = None

    device = "cuda:0"
    if "ppo" in algorithm_name:
        policy = PPO.load(checkpoint_path, env, device)
    elif "dapg" in algorithm_name:
        policy = DAPG.load(checkpoint_path, env, device)
    else:
        raise NotImplementedError

    print(env.observation_space)
    viewer = env.render(mode="human")

    done = False
    manual_action = False
    action = np.zeros(22)
    while not viewer.closed:
        reward_sum = 0
        obs = env.reset()
        for i in range(env.horizon):
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
