import time
import random

import torch
from sb3_contrib import MaskablePPO

from teris_game_custom_wrapper_cnn import TerisEnv

if torch.backends.mps.is_available():
    MODEL_PATH = r"trained_models_cnn_mps/ppo_teris_final"
else:
    MODEL_PATH = r"trained_models_cnn/ppo_teris_final"

NUM_EPISODE = 3

RENDER = True
FRAME_DELAY = 0.05 # 0.01 fast, 0.05 slow
ROUND_DELAY = 5

seed = random.randint(0, 1e9)
print(f"Using seed = {seed} for testing.")

if RENDER:
    env = TerisEnv(seed=seed, silent_mode=False)
else:
    env = TerisEnv(seed=seed, silent_mode=True)

# Load the trained model
model = MaskablePPO.load(MODEL_PATH)

total_reward = 0
total_score = 0
min_score = 1e9
max_score = 0

for episode in range(NUM_EPISODE):
    obs = env.reset()
    episode_reward = 0
    done = False
    
    num_step = 0
    info = None

    sum_step_reward = 0

    retry_limit = 9
    print(f"=================== Episode {episode + 1} ==================")
    while not done:
        action, _ = model.predict(obs, action_masks=env.get_action_mask())
        prev_mask = env.get_action_mask()
        num_step += 1
        obs, reward, done, info = env.step(action)

        if done:
            if info["level"] == 16:
                print(f"You are BREATHTAKING! Victory reward: {reward:.4f}.")
            else:
                # 0: ROTATE, 1: HARD DROP, 2: SOFT DROP, 3: LEFT, 4: RIGHT
                last_action = ["ROTATE", "HARD DROP", "SOFT DROP", "LEFT", "RIGHT", "NOTHING"][action]
                print(f"Gameover Penalty: {reward:.4f}. Last action: {last_action}")

        else:
            sum_step_reward += reward
            
        episode_reward += reward
        if RENDER:
            env.render()
            time.sleep(FRAME_DELAY)

    episode_score = env.game.score
    if episode_score < min_score:
        min_score = episode_score
    if episode_score > max_score:
        max_score = episode_score
    
    print(f"Episode {episode + 1}: Reward Sum: {episode_reward:.4f}, Score: {episode_score}, Total Steps: {num_step}")
    total_reward += episode_reward
    total_score += env.game.score
    if RENDER:
        time.sleep(ROUND_DELAY)

env.close()
print(f"=================== Summary ==================")
print(f"Average Score: {total_score / NUM_EPISODE}, Min Score: {min_score}, Max Score: {max_score}, Average reward: {total_reward / NUM_EPISODE}")
