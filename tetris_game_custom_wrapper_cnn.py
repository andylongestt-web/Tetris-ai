import math
import time
import cv2

import gym
import numpy as np

from teris_game import TetrisGame

class TerisEnv(gym.Env):
    def __init__(self, seed=42, silent_mode=True):
        super().__init__()
        self.game = TetrisGame(seed=seed, silent_mode=silent_mode)
        self.game.reset()

        self.silent_mode = silent_mode

        self.action_space = gym.spaces.Discrete(6) # 0: ROTATE, 1: HARD DROP, 2: SOFT DROP, 3: LEFT, 4: RIGHT, 5：NOTHING

        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(84, 84, 3),
            dtype=np.uint8
        )

        self.done = False

        self.last_action_time = time.time()

    def reset(self):
        self.game.reset()

        self.done = False

        obs = self._generate_observation()
        return obs

    def step(self, action):
        current_time = time.time()
        if current_time - self.last_action_time < 0.1:
            time.sleep(0.1 - (current_time - self.last_action_time))
        self.last_action_time = current_time

        self.done, info = self.game.step(action)

        # print(f"Excecuting action {action}")
        # print("Board_before_step:")
        # print(info["board_before_step"])
        # print("Board_after_step")
        # print(info["board_after_step"])
        # print("==========================")
        obs = self._generate_observation()

        reward = 0.0

        if info["level"] == 16:
            # We should of course give an infinitely large amount of reward when win
            reward = 100
            self.done = True
            if not self.silent_mode:
                print("YOU ACTUALLY WIN!! I WASN'T EXPECTING IT!!")
            return obs, reward, self.done, info

        if self.done:
            # Game Over penalty
            # reward = - math.exp(-self.game.score*0.0001) * 1
            reward = -10
            return obs, reward, self.done, info

        if info["lines_cleared"]>0:
            encourage_lines_score = 100
            reward = (info["lines_cleared"] ** 2) * encourage_lines_score

        # 8.4 We may choose to encourage hard drop so it looks fancier
        # if info["drop_score"]>0:
        #     encourage_hard_drop = 1
        #     reward += info["drop_score"] * encourage_hard_drop

        # 8.7 We feel good when blocks across column are kinda average
        num_before = self.cal_num_across_column(info["board_before_step"])
        num_after = self.cal_num_across_column(info["board_after_clear"])
        gap_num = self.cal_bumpiness(num_after) - self.cal_bumpiness(num_before) # better if eliminate the var! which means gap_var < 0 good
        average_par = 1
        reward -= gap_num * average_par

        # 8.6 We tend to feel anxious when the blocks pack higher
        sum_height_before = self.cal_sum_height(info["board_before_step"])
        sum_height_after = self.cal_sum_height(info["board_after_clear"])
        gap_height = sum_height_after - sum_height_before
        anxious_par = 10
        reward -= gap_height * anxious_par  

        # 8.15 Don't wanna see any hole!
        holes_before = self.cal_holes(info["board_before_step"])
        holes_after = self.cal_holes(info["board_after_clear"])
        gap_holes = holes_after - holes_before
        reward -= gap_holes * 2  

        return obs, reward, self.done, info
    
    def render(self):
        self.game.render()

    def get_action_mask(self):
        lpos = [self.game.position[0], self.game.position[1]-1]
        rpos = [self.game.position[0], self.game.position[1]+1]        
        piece = self.game.shapes[self.game.current_piece]
        # 0: ROTATE, 1: HARD DROP, 2: SOFT DROP, 3: LEFT, 4: RIGHT, 5：NOTHING
        return np.array([[True, True, True, self.game.valid_move(lpos, piece), self.game.valid_move(rpos, piece), True]])

    def cal_bumpiness(self, sum_height):
        sum = sum_height[0]
        for i in range(1, 10, 1):
            sum += abs(sum_height[i] - sum_height[i-1])
        return sum

    def cal_max_height(self, board):
        max_height = 0
        for j in range(0, 10, 1):
            num = 0
            for i in range(0, 20, 1):
                if board[i][j]:
                    num = num + 1
            max_height = max(max_height, num)
        return max_height
    
    def cal_sum_height(self, board):
        sum_height = 0
        for j in range(0, 10, 1):
            num = 0
            for i in range(0, 20, 1):
                if board[i][j]:
                    num = num + 1
            sum_height += num
        return sum_height
 
    def cal_holes(self, board):
        holes = 0
        for i in range(0, 19, 1):
            block = False
            for j in range(1, 9, 1):
                if board[i][j]:
                    block = True
                if not board[i][j] and block:
                    holes += 1
            if (not board[i][0] and board[i+1][0]) or (not board[i][9] and board[i+1][9]):
                holes += 1
        return holes

    def cal_num_across_column(self, board):
        cal = []
        for j in range(0, 10, 1):
            num = 0
            for i in range(0, 20, 1):
                if board[i][j]:
                    num = num + 1
            cal.append(num)
        return cal

    def _generate_observation(self):  
        obs = np.zeros((self.game.board_height, self.game.board_width, 3), dtype=np.uint8)
        
        # Fallen blocks
        for i in range(self.game.board_height):
            for j in range(self.game.board_width):
                if self.game.board[i][j]:
                    # obs[i, j] = self.game.colors[self.game.board[i][j]]
                    obs[i, j] = [255, 255, 255]

        # Falling blocks
        for i in range(self.game.shapes[self.game.current_piece].shape[0]):
            for j in range(self.game.shapes[self.game.current_piece].shape[1]):
                if self.game.shapes[self.game.current_piece][i, j]:
                    board_x = self.game.position[0] + i
                    board_y = self.game.position[1] + j
                    if 0 <= board_x < self.game.board_height and 0 <= board_y < self.game.board_width:
                        # obs[board_x, board_y] = self.game.colors[self.game.current_piece]
                        obs[board_x, board_y] = [255, 255, 255]

        target_width = 42  
        target_height = 84  

        obs_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)  
        obs_resized = cv2.resize(obs_bgr, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
        obs_rgb = cv2.cvtColor(obs_resized, cv2.COLOR_BGR2RGB)  

        final_obs = np.zeros((84, 84, 3), dtype=np.uint8)
        
        x_offset = (84 - target_width) // 2  
        y_offset = 0  
        final_obs[y_offset:y_offset + target_height, x_offset:x_offset + target_width, :] = obs_rgb

        return final_obs
    
# Test the environment using random actions
NUM_EPISODES = 1
RENDER_DELAY = 0.001
from matplotlib import pyplot as plt

if __name__ == "__main__":
    env = TerisEnv(silent_mode=False)

    sum_reward = 0

    # 0: ROTATE, 1: HARD DROP, 2: SOFT DROP, 3: LEFT, 4: RIGHT, 5：NOTHING
    action_list = [1, 1, 1, 0, 0, 0, 2, 2, 2, 3, 3, 3]
    
    for _ in range(NUM_EPISODES):
        obs = env.reset()
        done = False
        i = 0
        while not done:
            plt.imshow(obs, interpolation='nearest')
            plt.show()
            action = env.action_space.sample()
            action = action_list[i]
            i = (i + 1) % len(action_list)
            obs, reward, done, info = env.step(action)
            sum_reward += reward
            if np.absolute(reward) > 0.001:
                print(reward)
            env.render()
            
            time.sleep(RENDER_DELAY)
        print("sum_reward: %f" % sum_reward)
        print("episode done")
        # time.sleep(100)
    
    env.close()
    print("Average episode reward for random strategy: {}".format(sum_reward/NUM_EPISODES))

# conda activate TerisAI
# python teris_game_custom_wrapper_cnn.py


