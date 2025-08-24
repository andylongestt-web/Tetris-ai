import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
import random
import numpy as np
import math
import time

class TetrisGame:
    def __init__(self, seed=42, level=1, board_width=10, board_height=20, silent_mode=True):
        self.board_width = board_width
        self.board_height = board_height
        self.cell_size = 30
        self.width = self.board_width * self.cell_size
        self.height = self.board_height * self.cell_size
        self.border_size = 20
        self.display_width = self.width + 2 * self.border_size + 100
        self.display_height = self.height + 2 * self.border_size

        self.silent_mode = silent_mode
        if not silent_mode:
            pygame.init()
            pygame.display.set_caption("Tetris")
            self.screen = pygame.display.set_mode((self.display_width, self.display_height))
            self.font = pygame.font.Font(None, 30)

        self.board = [[0 for _ in range(self.board_width)] for _ in range(self.board_height)]
        self.current_piece = None
        self.next_piece = None
        self.position = [0, self.board_width // 2 - 2]
        self.score = 0
        self.level = level
        self.lines_cleared = 0
        self.drop_speeds = [1.0, 0.783, 0.617, 0.467, 0.367, 0.25, 0.183, 0.133, 0.1, 0.083, 0.067, 0.05, 0.033, 0.017, 0.01]
        self.drop_speed = self.drop_speeds[min(level - 1, 14)]
        self.colors = {'I': (0, 0, 255), 'J': (0, 0, 128), 'L': (255, 165, 0), 'O': (255, 255, 0), 'S': (0, 255, 0), 'T': (128, 0, 128), 'Z': (255, 0, 0)}
        self.shapes = {
            'I': np.array([[1, 1, 1, 1]]),
            'J': np.array([[1, 0, 0], [1, 1, 1]]),
            'L': np.array([[0, 0, 1], [1, 1, 1]]),
            'O': np.array([[1, 1], [1, 1]]),
            'S': np.array([[0, 1, 1], [1, 1, 0]]),
            'T': np.array([[0, 1, 0], [1, 1, 1]]),
            'Z': np.array([[1, 1, 0], [0, 1, 1]])
        }

        self.start_time = time.time()

        random.seed(seed)
        self.new_piece()

    def reset(self):
        self.board = [[0 for _ in range(self.board_width)] for _ in range(self.board_height)]
        self.current_piece = None
        self.next_piece = None
        self.position = [0, self.board_width // 2 - len(self.shapes[random.choice(list(self.shapes.keys()))][0]) // 2]
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.game_state = "welcome"
        self.drop_speed = self.drop_speeds[self.level - 1]
        self.new_piece()

    def new_piece(self):
        if self.next_piece is None:
            self.next_piece = random.choice(list(self.shapes.keys()))
        self.current_piece = self.next_piece
        self.next_piece = random.choice(list(self.shapes.keys()))
        self.position = [0, self.board_width // 2 - len(self.shapes[self.current_piece][0]) // 2]

    def rotate_piece(self):
        original_shape = self.shapes[self.current_piece].copy()
        rotated_shape = np.rot90(original_shape)
        original_pos = list(self.position)
        for _ in range(4):  # Try up to 4 offset positions
            if self.valid_move(self.position, rotated_shape):
                self.shapes[self.current_piece] = rotated_shape
                return
            self.position[1] += 1  # Move right
        self.position = original_pos
        self.shapes[self.current_piece] = original_shape

    def valid_move(self, pos, piece):
        for i in range(piece.shape[0]):
            for j in range(piece.shape[1]):
                if piece[i, j]:
                    new_x, new_y = pos[0] + i, pos[1] + j
                    if (new_x >= self.board_height or new_x < 0 or
                        new_y >= self.board_width or new_y < 0 or
                        (new_x >= 0 and self.board[new_x][new_y])):
                        return False
        return True

    def merge_piece(self):
        for i in range(self.shapes[self.current_piece].shape[0]):
            for j in range(self.shapes[self.current_piece].shape[1]):
                if self.shapes[self.current_piece][i, j]:
                    self.board[self.position[0] + i][self.position[1] + j] = self.current_piece
        return self.clear_lines(), self.board

    def clear_lines(self):
        lines = 0
        for i in range(self.board_height):
            if all(self.board[i]):
                del self.board[i]
                self.board.insert(0, [0 for _ in range(self.board_width)])
                lines += 1
        if lines:
            self.lines_cleared += lines
            level_up = self.lines_cleared // 10
            if level_up > self.level - 1:
                self.level = min(level_up + 1, 15)
                self.drop_speed = self.drop_speeds[self.level - 1]
            self.score += {1: 100, 2: 300, 3: 500, 4: 800}[min(lines, 4)] * self.level
            return {1: 100, 2: 300, 3: 500, 4: 800}[min(lines, 4)] * self.level
        return 0
        
    def step(self, action):
        lines_score = 0
        drop_score = 0

        # Do notice that we want the board before we clear the line
        board_after_step = [[[''] for _ in range(10)] for _ in range(20)]
        board_before_step = [[[''] for _ in range(10)] for _ in range(20)]

        for i in range(0, 20, 1):
            for j in range(0, 10, 1):
                board_before_step[i][j] = self.board[i][j]
                board_after_step[i][j] = self.board[i][j]

        if action == 1:  # Hard drop
            lines_score, drop_score, board_after_step = self.hard_drop()
            self.start_time = time.time()
        elif action == 3:  # Move left
            self.move(-1)
        elif action == 4:  # Move right
            self.move(1)
        elif action == 2:  # Accelerate drop
            lines_score = self.accelerate()
            self.start_time = time.time()
        elif action == 0:  # Rotate clockwise
            self.rotate_piece()
        if action != 1 and action != 2 :
            cur_time = time.time()
            if cur_time - self.start_time >= self.drop_speed:
                lines_score, board_after_step = self.drop_piece()
                self.start_time = cur_time

        done = not self.valid_move(self.position, self.shapes[self.current_piece])
        info = {
            "score": self.score,
            "level": self.level,
            "lines_cleared": self.lines_cleared,
            "position": list(self.position),
            "piece": self.current_piece,
            "lines_score": lines_score,
            "drop_score": drop_score,
            "board_before_step": board_before_step,
            "board_after_step": board_after_step,
            "board_after_clear": self.board
        }
        return done, info

    def hard_drop(self):
        start_row = self.position[0]  # 方块掉落前的起始行
        current_row = self.position[0]
        while self.valid_move([current_row + 1, self.position[1]], self.shapes[self.current_piece]):
            current_row += 1
        self.position[0] = current_row
        # 计算方块底部的实际行号
        lines_that_cleared, board_before = self.merge_piece()
        drop_distance = current_row - start_row
        self.score += drop_distance * 2
        self.new_piece()
        return lines_that_cleared, drop_distance*2, board_before

    def move(self, direction):
        new_pos = [self.position[0], self.position[1] + direction]
        if self.valid_move(new_pos, self.shapes[self.current_piece]):
            self.position[1] += direction

    def accelerate(self):
        lines_that_cleared = 0
        new_pos = [self.position[0] + 2, self.position[1]]
        if self.valid_move(new_pos, self.shapes[self.current_piece]):
            self.position[0] += 2
        else:
            lines_that_cleared, _ = self.drop_piece()
        return lines_that_cleared

    def drop_piece(self):
        lines_that_cleared = 0
        new_pos = [self.position[0] + 1, self.position[1]]
        board_before = self.board
        if self.valid_move(new_pos, self.shapes[self.current_piece]):
            self.position = new_pos
        else:
            lines_that_cleared, board_before = self.merge_piece()
            self.new_piece()
        return lines_that_cleared, board_before

    def draw_welcome_screen(self):
        title_text = self.font.render("TETRIS", True, (255, 255, 255))
        level_text = self.font.render("LEVEL 1", True, (0, 191, 255))
        play_text = self.font.render("PLAY", True, (255, 255, 255))
        self.screen.fill((0, 0, 0))
        self.screen.blit(title_text, (self.display_width // 2 - title_text.get_width() // 2, self.display_height // 4))
        self.screen.blit(level_text, (self.display_width // 2 - level_text.get_width() // 2, self.display_height // 2 - 20))
        self.screen.blit(play_text, (self.display_width // 2 - play_text.get_width() // 2, self.display_height // 2 + 20))
        pygame.display.flip()

    def new_piece(self):
        if self.next_piece is None:
            self.next_piece = random.choice(list(self.shapes.keys()))
        self.current_piece = self.next_piece
        self.next_piece = random.choice(list(self.shapes.keys()))
        # 创建独立副本，避免旋转影响
        self.shapes[self.current_piece] = self.shapes[self.current_piece].copy()
        self.position = [0, self.board_width // 2 - len(self.shapes[self.current_piece][0]) // 2]

    def render(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), (self.border_size - 2, self.border_size - 2, self.width + 4, self.height + 4), 2)
        for i in range(self.board_height):
            for j in range(self.board_width):
                if self.board[i][j]:
                    pygame.draw.rect(self.screen, self.colors[self.board[i][j]], (j * self.cell_size + self.border_size, i * self.cell_size + self.border_size, self.cell_size - 1, self.cell_size - 1))
        for i in range(self.shapes[self.current_piece].shape[0]):
            for j in range(self.shapes[self.current_piece].shape[1]):
                if self.shapes[self.current_piece][i, j]:
                    pygame.draw.rect(self.screen, self.colors[self.current_piece], ((self.position[1] + j) * self.cell_size + self.border_size, (self.position[0] + i) * self.cell_size + self.border_size, self.cell_size - 1, self.cell_size - 1))
        
        # 右边栏布局
        sidebar_x = self.width + 2 * self.border_size
        sidebar_width = 100
        row_height = self.height / 8  # 8 行（Next + next_piece + Score + score + Level + level + Lines + lines + Pause）
        font_height = self.font.get_height()

        # "Next" 居中
        next_text = self.font.render("Next", True, (255, 255, 255))
        next_text_x = sidebar_x + (sidebar_width - next_text.get_width()) // 2
        self.screen.blit(next_text, (next_text_x, self.border_size + row_height * 0))

        # 固定 Next 方块样式
        next_start_y = self.border_size + row_height * 1
        fixed_shapes = {
            'I': np.array([[1, 1, 1, 1]]),
            'J': np.array([[1, 0, 0], [1, 1, 1]]),
            'L': np.array([[0, 0, 1], [1, 1, 1]]),
            'O': np.array([[1, 1], [1, 1]]),
            'S': np.array([[0, 1, 1], [1, 1, 0]]),
            'T': np.array([[0, 1, 0], [1, 1, 1]]),
            'Z': np.array([[1, 1, 0], [0, 1, 1]])
        }
        next_shape = fixed_shapes[self.next_piece]
        # 计算方块的总宽度（以半大小 cell_size // 2 为单位）
        total_width = next_shape.shape[1] * (self.cell_size // 2)
        start_x = sidebar_x + (sidebar_width - total_width) // 2  # 居中 x 坐标
        for i in range(next_shape.shape[0]):
            for j in range(next_shape.shape[1]):
                if next_shape[i, j]:
                    x = start_x + j * (self.cell_size // 2)
                    y = next_start_y + i * (self.cell_size // 2)
                    pygame.draw.rect(self.screen, self.colors[self.next_piece], (x, y, self.cell_size // 2 - 1, self.cell_size // 2 - 1))

        # "Score" 和分数
        score_label = self.font.render("Score", True, (255, 255, 255))
        score_value = self.font.render(str(self.score), True, (255, 255, 255))
        self.screen.blit(score_label, (sidebar_x + (sidebar_width - score_label.get_width()) // 2, self.border_size + row_height * 2))
        self.screen.blit(score_value, (sidebar_x + (sidebar_width - score_value.get_width()) // 2, self.border_size + row_height * 3))

        # "Level" 和等级
        level_label = self.font.render("Level", True, (255, 255, 255))
        level_value = self.font.render(str(self.level), True, (255, 255, 255))
        self.screen.blit(level_label, (sidebar_x + (sidebar_width - level_label.get_width()) // 2, self.border_size + row_height * 4))
        self.screen.blit(level_value, (sidebar_x + (sidebar_width - level_value.get_width()) // 2, self.border_size + row_height * 5))

        # "Lines" 和总消去行数
        lines_label = self.font.render("Lines", True, (255, 255, 255))
        lines_value = self.font.render(str(self.lines_cleared), True, (255, 255, 255))
        self.screen.blit(lines_label, (sidebar_x + (sidebar_width - lines_label.get_width()) // 2, self.border_size + row_height * 6))
        self.screen.blit(lines_value, (sidebar_x + (sidebar_width - lines_value.get_width()) // 2, self.border_size + row_height * 7))

        # 暂停按钮
        pause_text = self.font.render("||", True, (255, 255, 255))
        self.screen.blit(pause_text, (sidebar_x + (sidebar_width - pause_text.get_width()) // 2, self.border_size + row_height * 8 - font_height))

        pygame.display.flip()

if __name__ == "__main__":
    import time
    game = TetrisGame(silent_mode=False)
    game.screen = pygame.display.set_mode((game.display_width, game.display_height))
    game.font = pygame.font.Font(None, 30)
    
    game_state = "welcome"
    
    start_time = time.time()
    last_action_time = time.time()
    action = -1

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if game_state == "welcome" and event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                play_rect = game.font.render("PLAY", True, (0, 0, 0)).get_rect(center=(game.display_width // 2, game.display_height // 2 + 20))
                if play_rect.collidepoint(mouse_pos):
                    game_state = "running"
            if game_state == "running" and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_2:
                    action = 1
                elif event.key == pygame.K_4:
                    action = 3
                elif event.key == pygame.K_6:
                    action = 4
                elif event.key == pygame.K_8:
                    action = 2
                elif event.key == pygame.K_5:
                    action = 0

        if game_state == "running":
            current_time = time.time()
            # Handle immediate actions
            if current_time - last_action_time >= 0.1:  # Debounce
                done, _ = game.step(action)
                game.render()
                last_action_time = current_time
                if done:
                    game_state = "welcome"
                    game.reset()
                action = -1  # Reset action after processing
        elif game_state == "welcome":
            game.draw_welcome_screen()

        pygame.time.wait(1)