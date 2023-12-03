# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def evaluate(self, chess_board, my_pos, adv_pos):
        score = 0

        # 1. Check for being surrounded by three walls
        if self.is_three_sided(my_pos, chess_board):
            score -= 100

        # 2. Count number of move options
        my_move_options = self.count_move_options(my_pos, chess_board)
        score += my_move_options * 5  # Mobility is important

        # 3. Calculate distance from the opponent
        distance_to_opponent = self.calculate_distance(my_pos, adv_pos)
        # In early and middle game, maintain a moderate distance
        score += max(5 - distance_to_opponent, 0) * 3

        # 4. Assess the game stage by the number of walls
        total_walls = np.sum(chess_board)
        board_size = chess_board.shape[0]
        early_game_threshold = board_size * 2  # Early game if fewer than 2 walls per row
        middle_game_threshold = board_size * 4  # Middle game up to 4 walls per row

        if total_walls < early_game_threshold:
            score += my_move_options * 3  # Focus on mobility in the early game
        elif total_walls < middle_game_threshold:
            score += (10 - distance_to_opponent) * 2  # Begin aggressive play
        else:
            score += (board_size**2 - total_walls) * 2  # Focus on area control in the end game

        # 5. Check for continuing a wall pattern
        if self.is_continuing_wall(my_pos, chess_board):
            score += 10  # Reward for strategic wall placement

        return score


    # Helper functions for evaluate (implement these based on game logic)
    def is_three_sided(self, pos, chess_board):
        r, c = pos
        sides_with_walls = 0

        # Moves (Up, Right, Down, Left)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        # Check each side for a wall
        for i, move in enumerate(moves):
            new_r, new_c = r + move[0], c + move[1]
            if chess_board[r, c, i] or not (0 <= new_r < chess_board.shape[0] and 0 <= new_c < chess_board.shape[1]):
                sides_with_walls += 1

        return sides_with_walls == 3


    def count_move_options(self, pos, chess_board):
        r, c = pos
        move_options = 0

        # Moves (Up, Right, Down, Left)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        # Count viable moves
        for i, move in enumerate(moves):
            new_r, new_c = r + move[0], c + move[1]
            if not chess_board[r, c, i] and 0 <= new_r < chess_board.shape[0] and 0 <= new_c < chess_board.shape[1]:
                move_options += 1

        return move_options

    def calculate_distance(self, pos1, pos2):
        # Manhattan distance
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def is_continuing_wall(self, pos, chess_board):
        r, c = pos

        # Moves (Up, Right, Down, Left)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        for i, move in enumerate(moves):
            new_r, new_c = r + move[0], c + move[1]
            if 0 <= new_r < chess_board.shape[0] and 0 <= new_c < chess_board.shape[1]:
                if chess_board[new_r, new_c, (i + 2) % 4]:  # Check opposite direction for a continuation
                    return True

        return False

    def minimax(self, chess_board, depth, is_maximizing_player, my_pos, adv_pos, max_step):
        
        if depth == 0:
            
            return self.evaluate(chess_board, my_pos, adv_pos)

        if is_maximizing_player:
            # Maximizing player tries to maximize the score
            max_eval = float('-inf')
            for move in self.generate_moves(chess_board, my_pos, adv_pos, max_step):
                new_pos, _ = move
                eval = self.minimax(chess_board, depth - 1, False, new_pos, adv_pos, max_step)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            # Minimizing player tries to minimize the score
            min_eval = float('inf')
            for move in self.generate_moves(chess_board, adv_pos, my_pos, max_step):
                new_pos, _ = move
                eval = self.minimax(chess_board, depth - 1, True, my_pos, new_pos, max_step)
                min_eval = min(min_eval, eval)
            return min_eval

    def step(self, chess_board, my_pos, adv_pos, max_step):
        best_move = None
        best_score = float('-inf')
        depth = 2

        if chess_board.shape[0] < 8:
            depth = 5
        elif chess_board.shape[0] >= 8 and chess_board.shape[0] <= 10:
            depth = 4
        else:
            depth = 3

        # Generate all possible moves and apply minimax
        for move in self.generate_moves(chess_board, my_pos, adv_pos, max_step):
            score = self.minimax(chess_board, depth, False, my_pos, adv_pos, max_step)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move
    
    def generate_moves(self, chess_board, my_pos, adv_pos, max_step):

        # Moves (Up, Right, Down, Left)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        possible_moves = []

        # Generate all possible positions after moving up to max_step steps
        for step in range(1, max_step + 1):
            for move in moves:
                new_r = my_pos[0] + move[0] * step
                new_c = my_pos[1] + move[1] * step

                # Check if the new position is valid
                if (0 <= new_r < chess_board.shape[0] and
                    0 <= new_c < chess_board.shape[1] and
                    not (new_r, new_c) == adv_pos and
                    self.is_path_clear(chess_board, my_pos, (new_r, new_c), adv_pos, move, moves.index(move))):
                    
                    # For each valid position, generate possible barrier placements
                    for dir in range(4):

                        # Check if it's possible to place a barrier in this direction
                        if not chess_board[new_r, new_c, dir]:
                            possible_moves.append(((new_r, new_c), dir))

        return possible_moves

    def is_path_clear(self, chess_board, start_pos, end_pos, adv_pos, move, dir):

        # Check if the path from start_pos to end_pos is clear of barriers
        r, c = start_pos
        while (r, c) != end_pos:

            # If there's a barrier in the direction of the move or adv is on path
            if chess_board[r, c, dir] or (r, c) == adv_pos:
                return False
            
            # continue checking path
            r += move[0]
            c += move[1]

        return True