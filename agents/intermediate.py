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
            score -= 150  # Increase the penalty for being nearly trapped

        # 2. Count number of move options (Mobility)
        my_move_options = self.count_move_options(my_pos, chess_board)
        score += my_move_options * 10  # Increase the weight for mobility

        # 3. Calculate distance from the opponent
        distance_to_opponent = self.calculate_distance(my_pos, adv_pos)
        # Dynamic distance strategy
        if distance_to_opponent < 3:
            score += (5 - distance_to_opponent) * 5  # Reward aggressive positioning in certain scenarios
        else:
            score += distance_to_opponent * 2  # Reward maintaining a safer distance in others

        # 4. Assess the game stage by the number of walls
        total_walls = np.sum(chess_board)
        board_size = chess_board.shape[0]
        early_game_threshold = board_size * 2
        middle_game_threshold = board_size * 4

        if total_walls < early_game_threshold:
            score += my_move_options * 5  # Increase focus on mobility in the early game
        elif total_walls < middle_game_threshold:
            score += 20 - (distance_to_opponent * 2)  # More aggressive play in the middle game
        else:
            score += (board_size**2 - total_walls) * 3  # Heavier focus on area control in the end game

        # 5. Check for continuing a wall pattern
        if self.is_continuing_wall(my_pos, chess_board):
            score += 20  # Increase reward for strategic wall placement

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
            return self.evaluate(chess_board, my_pos, adv_pos), None

        best_move = None

        if is_maximizing_player:
            max_eval = float('-inf')
            for move in self.generate_moves(chess_board, my_pos, adv_pos, max_step):
                new_pos, _ = move
                eval, _ = self.minimax(chess_board, depth - 1, False, new_pos, adv_pos, max_step)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in self.generate_moves(chess_board, adv_pos, my_pos, max_step):
                new_pos, _ = move
                eval, _ = self.minimax(chess_board, depth - 1, True, my_pos, new_pos, max_step)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
            return min_eval, best_move

    def step(self, chess_board, my_pos, adv_pos, max_step):
        best_move = None
        best_score = float('-inf')
        depth = 1

        # Adjust depth based on board size
        if chess_board.shape[0] < 8:
            depth = 1
        elif chess_board.shape[0] >= 8 and chess_board.shape[0] <= 10:
            depth = 1
        else:
            depth = 1

        # Generate all possible moves and apply minimax
        moves = self.generate_moves(chess_board, my_pos, adv_pos, max_step)
        for move in moves:
            score, _ = self.minimax(chess_board, depth, False, my_pos, adv_pos, max_step)
            if score > best_score:
                best_score = score
                best_move = move

        # Ensure that a move is returned
        if best_move is not None:
            return best_move
        else:
            print(my_pos)
            print(adv_pos)
            print(chess_board)
            return (my_pos, 0)
        # return best_move if best_move is not None else (my_pos, 0)  # Default move if no best move is found

    
    def generate_moves(self, chess_board, my_pos, adv_pos, max_step):

        # Moves (Up, Right, Down, Left)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        possible_moves = []

        # Generate all possible positions after moving up to max_step steps
        for step in range(0, max_step):
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

    def check_endgame(self, chess_board, p0_pos, p1_pos):
        """
        ** edit end game function from world.py **
        Check if the game ends.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        """

        # Moves (Up, Right, Down, Left)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        board_size = chess_board.shape[0]

        # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(moves[1:3]):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(p0_pos))
        p1_r = find(tuple(p1_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)

        # Game ends if players are in different regions
        return (p0_r != p1_r), p0_score, p1_score