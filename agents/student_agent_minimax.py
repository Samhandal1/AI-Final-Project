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

    def minimax(self, chess_board, is_maximizing_player, my_pos, adv_pos, max_step):

        isEndGame, my_score, adv_score = self.check_endgame(chess_board, my_pos, adv_pos)

        if isEndGame:

            if my_score > adv_score:
                return 1
            elif my_score < adv_score:
                return 0
            else:
                return 0.5

        if is_maximizing_player:
            # Maximizing player tries to maximize the score
            max_eval = float('-inf')
            for move in self.generate_moves(chess_board, my_pos, adv_pos, max_step):
                new_pos, _ = move
                eval = self.minimax(chess_board, False, new_pos, adv_pos, max_step)
                max_eval = max(max_eval, eval)
            return max_eval
        else:
            # Minimizing player tries to minimize the score
            min_eval = float('inf')
            for move in self.generate_moves(chess_board, adv_pos, my_pos, max_step):
                new_pos, _ = move
                eval = self.minimax(chess_board, True, my_pos, new_pos, max_step)
                min_eval = min(min_eval, eval)
            return min_eval

    def step(self, chess_board, my_pos, adv_pos, max_step):
        best_move = None
        best_score = float('-inf')

        # Generate all possible moves and apply minimax
        for move in self.generate_moves(chess_board, my_pos, adv_pos, max_step):
            score = self.minimax(chess_board, False, my_pos, adv_pos, max_step)
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