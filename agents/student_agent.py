# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
import random


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

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        # Taken for random_agent.py
        def __init__(self):
        super(HumanAgent, self).__init__()
        self.name = "HumanAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

        # Define moves (Up, Right, Down, Left)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()


        # Check that players are not randomly positioned in same cell
        # Only do at the very beginning, not before every step
        if (my_pos[0] == adv_pos[1] and my_pos[1] == adv_pos[1]) :
            return False

        # logic goes here
        
        # Get list of all possible moves (use check_valid_step and check for barriers)
        # Simulate wall placement for all possible moves
        # Evaluate moves with minimax and choose best one (make sure we dont end up in an immediate loss)
        # Check if game has ended with check_endgame


        time_taken = time.time() - start_time
        
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, self.dir_map["u"]

    def generate_potential_moves(self, start_pos, chess_board, max_step):
        """
        Generate a list of potential moves from the current position.

        Parameters
        ----------
        start_pos : tuple
            The start position of the agent (x, y).
        chess_board : np.ndarray
            The chess board state.
        max_step : int
            The maximum number of steps the agent can move.

        Returns
        -------
        potential_moves : list
            A list of potential moves and barrier placements.
        """

        # Moves (Up, Right, Down, Left)
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        potential_moves = []
        board_size = chess_board.shape[0]
        visited = set()
        queue = [(start_pos, 0)]

        while queue:
            current_pos, steps = queue.pop(0)
            if steps > max_step:
                continue
            visited.add(current_pos)

            # Check for potential barrier placements at the current position
            for dir in range(4):  # 0: up, 1: right, 2: down, 3: left
                if not chess_board[current_pos[0], current_pos[1], dir]:
                    potential_moves.append({'position': current_pos, 'direction': dir})

            # Explore adjacent positions
            for dx, dy in moves:  # up, right, down, left
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)

                if 0 <= next_pos[0] < board_size and 0 <= next_pos[1] < board_size:
                    if next_pos not in visited:
                        queue.append((next_pos, steps + 1))

        return potential_moves


    def check_valid_step(self, start_pos, end_pos, barrier_dir):
        """
        Check if the step the agent takes is valid (reachable and within max steps).

        Parameters
        ----------
        start_pos : tuple
            The start position of the agent.
        end_pos : np.ndarray
            The end position of the agent.
        barrier_dir : int
            The direction of the barrier.
        """
        # Moves (Up, Right, Down, Left)
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        # Endpoint already has barrier or is border
        r, c = end_pos
        if self.chess_board[r, c, barrier_dir]:
            return False
        if np.array_equal(start_pos, end_pos):
            return True

        # Get position of the adversary
        adv_pos = self.p0_pos if self.turn else self.p1_pos

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == self.max_step:
                break
            for dir, move in enumerate(self.moves):
                if self.chess_board[r, c, dir]:
                    continue

                next_pos = cur_pos + move
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached
        
    def is_barrier(self, cur_pos, chess_board, barrier_dir):
        """
        Check if there is a barrier in the specified direction from the current position.

        Parameters
        ----------
        cur_pos : tuple
            The current position of the agent (x, y).
        chess_board : np.ndarray
            The chess board state.
        barrier_dir : int
            The direction to check for a barrier (0: up, 1: right, 2: down, 3: left).

        Returns
        -------
        bool
            True if there is a barrier, False otherwise.
        """
        x, y = cur_pos
        board_size = chess_board.shape[0]

        # Check for barriers in each direction considering board boundaries
        if barrier_dir == 0 and x > 0:  # Up
            return chess_board[x, y, 0]
        elif barrier_dir == 1 and y < board_size - 1:  # Right
            return chess_board[x, y, 1]
        elif barrier_dir == 2 and x < board_size - 1:  # Down
            return chess_board[x, y, 2]
        elif barrier_dir == 3 and y > 0:  # Left
            return chess_board[x, y, 3]
        return False  # If the direction is off the board, return False

    def set_barrier(self, cur_pos, chess_board, barrier_dir):  # applies move on copy chessboard

        # Moves (Up, Right, Down, Left)
        # self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        chess_board[int(cur_pos[0])][int(cur_pos[1])][int(barrier_dir)] = True
        if barrier_dir == 0:
            chess_board[cur_pos[0] - 1, cur_pos[1], 2] = True  # go up a row and set down to true

        elif barrier_dir == 1:
            chess_board[cur_pos[0], cur_pos[1] + 1, 3] = True  # go right a position and set left true

        elif barrier_dir == 2:
            chess_board[cur_pos[0] + 1, cur_pos[1], 0] = True  # go down a position and set up true

        elif barrier_dir == 3:
            chess_board[cur_pos[0] - 1, cur_pos[1] - 1, 1] = True  # go left a position and set right true
        return chess_board

    """ From world.py
    def set_barrier(self, r, c, dir):
        # Set the barrier to True
        self.chess_board[r, c, dir] = True
        # Set the opposite barrier to True
        move = self.moves[dir]
        self.chess_board[r + move[0], c + move[1], self.opposites[dir]] = True
    """

    def check_endgame(self, chess_board, p0_pos, p1_pos):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """

        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        board_size = len(chess_board[0])

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
                    #########Add into else block with more logic
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

                    ##########

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(p0_pos))
        p1_r = find(tuple(p1_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1  # Tie
        if player_win >= 0:
            logging.info(
                f"Game ends! Player {self.player_names[player_win]} wins having control over {win_blocks} blocks!"
            )
        else:
            logging.info("Game ends! It is a Tie!")
        return True, p0_score, p1_score


