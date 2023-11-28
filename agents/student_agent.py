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

        # Define moves (Up, Right, Down, Left)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        #start_time = time.time()


        # Check that players are not randomly positioned in same cell
        # Only do at the very beginning, not before every step
        possible_positions = self.generate_potential_moves(my_pos, chess_board,
                                      max_step)  

        mini = self.minimax(0, possible_positions, my_pos, adv_pos, chess_board, max_step,
                            True) 

        # logic goes here
        
        # Get list of all possible moves (use check_valid_step and check for barriers)
        # Simulate wall placement for all possible moves
        # Evaluate moves with minimax and choose best one (make sure we dont end up in an immediate loss)
        # Check if game has ended with check_endgame


        #time_taken = time.time() - start_time
        
        #print("My AI's turn took ", time_taken, "seconds.")

        
        
        return mini[0], int(mini[1])


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
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        potential_moves = []
        visited = {tuple(start_pos)}
        queue = [(start_pos, 0)]

        while queue:
            current_pos, steps = queue.pop(0)
            cur_r, cur_c = current_pos

            # Check for potential barrier placements at the current position
            for dir, move in enumerate(moves):
                if not chess_board[cur_r][cur_c][dir]:
                    potential_moves.append((current_pos, dir))

            if steps < max_step:
                # Explore adjacent positions
                for move in moves:
                    next_pos = (cur_r + move[0], cur_c + move[1])

                    # Check for valid position within the board boundaries
                    if 0 <= next_pos[0] < chess_board.shape[0] and 0 <= next_pos[1] < chess_board.shape[1]:
                        if next_pos not in visited:
                            visited.add(next_pos)
                            queue.append((next_pos, steps + 1))

        return potential_moves
    
    


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

    def minimax(self, depth, possible_moves, my_pos, adv_pos, chess_board, max_step, is_maximizing_player, alpha=float('-inf'), beta=float('inf')):
        if depth == 0 or self.check_endgame(chess_board, my_pos, adv_pos):
            return self.evaluate(chess_board, my_pos, adv_pos), None

        if is_maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for move in possible_moves:
                simulated_board = deepcopy(chess_board)
                # Apply move on the simulated board
                # You need to implement apply_move
                self.apply_move(simulated_board, move)
                eval = self.minimax(depth - 1, possible_moves, adv_pos, simulated_board, max_step, False, alpha, beta)[0]
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in possible_moves:
                simulated_board = deepcopy(chess_board)
                # Apply move on the simulated board
                # You need to implement apply_move
                self.apply_move(simulated_board, move)
                eval = self.minimax(depth - 1, possible_moves, adv_pos, simulated_board, max_step, True, alpha, beta)[0]
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def evaluate(self, chess_board, my_pos, adv_pos):
        # Maybe count number of squares on each side if game is over after this move?
        pass

    def apply_move(self, chess_board, move):
        # Extract position and direction from the move
        pos, dir = move

        # Update the chess board with the barrier
        self.set_barrier(pos, chess_board, dir)

        # Update the position of your player
        # Assuming 'move' also includes the next position of the player
        # If not, you'll need to calculate it based on the current position and direction
        self.my_pos = pos


