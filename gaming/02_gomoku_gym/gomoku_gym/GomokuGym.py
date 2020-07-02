import gym
from gym import spaces

from gomoku_gym.PyGameBoard import GameBoard

REWARD_W = 1
REWARD_B = -1
REWARD_NONE = 0

class GomokuEnv(gym.Env):

    def __init__(self, grid_num=5, connect_num=3):
        self.grid_num = grid_num
        self.connect_num = connect_num
        self.action_space = spaces.Discrete(grid_num* grid_num)
        self.observation_space = spaces.Discrete(grid_num * grid_num)

        self.board_game = GameBoard(board_num=self.grid_num, connect_num=self.connect_num)

        self.seed()
        self.reset()

    def reset(self):
        self.done = False
        self.board_game = GameBoard(board_num=self.grid_num, connect_num=self.connect_num)
        return self.get_state()

    def step(self, action):
        """Step environment by action.

        Args:
            action (int): Location

        Returns:
            list: Obeservation
            int: Reward
            bool: Done
            dict: Additional information
        """
        # assert self.action_space.contains(action)

        r, c = action
        # loc = action
        # if self.done:
        #     return self._get_obs(), 0, True, None

        reward = REWARD_NONE
        # place
        self.board_game.set_piece(r, c)
        self.done = self.board_game.check_win()
        if self.board_game.game_over:
            reward = REWARD_W if self.board_game.piece == GameBoard.PIECE_W else REWARD_B
        self.board_game.switch()

        return self.get_state(), reward, self.done, None

    def get_state(self):
        return self.board_game.board, self.board_game.piece

    def render(self, mode='human', close=False):
        self.action = self.board_game.next_user_input()
        return self.action

        # if close:
        #     return
        # if mode == 'human':
        #     # self._show_board(print)  # NOQA
        #     print('')
        # else:
        #     pass
        #     # self._show_board(logging.info)
        #     # logging.info('')

    def available_actions(self):
        return [i for i, c in enumerate(self.board) if c == 0]