import gym
from gym import spaces

from gomoku_gym.PyGameBoard import GameBoard


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
        # self.board = [0] * self.grid_num * self.grid_num
        self.done = False

        self.board_game = GameBoard(board_num=self.grid_num, connect_num=self.connect_num)
        return self._get_obs()

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
        assert self.action_space.contains(action)

        loc = action
        if self.done:
            return self._get_obs(), 0, True, None

        reward = NO_REWARD
        # place
        self.board[loc] = tocode(self.mark)
        status = check_game_status(self.board)
        logging.debug("check_game_status board {} mark '{}'"
                      " status {}".format(self.board, self.mark, status))
        if status >= 0:
            self.done = True
            if status in [1, 2]:
                # always called by self
                reward = O_REWARD if self.mark == 'O' else X_REWARD

        # switch turn
        self.mark = next_mark(self.mark)
        return self._get_obs(), reward, self.done, None

    def _get_obs(self):
        return tuple(self.board), self.mark

    def render(self, mode='human', close=False):
        self.game.chessboard.action_done = False
        while not self.game.chessboard.action_done:
            self.game.update()
            self.game.draw()
            self.game.clock.tick(60)
        print('quit render')
        return self.id
        # if close:
        #     return
        # if mode == 'human':
        #     self._show_board(print)  # NOQA
        #     print('')
        # else:
        #     self._show_board(logging.info)
        #     logging.info('')

    def available_actions(self):
        return [i for i, c in enumerate(self.board) if c == 0]