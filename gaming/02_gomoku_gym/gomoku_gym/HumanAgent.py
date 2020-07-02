#!/usr/bin/env python
import sys

import click

from gomoku_gym.GomokuGym import GomokuEnv
from gym_tictactoe.env import TicTacToeEnv, agent_by_mark, next_mark


class HumanAgent(object):
    def __init__(self, mark):
        self.mark = mark

    def act(self, ava_actions):
        while True:
            uloc = input("Enter location[1-9], q for quit: ")
            if uloc.lower() == 'q':
                return None
            try:
                action = int(uloc) - 1
                if action not in ava_actions:
                    raise ValueError()
            except ValueError:
                print("Illegal location: '{}'".format(uloc))
            else:
                break

        return action


def play():
    env = GomokuEnv()
    agents = [HumanAgent('O'), HumanAgent('X')]
    episode = 0
    while True:
        env.reset()
        # _, mark = state
        done = False
        while not done:
            action = env.render()
            # ava_actions = env.available_actions()
            if action is None:
                sys.exit()

            state, reward, done, info = env.step(action)

            # print('')
            # env.render()
            if done:
                # env.show_result(True, mark, reward)
                break
            else:
                _, mark = state
        episode += 1


if __name__ == '__main__':
    play()
