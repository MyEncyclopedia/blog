#!/usr/bin/env python
import random

from connect_n_gym.ConnectNGym import ConnectNGym


class BaseAgent(object):
    def __init__(self, strategy):
        self.strategy = strategy

    def act(self, state, available_actions):
        return random.choice(available_actions)


def play():
    env = ConnectNGym()
    agents = [BaseAgent('O'), BaseAgent('X')]

    state = env.reset()
    _, mark = state
    done = False
    env.show_board(False)
    agent_id = 0
    while not done:
        available_actions = env.get_available_actions()
        agent = agents[agent_id]
        action = agent.act(state, available_actions)
        state, reward, done, info = env.step(action)
        env.show_board(False)

        if done:
            break

        agent_id = (agent_id + 1) % 2

if __name__ == '__main__':
    play()
