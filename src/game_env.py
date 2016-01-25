from collections import deque
import numpy as np

class Environment(object):
    """Shell for simplyfying certain tasks for agent"""
    def __init__(self, game, params):
        #game parameters
        self.game = game
        self.img_size = game.get_dims()
        #history
        self.history = params.history
        self.screen_history = deque(maxlen=self.history + 1)
        for _ in range(self.screen_history.maxlen):
            self.screen_history.append(np.expand_dims(np.zeros(self.img_size), -1))

    def get_actions(self):
        return self.game.actions

    def get_num_actions(self):
        return len(self.game.actions)

    def get_img_size(self):
        return self.img_size

    def get_state(self):
        """current state of the agent in the game (concatenation of the last self.history frames)"""
        # print [self.screen_history[hist].shape for hist in range(self.history)]
        return np.concatenate([self.screen_history[hist] for hist in range(-1, -1-self.history, -1)], axis=2)

    def take_action(self, a):
        """take the action in the game, update history and return new state, reward and terminal"""
        #get new game screen
        (screen, reward, terminal) = self.game.execute_action(a)
        screen = np.expand_dims(screen, -1)
        self.screen_history.append(screen)
        return (self.get_state(), reward, terminal)
