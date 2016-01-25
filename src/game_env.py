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
        self.screen_history = deque([np.expand_dims(0*np.ones(self.img_size), -1),np.expand_dims(1*np.ones(self.img_size), -1)], maxlen=self.history)

    def get_actions(self):
        return self.game.actions

    def get_num_actions(self):
        return len(self.game.actions)

    def get_img_size(self):
        return self.img_size

    def get_state(self):
        """current screen of the game"""
        screen = np.expand_dims(self.game.grab_screen(), -1)
        self.screen_history.append(screen)
        # print [self.screen_history[hist].shape for hist in range(self.history)]
        state = np.concatenate([self.screen_history[hist] for hist in range(self.history)], axis=2)
        return state
