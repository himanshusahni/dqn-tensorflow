"""
Test domains for deep transfer
"""
import numpy as np
from collections import deque
class fire_fighter(object):
    """
    Taxi cab style domain
    """
    actions = ['Left', 'Right', 'Up', 'Down']
    def __init__(self, params):
        self.screen_size = params.img_size
        self.counter = 0

    def grab_screen(self):
        """current screen of the game
        returns: numpy nd-array of shape screen_size"""
        return self.counter*np.ones(self.screen_size)

    def get_dims(self):
        """row and column of screen in pixels"""
        return self.screen_size

    def execute_action(self, a):
        """takes action a in the game and gives new screen, reward and terminal
        if the new state is terminal, then start a new game.
        a: index of action to be executed
        returns: numpy.ndarray, shape=self.screen_size: new game screen (from grab_screen)
                 float32: reward for executing given action
                 boolean: whether the new state is a terminal state or not
        """
        self.counter += 1
        return (self.grab_screen(), 1, False)
