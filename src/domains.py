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
        """current screen of the game"""
        self.counter += 1
        screen = self.counter*np.ones(self.screen_size)
        return screen

    # def create_screen(self, )
