from collections import deque
import numpy as np



class game(object):
    def __init__(self, params, num):
        self.actions = ['Left', 'Right', 'Up', 'Down']
        self.screen_size = params.img_size
        self.history = params.history
        self.counter = 0
        self.screen_history = deque([np.expand_dims(0*np.ones(params.img_size), -1),np.expand_dims(1*np.ones(params.img_size), -1)], maxlen=self.history)
        self.num = num
    def grab_screen(self):
        """current screen of the game"""
        self.counter += 1
        screen = np.expand_dims(self.counter*np.ones(self.screen_size), -1)
        self.screen_history.append(screen)
        # print [self.screen_history[hist].shape for hist in range(self.history)]
        state = np.concatenate([self.screen_history[hist] for hist in range(self.history)], axis=2)
        state[:][-1][:] = self.num
        return state
