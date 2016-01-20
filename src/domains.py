"""
Test domains for deep transfer
"""
import tensorflow as tf
from collections import deque
class fire_fighter(object):
    """
    Taxi cab style domain
    """
    actions = ['Left', 'Right', 'Up', 'Down']
    def __init__(self, params, num):
        self.screen_size = params.img_size
        self.game_num = num
        self.counter = tf.Variable(2.0, trainable=False)
        history = [tf.expand_dims((self.counter - 1)*tf.ones(self.screen_size), -1), tf.expand_dims((self.counter-2)*tf.ones(self.screen_size), -1)]
        self.history = deque(maxlen = 3)
        self.history.append(tf.expand_dims((self.counter - 1)*tf.ones(self.screen_size), -1))
        self.history.append(tf.expand_dims((self.counter)*tf.ones(self.screen_size), -1))
    def grab_screen(self):
        """current screen of the game"""
        self.counter = self.counter.assign_add(1)
        screen = self.counter*tf.ones(self.screen_size)
        screen = tf.expand_dims(screen, -1)
        print screen.get_shape().as_list()
        self.history.append(screen)
        screen = tf.concat(2, [screen, self.history[-2]])
        screen = tf.concat(2, [screen, self.history[-3]])
        screen = tf.concat(2, [screen, tf.expand_dims((self.game_num)*tf.ones(self.screen_size), -1)])
        return screen

    # def create_screen(self, )
