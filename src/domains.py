"""
Test domains for deep transfer
"""
import tensorflow as tf

class fire_fighter(object):
    """
    Taxi cab style domain
    """
    actions = ['Left', 'Right', 'Up', 'Down']
    def __init__(self, params, num):
        self.screen_size = params.img_size
        self.game_num = num
        self.counter = tf.Variable(0.0, trainable=False)

    def grab_screen(self):
        """current screen of the game"""
        self.counter = self.counter.assign_add(1)
        screen = self.counter*tf.ones(self.screen_size)
        screen = tf.expand_dims(screen, -1)
        print screen.get_shape().as_list()

        screen = tf.concat(2, [screen, tf.expand_dims((self.counter - 1)*tf.ones(self.screen_size), -1)])
        screen = tf.concat(2, [screen, tf.expand_dims(self.game_num*tf.ones(self.screen_size), -1)])

        return screen
