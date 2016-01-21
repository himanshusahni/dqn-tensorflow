import tensorflow as tf
import numpy as np
import threading
import sys
import time

import params
import game_env
import convnet

class dqn(object):
    """deep q learning agent for arbitrary domains."""
    def __init__(self, params, sess):
        self.sess = sess
        super(dqn, self).__init__()
        self.available_threads = params.agent.available_threads
        net_params = params.net
        self.max_steps = params.agent.steps
        #create the convnet

        # self.net = convnet.ConvNetGenerator(net_params)
        self.batch_size = net_params.batch_size
        self.img_height = net_params.img_height
        self.img_width = net_params.img_width

        #keep track of history
        self.history = net_params.history   #TODO: look for a better way to buffer history
        self.steps = 0
        #create experience buffer
        self.replay_memory = params.agent.replay_memory     #max length of memory queue
        self.min_replay = params.agent.min_replay       #min length of memory queue
        # self.experience = tf.RandomShuffleQueue(self.replay_memory,
        #                             self.min_replay, dtypes=(tf.float32, tf.float32, tf.bool),
        #                             shapes = ([self.img_height, self.img_width, self.history], [1], [1]),  #image(2), history, reward, terminal_flag
        #                             name = 'experience_replay')
        self.experience = tf.RandomShuffleQueue(self.replay_memory,
                                    self.min_replay, dtypes=tf.float32,
                                    shapes = [self.img_height, self.img_width, self.history],  #image(2), history, reward, terminal_flag
                                    name = 'experience_replay')
        self.dequeue_op = self.experience.dequeue()
        self.coord = tf.train.Coordinator()
        self.state_placeholder = tf.placeholder(tf.float32, [self.img_height, self.img_width, self.history])
        self.reward_placeholder = tf.placeholder(tf.float32, [1])
        self.terminal_placeholder = tf.placeholder(tf.bool, [1])
        # self.enqueue_op = self.experience.enqueue((self.state_placeholder, self.reward_placeholder, self.terminal_placeholder))
        self.enqueue_op = self.experience.enqueue(self.state_placeholder)
    def perceive(self, game_):
        """
        main function.
        """
        with self.coord.stop_on_exception():
            while not self.coord.should_stop():
                try:
                    state = game_.grab_screen()
                    reward = np.array([1])
                    terminal = np.array([False])
                    # self.sess.run(self.enqueue_op, feed_dict={self.state_placeholder: state,
                    #                                                               self.reward_placeholder: reward,
                    #                                                               self.terminal_placeholder: terminal})
                    self.sess.run(self.enqueue_op, feed_dict={self.state_placeholder: state})
                except Exception as e:
                    print e

    def take_game_action(self, action, game):
        """tells the game environment to execute an action. does not return anything"""

    def get_action(self, state):
        """returns action recommended by target network"""


    def train(self):
        """draws minibatch from experience queue and updates current net"""
        dequed = self.experience.dequeue()
        return dequed

    def get_complete_state(self, rawstate, thread_num):
        """
        adds new screen grab to history buffer and returns complete state.
        In the very beginning just appends tensors of all zeros to make up the
        self.history channel tensors.
        """
        #increment buffer position
        self.buffer_pos[thread_num] += 1
        self.buffer_pos[thread_num] %= self.len_buffer
        #add screen to buffer
        self.history[thread_num][self.buffer_pos[thread_num]] = tf.expand_dims(rawstate, -1)
        #grab last 'history' screens into a state
        state = tf.identity(self.history[thread_num][self.buffer_pos[thread_num]])  #TODO:can defiitely do better than this concaatination at each step
        back_counter = self.buffer_pos[thread_num]
        for i in range(self.history - 1):
            if back_counter == 0:
                back_counter = self.len_buffer
            back_counter -= 1
            state = tf.concat(2, [self.history[back_counter], state])       #concatenate in the channel dimension
        return state



sess = tf.Session()

sess.run(tf.initialize_all_variables())
agent = dqn(params, sess)
#start the experience collection!
games = [game_env.game(params.net, i) for i in range(params.agent.available_threads)]
for _ in range(agent.available_threads):
    threading.Thread(target=agent.perceive, args=(games[_],)).start()


with sess.as_default():
    while(1):
        print sess.run(agent.dequeue_op)
        print "break!"
        # print agent.experience.size().eval()
        time.sleep(.5)

# agent.coord.request_stop()
# agent.coord.join(enqueue_threads)
# for i in range(64):
#     print "iter " + str(i)
#     if agent.coord.should_stop():
#         break
#     print sess.run(agent.train())
#     print state.get_shape()
