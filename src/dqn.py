import tensorflow as tf
import numpy as np
import threading
import sys
import time

import params
import game_env
import convnet
import domains

class dqn(object):
    """deep q learning agent for arbitrary domains."""
    def __init__(self, sess, gameworld):
        super(dqn, self).__init__()
        #tensorflow session
        self.sess = sess

        #agent parameters
        agent_params = params.agent_params
        self.max_steps = agent_params.steps                 #max training steps
        self.steps = 0
        self.replay_memory = agent_params.replay_memory     #max length of memory queue
        self.min_replay = agent_params.min_replay           #min length of memory queue
        self.history = agent_params.history                 #no. of frames of memory in state
        self.num_gameplay_threads = agent_params.num_gameplay_threads

        #create game environments and gameplaying threads
        self.games = [game_env.Environment(gameworld(params.game_params), agent_params, _) for _ in range(self.num_gameplay_threads)]
        self.gameplay_threads = [threading.Thread(target=self.perceive, args=(self.games[thread],)) for thread in range(self.num_gameplay_threads)]
        self.img_size = self.games[0].get_img_size()
        (self.img_height, self.img_width) = self.img_size
        self.available_actions = self.games[0].get_actions()

        #create experience buffer
        # self.experience = tf.RandomShuffleQueue(self.replay_memory,
        #                             self.min_replay, dtypes=(tf.float32, tf.float32, tf.bool),
        #                             shapes = ([self.img_height, self.img_width, self.history], [1], [1]),  #image(2), history, reward, terminal_flag
        #                             name = 'experience_replay')
        self.experience = tf.RandomShuffleQueue(self.replay_memory,
                                    self.min_replay, dtypes=tf.float32,
                                    shapes = [self.img_height, self.img_width, self.history],  #image(2), history, reward, terminal_flag
                                    name = 'experience_replay')
        #enqueue and dequeue ops to the experience memory
        self.dequeue_op = self.experience.dequeue()
        self.coord = tf.train.Coordinator()
        self.state_placeholder = tf.placeholder(tf.float32, [self.img_height, self.img_width, self.history])
        self.reward_placeholder = tf.placeholder(tf.float32, [1])
        self.terminal_placeholder = tf.placeholder(tf.bool, [1])
        # self.enqueue_op = self.experience.enqueue((self.state_placeholder, self.reward_placeholder, self.terminal_placeholder))
        self.enqueue_op = self.experience.enqueue(self.state_placeholder)

        #set up convnet
        net_params = params.net_params
        self.batch_size = net_params.batch_size
        net_params.output_dims = len(self.available_actions)
        net_params.history = self.history
        net_params.img_size = self.img_size
        (net_params.img_height, net_params.img_width) = self.img_size
        self.net = convnet.ConvNetGenerator(net_params)




    def perceive(self, env):
        """
        main function.
        """
        with self.coord.stop_on_exception():
            while not self.coord.should_stop():
                state = env.get_state()
                reward = np.array([1])
                terminal = np.array([False])
                # self.sess.run(self.enqueue_op, feed_dict={self.state_placeholder: state,
                #                                                               self.reward_placeholder: reward,
                #                                                               self.terminal_placeholder: terminal})
                self.sess.run(self.enqueue_op, feed_dict={self.state_placeholder: state})

    def start_playing(self):
        """
        Starts the gameplay threads. Can only be called once for an agent!!
        Calling multiple times will raise an exception!!
        """
        for thread in agent.gameplay_threads:
            thread.start()

    def get_action(self, state):
        """returns action recommended by target network"""


    def train(self):
        """draws minibatch from experience queue and updates current net"""
        dequed = self.experience.dequeue()
        return dequed



sess = tf.Session()
sess.run(tf.initialize_all_variables())
agent = dqn(sess, domains.fire_fighter)

#start the experience collection!
agent.start_playing()


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
