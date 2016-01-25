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

        #create game environments and gameplaying threads
        self.env = game_env.Environment(gameworld(params.game_params), agent_params)
        self.img_size = self.env.get_img_size()
        (self.img_height, self.img_width) = self.img_size
        self.available_actions = self.env.get_actions()

        #create experience buffer
        self.experience = tf.RandomShuffleQueue(self.replay_memory,
                                    self.min_replay, dtypes=(tf.float32, tf.float32, tf.float32, tf.bool),
                                    shapes = ([self.img_height, self.img_width, self.history], [], [], []),  #state(rows,cols,history), action, reward, terminal
                                    name = 'experience_replay')

        #enqueue and dequeue ops to the experience memory
        self.dequeue_op = self.experience.dequeue()
        self.enq_state_placeholder = tf.placeholder(tf.float32, [self.img_height, self.img_width, self.history])
        self.action_placeholder = tf.placeholder(tf.float32, [])
        self.reward_placeholder = tf.placeholder(tf.float32, [])
        self.terminal_placeholder = tf.placeholder(tf.bool, [])
        self.enqueue_op = self.experience.enqueue((self.enq_state_placeholder, self.action_placeholder,
                                                    self.reward_placeholder, self.terminal_placeholder))

        #set up convnet
        net_params = params.net_params
        self.batch_size = net_params.batch_size
        net_params.output_dims = len(self.available_actions)
        net_params.history = self.history
        net_params.img_size = self.img_size
        (net_params.img_height, net_params.img_width) = self.img_size
        self.net_state_placeholder = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.history])
        self.net = convnet.ConvNetGenerator(net_params, self.net_state_placeholder)




    def perceive(self):
        """
        method for collecting game playing experience. Can be run multithreaded with
        different game sessions. Enqueues all sessions to a RandomShuffleQueue
        """
        try:
            state = self.env.get_state()     #get current state from environment
            #pick best action according to convnet
            action_values = self.net.logits.eval(feed_dict={self.net_state_placeholder:  np.expand_dims(state, axis=0)})
            max_a = np.argmax(action_values)
            reward = 1
            terminal = False
            #insert into queue
            self.sess.run(self.enqueue_op, feed_dict={self.enq_state_placeholder: state,
                                                          self.action_placeholder: max_a,
                                                          self.reward_placeholder: reward,
                                                          self.terminal_placeholder: terminal})
        except Exception as e:
            print e


    def get_action(self, state):
        """returns action recommended by target network"""


    def train(self):
        """draws minibatch from experience queue and updates current net"""
        dequed = self.experience.dequeue()
        return dequed


#create session and agent
sess = tf.Session()
agent = dqn(sess, domains.fire_fighter)
sess.run(tf.initialize_all_variables())

steps = 0
with sess.as_default():
    while(steps < 100):
        agent.perceive()
        steps+= 1
    while (steps > 80):
        print sess.run(agent.train())
        steps-=1

# agent.coord.request_stop()
# agent.coord.join(enqueue_threads)
# for i in range(64):
#     print "iter " + str(i)
#     if agent.coord.should_stop():
#         break
#     print sess.run(agent.train())
#     print state.get_shape()
