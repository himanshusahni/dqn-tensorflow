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
        self.gamma = agent_params.gamma

        #create game environments and gameplaying threads
        self.env = game_env.Environment(gameworld(params.game_params), agent_params)
        self.state = self.env.get_state()
        self.img_size = self.env.get_img_size()
        (self.img_height, self.img_width) = self.img_size
        self.available_actions = self.env.get_actions()
        self.num_actions = len(self.available_actions)
        self.batch_size = params.net_params.batch_size

        #create experience buffer
        self.experience = tf.RandomShuffleQueue(self.replay_memory,
                                    self.min_replay, dtypes=(tf.float32, tf.int32, tf.float32, tf.float32, tf.bool),
                                    #state(rows,cols,history), action, reward, next_state(rows,cols,history), terminal
                                    shapes = ([self.img_height, self.img_width, self.history], [self.num_actions], [], [self.img_height, self.img_width, self.history], []),
                                    name = 'experience_replay')

        #enqueue op to the experience memory
        self.enq_state_placeholder = tf.placeholder(tf.float32, [self.img_height, self.img_width, self.history])
        self.action_placeholder = tf.placeholder(tf.int32, [self.num_actions])
        self.reward_placeholder = tf.placeholder(tf.float32, [])
        self.next_state_placeholder = tf.placeholder(tf.float32, [self.img_height, self.img_width, self.history])
        self.terminal_placeholder = tf.placeholder(tf.bool, [])
        self.enqueue_op = self.experience.enqueue((self.enq_state_placeholder, self.action_placeholder,
                                                    self.reward_placeholder, self.next_state_placeholder,
                                                    self.terminal_placeholder))


        #set up convnet
        net_params = params.net_params
        net_params.output_dims = self.num_actions
        net_params.history = self.history
        net_params.img_size = self.img_size
        (net_params.img_height, net_params.img_width) = self.img_size
        self.batch_state_placeholder = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.history])
        with tf.variable_scope("train") as self.train_scope:
            self.train_net = convnet.ConvNetGenerator(net_params, self.batch_state_placeholder)
        with tf.variable_scope("target") as self.target_scope:
            self.target_net = convnet.ConvNetGenerator(net_params, self.batch_state_placeholder)

        #ops to train network

    def perceive(self):
        """
        method for collecting game playing experience. Can be run multithreaded with
        different game sessions. Enqueues all sessions to a RandomShuffleQueue
        """
        try:
            #pick best action according to convnet on current state
            action_values = self.train_net.logits.eval(feed_dict={self.batch_state_placeholder: np.expand_dims(self.state, axis=0)})
            max_a = np.argmax(action_values)
            # max_a = np.random.randint(0,3)
            #execute that action in the environment,
            (next_state, reward, terminal) = self.env.take_action(max_a)
            action_one_hot = np.zeros(self.num_actions, dtype='int32')
            action_one_hot[max_a] = 1
            #insert into queue
            self.sess.run(self.enqueue_op, feed_dict={self.enq_state_placeholder: self.state,
                                                          self.action_placeholder: action_one_hot,
                                                          self.reward_placeholder: reward,
                                                          self.next_state_placeholder: next_state,
                                                          self.terminal_placeholder: terminal})
            #update current state
            self.state = next_state
        except Exception as e:
            print e


    def getQUpdate(self, states, actions, rewards, next_states, terminals):
        """calulcates gradients on a minibatch"""
        #get max action for next state: Q_target
        with tf.variable_scope(self.target_scope, reuse = True):
            Q_target = self.target_net.inference(next_states)
            Q_target_max = tf.reduce_max(Q_target, reduction_indices=[1])
        #terminal states have Q=0, Q_target_max = (1-terminal)*Q_target_max
        Q_target_max = tf.mul(Q_target_max, tf.add(1.0, tf.mul(tf.cast(terminals, tf.float32), -1)))
        #discount: (1-terminal)*gamma*Q_target_max
        Q_target_disc = tf.mul(Q_target_max , self.gamma)
        #total estimated value : r + (1-terminal)*gamma*Q_target_max
        est_value = tf.add(rewards, Q_target_disc)
        #get action values for current state: Q_train
        with tf.variable_scope(self.train_scope, reuse = True):
            Q_train = self.train_net.inference(states)
        #first zero out all the action values except the one taken
        Q_train_one_hot = tf.mul(Q_train, actions)
        #now gather them using sum
        Q_train_actions = tf.reduce_sum(Q_train_one_hot, reduction_indices=[1])
        #final targets = r + (1-terminal)*gamma*Q_target_max - Q_train_actions
        targets = tf.add(est_value, tf.mul(Q_train_actions, -1))
        return dense_targets

    def qLearnMinibatch(self):
        """draws minibatch from experience queue and updates current net"""
        try:
            #draw minibatch = [states, actions, rewards, next_states, terminals]
            minibatch = self.experience.dequeue_many(self.batch_size)
            targets = self.getQUpdate(*minibatch)
            #TODO: verify if targets are the loss or the gradients, this is assuming it is loss








            return targets

        except Exception as e:
            print e


#create session and agent
sess = tf.Session()
agent = dqn(sess, domains.fire_fighter)
sess.run(tf.initialize_all_variables())

steps = 0
with sess.as_default():
    while(steps < 100):
        agent.perceive()
        steps+= 1
    while (steps > 90):
        print sess.run(agent.qLearnMinibatch())

        steps-=params.net_params.batch_size

# agent.coord.request_stop()
# agent.coord.join(enqueue_threads)
# for i in range(64):
#     print "iter " + str(i)
#     if agent.coord.should_stop():
#         break
#     print sess.run(agent.train())
#     print state.get_shape()
