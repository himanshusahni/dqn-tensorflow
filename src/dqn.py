import tensorflow as tf
import numpy as np
import threading
import sys
import time
import random
import traceback
from collections import deque

import params
import convnet
import domains

class dqn():
    """deep q learning agent"""

    def __init__(self, sess, game_env):
        #firstly set up all parameters
        self.max_steps = params.steps                 #max training steps
        self.replay_memory = params.replay_memory     #max length of memory queue
        self.min_replay = params.min_replay           #min length of memory queue
        self.history = params.history                 #no. of frames of memory in state
        self.gamma = params.gamma                     #discount factor
        self.ep = params.ep                           #exploration epsilon
        self.ep_delta = (params.ep - params.ep_end)/params.ep_endt
        self.learn_start = params.learn_start         #steps after which epsilon gets annealed
        self.ep_endt = params.ep_endt                 #steps after which epsilon stops annealing
        self.batch_size = params.batch_size           #training batch size
        self.clip_delta = params.clip_delta           #loss clipping value (0 means no clipping)

        #learning rate decay
        self.global_step = tf.Variable(0, trainable=False)   #number of training steps
        if params.lr_anneal:
            self.learning_rate = tf.train.exponential_decay(params.lr, self.global_step, params.lr_anneal, 0.96, staircase=True)
        else:
            self.learning_rate = params.lr                #learning rate

        self.sess = sess    #tensorflow session

        #game environment
        self.env = game_env
        self.state = self.env.new_game()
        self.img_size = self.env.get_img_size()
        (self.img_height, self.img_width) = self.img_size
        self.available_actions = self.env.get_actions()
        self.num_actions = len(self.available_actions)

        #create experience buffer
        self.experience = deque(maxlen=self.replay_memory)

        #create train and target networks
        with tf.variable_scope("train") as self.train_scope:
            self.train_net = convnet.ConvNetGenerator(self.img_size, self.num_actions, trainable=True)
        with tf.variable_scope("target") as self.target_scope:
            self.target_net = convnet.ConvNetGenerator(self.img_size, self.num_actions, trainable=False)

        #ops to train network
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        if params.summary > 0:
            tf.scalar_summary('learning_rate', self.learning_rate)

        #get loss
        self.actions_placeholder = tf.placeholder(tf.float32, [None, self.num_actions])
        self.target_placeholder = tf.placeholder(tf.float32, [None])
        self.Q_train = self.train_net.logits
        self.Q_train_actions = tf.reduce_sum(tf.mul(self.Q_train, self.actions_placeholder), reduction_indices=[1])
        self.diff = self.target_placeholder - self.Q_train_actions
    	#diff_abs = tf.abs(self.diff)
    	#diff_clipped = tf.clip_by_value(diff_abs, 0, 1)
    	#linear_part = diff_abs - diff_clipped
    	#quadratic_part = tf.square(diff_clipped)
    	#total = quadratic_part + linear_part
    	#self.loss = tf.reduce_mean(total)
    	#self.opt_op = tf.train.AdamOptimizer(1e-6).minimize(self.loss)
        # self.clipped_loss_vec = tf.clip_by_value(self.target_placeholder - self.Q_train_actions, -self.clip_delta, self.clip_delta)
        self.loss = tf.reduce_mean(tf.square(self.diff))
        if params.summary > 0:
            tf.scalar_summary("loss", self.loss)

        #create the gradient descent op
        #grads_and_vars = self.opt.compute_gradients(self.loss)
        #capped_grads_and_vars = [(tf.clip_by_value(g, -self.clip_delta, self.clip_delta), v) for g, v in grads_and_vars]    #gradient capping
        #save gradients
        # gradient_summaries = [tf.histogram_summary("grad - " + v.name, g) for g, v in grads_and_vars]
        self.train_op = self.opt.minimize(self.loss, global_step=self.global_step)


    def perceive(self, valid = False):
        """
        method for collecting game playing experience. Can be run multithreaded with
        different game sessions. Enqueues all sessions to a RandomShuffleQueue
        """
        try:
            #e-greedy action selection
            if not valid:
                chosen_ep = self.ep
            else:
                chosen_ep = self.valid_ep
            if (random.random() < chosen_ep):
                chosen_a = random.randint(0,self.num_actions - 1)
            else:
                #pick best action according to convnet on current state
                action_values = self.train_net.logits.eval(feed_dict={self.train_net.state_placeholder: np.expand_dims(self.state, axis=0)})
                max_a = np.argmax(action_values)
                chosen_a = max_a

            #execute that action in the environment,
            (next_state, reward, terminal) = self.env.take_action(chosen_a, valid)
            action_one_hot = np.zeros(self.num_actions)
            action_one_hot[chosen_a] = 1.0
            #insert into queue
            if not valid:
                self.experience.append((self.state, action_one_hot, reward, next_state, terminal))
            self.state = next_state
            #start a new game if terminal
            if terminal:
                self.state = self.env.new_game()
            return (reward, terminal)
        except Exception as e:
            traceback.print_exc()
            sys.exit()


    # def getQUpdate(self, states, actions, rewards, next_states, terminals):
    #     """calulcates gradients on a minibatch"""
    #     #get max action for next state: Q_target
    #     with tf.variable_scope(self.target_scope, reuse = True):
    #         Q_target = self.target_net.inference(next_states)
    #         Q_target_max_pre = tf.reduce_max(Q_target, reduction_indices=[1])
    #     #terminal states have Q=0, Q_target_max = (1-terminal)*Q_target_max
    #     Q_target_max = tf.mul(Q_target_max_pre, tf.add(1.0, tf.mul(tf.cast(terminals, tf.float32), -1)))
    #     #discount: (1-terminal)*gamma*Q_target_max
    #     Q_target_disc = tf.mul(Q_target_max , self.gamma)
    #     #total estimated value : r + (1-terminal)*gamma*Q_target_max
    #     est_value = tf.add(rewards, Q_target_disc)
    #     #get action values for current state: Q_train
    #     with tf.variable_scope(self.train_scope, reuse = True):
    #         Q_train = self.train_net.inference(states)
    #     #first zero out all the action values except the one taken
    #     Q_train_one_hot = tf.mul(Q_train, tf.cast(actions, tf.float32))
    #     #now gather them using sum
    #     Q_train_actions = tf.reduce_sum(Q_train_one_hot, reduction_indices=[1])
    #     #final targets = r + (1-terminal)*gamma*Q_target_max - Q_train_actions
    #     targets = tf.add(est_value, tf.mul(Q_train_actions, -1))
    #     return targets
    #
    # def qLearnMinibatch(self, global_step):
    #     """draws minibatch from experience queue and updates current net"""
    #     try:
    #         #get loss
    #         loss_all = tf.square(targets)
    #         loss = tf.reduce_mean(loss_all)
    #         loss_summary = tf.scalar_summary("loss", tf.reduce_sum(tf.mul(loss, loss)))
    #         #create the gradient descent op
    #         grads_and_vars = self.opt.compute_gradients(loss)
    #         capped_grads_and_vars = [(tf.clip_by_value(g, -self.clip_delta, self.clip_delta), v) for g, v in grads_and_vars]    #gradient capping
    #         #save gradients
    #         gradient_summaries = [tf.histogram_summary("grad - " + v.name, g) for g, v in capped_grads_and_vars]
    #         self.opt_op = self.opt.apply_gradients(capped_grads_and_vars, global_step=global_step)
    #         tf.scalar_summary('global_step', global_step)
    #         return self.opt_op
    #
    #     except Exception as e:
    #         traceback.print_exc()
    #         sys.exit()
