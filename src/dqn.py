import tensorflow as tf
import numpy as np
import threading
import sys
import time
import random

import params
import game_env
import convnet
import domains

class dqn(object):
    """deep q learning agent for arbitrary domains."""
    def __init__(self, sess, gameworld):
        super(dqn, self).__init__()
        self.sess = sess    #tensorflow session

        #agent parameters
        agent_params = params.agent_params
        self.max_steps = agent_params.steps                 #max training steps
        self.steps = 0
        self.replay_memory = agent_params.replay_memory     #max length of memory queue
        self.min_replay = agent_params.min_replay           #min length of memory queue
        self.history = agent_params.history                 #no. of frames of memory in state
        self.gamma = agent_params.gamma                     #discount factor
        self.ep_delta = (agent_params.ep - agent_params.ep_end)/(agent_params.ep_endt - agent_params.learn_start)
        self.learn_start = agent_params.learn_start         #steps after which epsilon gets annealed
        self.ep_endt = agent_params.ep_endt                 #steps after which epsilon stops annealing
        self.valid_ep = agent_params.valid_ep               #epsilon to use during validation runs
        self.num_gameplay_threads = agent_params.num_gameplay_threads

        #create game environments and gameplaying threads
        self.envs = [game_env.Environment(gameworld, thread_num) for thread_num in range(self.num_gameplay_threads)]
        self.gameplay_threads = [threading.Thread(target=self.game_driver, args=(self.envs[thread_num], self.sess)) for thread_num in range(self.num_gameplay_threads)]
        self.img_size = params.game_params.img_size
        (self.img_height, self.img_width) = self.img_size
        self.available_actions = self.envs[0].get_actions()
        self.num_actions = len(self.available_actions)
        self.coord = tf.train.Coordinator()

        #create experience buffer
        self.experience = tf.RandomShuffleQueue(self.replay_memory, self.min_replay,
                                    dtypes=(tf.float32, tf.int32, tf.float32, tf.float32, tf.bool),
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
        self.batch_size = net_params.batch_size
        self.clip_delta = net_params.clip_delta
        net_params.output_dims = self.num_actions
        net_params.history = self.history
        net_params.img_size = self.img_size
        (net_params.img_height, net_params.img_width) = self.img_size
        self.batch_state_placeholder = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.history])
        with tf.variable_scope("train") as self.train_scope:
            self.train_net = convnet.ConvNetGenerator(net_params, self.batch_state_placeholder, trainable=True)
        with tf.variable_scope("target") as self.target_scope:
            self.target_net = convnet.ConvNetGenerator(net_params, self.batch_state_placeholder, trainable=False)

        #ops to train network
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=net_params.lr)

    def start_playing(self):
        """
        Starts the gameplay threads. Can only be called once for an agent!!
        Calling multiple times will raise an exception!!
        """
        for thread in self.gameplay_threads:
            thread.start()

    def game_driver(self, game, sess):
        with self.coord.stop_on_exception():
            while not self.coord.should_stop():
                self.perceive(game, sess)


    def perceive(self, game, sess, valid = False):
        """
        method for collecting game playing experience. Can be run multithreaded with
        different game sessions. Enqueues all sessions to a RandomShuffleQueue
        """
        try:
            with sess.as_default():
                # max_a = np.random.randint(0,self.num_actions - 1)
                #e-greedy action selection
                if not valid:
                    if self.learn_start < game.counter < self.ep_endt:
                        game.ep -= self.ep_delta    #anneal epsilon
                    chosen_ep = game.ep
                else:
                    chosen_ep = self.valid_ep
                if (random.random() < chosen_ep):
                    chosen_a = np.random.randint(0,self.num_actions - 1)
                else:
                    #pick best action according to convnet on current state
                    action_values = self.train_net.logits.eval(feed_dict={self.batch_state_placeholder: np.expand_dims(game.get_state(), axis=0)})
                    max_a = np.argmax(action_values)
                    chosen_a = max_a

                #execute that action in the environment,
                (next_state, reward, terminal) = game.take_action(chosen_a, valid)
                action_one_hot = np.zeros(self.num_actions, dtype='int32')
                action_one_hot[chosen_a] = 1
                #insert into queue
                if not valid:
                    sess.run(self.enqueue_op, feed_dict={self.enq_state_placeholder: game.get_state(),
                                                              self.action_placeholder: action_one_hot,
                                                              self.reward_placeholder: reward,
                                                              self.next_state_placeholder: next_state,
                                                              self.terminal_placeholder: terminal})
                #start a new game if terminal
                if terminal:
                    game.new_game()
                return (reward, terminal)
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
        Q_train_one_hot = tf.mul(Q_train, tf.cast(actions, tf.float32))
        #now gather them using sum
        Q_train_actions = tf.reduce_sum(Q_train_one_hot, reduction_indices=[1])
        #final targets = r + (1-terminal)*gamma*Q_target_max - Q_train_actions
        targets = tf.add(est_value, tf.mul(Q_train_actions, -1))
        return targets

    def qLearnMinibatch(self):
        """draws minibatch from experience queue and updates current net"""
        try:
            #draw minibatch = [states, actions, rewards, next_states, terminals]
            minibatch = self.experience.dequeue_many(self.batch_size)
            targets = self.getQUpdate(*minibatch)
            #get loss
            loss = tf.square(targets)
            loss_summary = tf.scalar_summary("loss", tf.reduce_sum(tf.mul(loss, loss)))
            #create the gradient descent op
            grads_and_vars = self.opt.compute_gradients(loss)
            capped_grads_and_vars = [(tf.clip_by_value(g, -self.clip_delta, self.clip_delta), v) for g, v in grads_and_vars]    #gradient capping
            #save gradients
            gradient_summaries = [tf.histogram_summary("grad - " + v.name, g) for g, v in capped_grads_and_vars]
            self.opt_op = self.opt.apply_gradients(capped_grads_and_vars)
            return self.opt_op

        except Exception as e:
            print e


if __name__ == "__main__":
    #create session and agent
    sess = tf.Session()
    agent = dqn(sess, domains.fire_fighter)
    sess.run(tf.initialize_all_variables())
    train_op = agent.qLearnMinibatch()
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("./logs", sess.graph_def)
    saver = tf.train.Saver()
    steps = 0
    valid_game = game_env.Environment(domains.fire_fighter, -1)
    hist_size_op = agent.experience.size()
    num_hist = 0
    try:
        with sess.as_default():
            #run 10,000 steps in the beginning random
            print "STARTING AGENT GAMEPLAY!"
            agent.start_playing()
            while not (num_hist > params.agent_params.learn_start):
                time.sleep(1)
                num_hist = hist_size_op.eval()
                print "Size of history: " + str(num_hist)
            print "DONE RANDOM PLAY"
            while(steps < params.agent_params.steps):
                #train a minibatch
                result = sess.run([merged,train_op])
                if steps % params.agent_params.log_freq == 0:
                    summary_str = result[0]
                    writer.add_summary(summary_str, steps)
                    print "Size of history: " + str(hist_size_op.eval())
                    print "Training steps executed: " + str(steps)
                steps += 1
                #copy over target network if needed
                if steps % params.agent_params.target_q == 0:
                    print "COPYING TARGET NETWORK OVER AT " + str(steps)
                    agent.target_net.copy_weights(agent.train_net.var_dir, sess)
                #validate!
                if (steps >= params.agent_params.valid_start) and (steps % params.agent_params.valid_freq == 0):
                    print "Starting a validation run!"
                    valid_game.new_game()  #terminate current game and set up a new validation game
                    avg_r = 0.0
                    for v_episodes in range(params.agent_params.valid_episodes):
                        print "RUNNING VALIDATION EPISODE " + str(v_episodes)
                        ep_r = 0.0
                        terminal = False
                        ep_steps = 0
                        while not terminal:
                            ep_steps += 1
                            r, terminal = agent.perceive(valid_game, sess, valid = True)
                            ep_r += r
                        print "EPISODE ENDED. EPISODE REWARD " + str(ep_r)
                        avg_r += ep_r
                    avg_r /= params.agent_params.valid_episodes
                    print "Validation reward after " + str(steps) + " steps is " + str(avg_r)
                #save
                if (steps % params.agent_params.save_freq == 0):
                    print "SAVING MODEL AFTER " + str(steps) + " ..."
                    saver.save(sess, "./models/model", global_step = steps)
    except Exception as e:
        print e
    finally:
        agent.coord.request_stop()
        agent.coord.join(agent.gameplay_threads)
