import tensorflow as tf
import numpy as np
import threading
import sys
import time
import random
import traceback
from collections import deque

import params
import game_env
import convnet
import domains

class dqn(object):
    """deep q learning agent for arbitrary domains."""
    def __init__(self, sess, gameworld, global_step):
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
        #create game environments and gameplaying threads
        self.env = game_env.Environment(gameworld)
        self.img_size = self.env.get_img_size()
        (self.img_height, self.img_width) = self.img_size
        self.available_actions = self.env.get_actions()
        self.num_actions = len(self.available_actions)
        #create experience buffer
        self.experience = deque(maxlen=agent_params.replay_memory)

        #set up convnet
        net_params = params.net_params
        self.batch_size = net_params.batch_size
        self.clip_delta = net_params.clip_delta
        net_params.output_dims = self.num_actions
        net_params.history = self.history
        net_params.img_size = self.img_size
        (net_params.img_height, net_params.img_width) = self.img_size
        with tf.variable_scope("train") as self.train_scope:
            self.train_net = convnet.ConvNetGenerator(net_params, trainable=True)
        with tf.variable_scope("target") as self.target_scope:
            self.target_net = convnet.ConvNetGenerator(net_params, trainable=False)

        #ops to train network
        learning_rate = tf.train.exponential_decay(net_params.lr, global_step, net_params.lr_step, 0.96, staircase=True)
        self.opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        tf.scalar_summary('learning_rate', learning_rate)
        #get loss
        self.train_actions_placeholder = tf.placeholder(tf.float32, [None, self.num_actions])
        self.target_placeholder = tf.placeholder(tf.float32, [self.batch_size])
        Q_train = self.train_net.logits
        Q_train_actions = tf.reduce_sum(tf.mul(Q_train, self.train_actions_placeholder))
        self.loss = tf.reduce_mean(tf.square(self.target_placeholder - Q_train_actions))
        loss_summary = tf.scalar_summary("loss", self.loss)
        #create the gradient descent op
        grads_and_vars = self.opt.compute_gradients(self.loss)
        capped_grads_and_vars = [(tf.clip_by_value(g, -self.clip_delta, self.clip_delta), v) for g, v in grads_and_vars]    #gradient capping
        #save gradients
        gradient_summaries = [tf.histogram_summary("grad - " + v.name, g) for g, v in capped_grads_and_vars]
        self.opt_op = self.opt.apply_gradients(capped_grads_and_vars, global_step=global_step)
        tf.scalar_summary('global_step', global_step)

    def perceive(self, dumpFile, valid = False):
        """
        method for collecting game playing experience. Can be run multithreaded with
        different game sessions. Enqueues all sessions to a RandomShuffleQueue
        """
        try:
            # max_a = np.random.randint(0,self.num_actions - 1)
            #e-greedy action selection
            if not valid:
                if self.learn_start < self.env.counter < self.ep_endt:
                    self.env.ep -= self.ep_delta    #anneal epsilon
                chosen_ep = self.env.ep
            else:
                chosen_ep = self.valid_ep
            if (random.random() < chosen_ep):
                chosen_a = np.random.randint(0,self.num_actions - 1)
            else:
                #pick best action according to convnet on current state
                action_values = self.train_net.logits.eval(feed_dict={self.batch_state_placeholder: np.expand_dims(self.env.get_state(), axis=0)})
                max_a = np.argmax(action_values)
                chosen_a = max_a

            #execute that action in the environment,
            (next_state, reward, terminal) = self.env.take_action(chosen_a, valid)
            action_one_hot = np.zeros(self.num_actions)
            action_one_hot[chosen_a] = 1.0
            #insert into queue
            if not valid:
                self.experience.append((self.env.get_state(), action_one_hot, reward, next_state, terminal))

            #start a new game if terminal
            if terminal:
                dumpFile.write(str(self.env.counter) + " " + " " + str(self.env.episodes) + " " +
                                                str(self.env.ep_reward) + " " + str(self.env.ep) + " " + str(self.env.training_steps) + "\n")
                self.requestNewGame()
            return (reward, terminal)
        except Exception as e:
            traceback.print_exc()
            sys.exit()



    def requestNewGame(self):
        """starts a new game from the environment"""
        self.env.new_game()

    def getQUpdate(self, states, actions, rewards, next_states, terminals):
        """calulcates gradients on a minibatch"""
        #get max action for next state: Q_target
        with tf.variable_scope(self.target_scope, reuse = True):
            Q_target = self.target_net.inference(next_states)
            Q_target_max_pre = tf.reduce_max(Q_target, reduction_indices=[1])
        #terminal states have Q=0, Q_target_max = (1-terminal)*Q_target_max
        Q_target_max = tf.mul(Q_target_max_pre, tf.add(1.0, tf.mul(tf.cast(terminals, tf.float32), -1)))
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

    def qLearnMinibatch(self, global_step):
        """draws minibatch from experience queue and updates current net"""
        try:
            #get loss
            loss_all = tf.square(targets)
            loss = tf.reduce_mean(loss_all)
            loss_summary = tf.scalar_summary("loss", tf.reduce_sum(tf.mul(loss, loss)))
            #create the gradient descent op
            grads_and_vars = self.opt.compute_gradients(loss)
            capped_grads_and_vars = [(tf.clip_by_value(g, -self.clip_delta, self.clip_delta), v) for g, v in grads_and_vars]    #gradient capping
            #save gradients
            gradient_summaries = [tf.histogram_summary("grad - " + v.name, g) for g, v in capped_grads_and_vars]
            self.opt_op = self.opt.apply_gradients(capped_grads_and_vars, global_step=global_step)
            tf.scalar_summary('global_step', global_step)
            return self.opt_op

        except Exception as e:
            traceback.print_exc()
            sys.exit()


if __name__ == "__main__":
    #create session and agent
    sess = tf.Session()
    global_step = tf.Variable(0, trainable=False)
    agent = dqn(sess, domains.fire_fighter, global_step)
    #initialize everything
    sess.run(tf.initialize_all_variables())

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("./logs", sess.graph_def)
    saver = tf.train.Saver()
    valid_game = game_env.Environment(domains.fire_fighter)
    steps = 0
    try:
        with sess.as_default():
            #run random steps in the beginning
            with open("models/stats_dump.txt", 'wb') as dumpFile:
                print "STARTING RANDOM INITIALIZATIONS"
                while(steps < params.agent_params.learn_start):
                    agent.perceive(dumpFile)
                    steps += 1
                    num_hist = len(agent.experience)
                    if steps % params.agent_params.log_freq==0:
                        print "Size of history: " + str(num_hist)
                print "DONE RANDOM PLAY"
                start_time = time.time()
                avg_loss = 0
                while(steps < params.agent_params.steps):
                    #draw a minibatch
                    minibatch = random.sample(agent.experience, agent.batch_size)
                    states = np.array([minibatch[i][0] for i in range(agent.batch_size)]).astype(np.float32)
                    actions = np.array([minibatch[i][1] for i in range(agent.batch_size)]).astype(np.float32)
                    rewards = np.array([minibatch[i][2] for i in range(agent.batch_size)]).astype(np.float32)
                    next_states = np.array([minibatch[i][3] for i in range(agent.batch_size)]).astype(np.float32)
                    terminals = np.array([minibatch[i][4] for i in range(agent.batch_size)]).astype(np.float32)

                    #calculate the bellman target = r + gamma*(1-terminal)*max(Q_target)
                    Q_target = agent.target_net.logits.eval(feed_dict={agent.target_net.state_placeholder:next_states})
                    Q_target_max = np.amax(Q_target, axis=1)
                    Q_target_terminal = (1-terminals)*Q_target_max
                    Q_target_gamma = agent.gamma*Q_target_terminal
                    target = rewards + Q_target_gamma
                    (summary, result, loss) = sess.run([merged,agent.opt_op, agent.loss], feed_dict={agent.target_placeholder:target, agent.train_net.state_placeholder:states, agent.train_actions_placeholder: actions})
                    avg_loss += loss
                    #perceive the next state
                    agent.perceive(dumpFile)
                    steps += 1
                    #copy over target network if needed
                    if steps % params.agent_params.target_q == 0:
                        print "COPYING TARGET NETWORK OVER AT " + str(steps)
                        agent.target_net.copy_weights(agent.train_net.var_dir, sess)
                    ############################## all book keeping now ##############################
                    #save
                    if (steps % params.agent_params.save_freq == 0):
                        print "SAVING MODEL AFTER " + str(steps) + " ..."
                        saver.save(sess, "./models/model", global_step = steps)
                        ###DEBUGGING###
                        print "Dumping Minibatch!"
                        mini_op = agent.experience.dequeue_many(agent.batch_size)
                        states, actions, rewards, next_states, terminals = sess.run(mini_op)
                        np.save("models/states-" + str(steps), states)
                        np.save("models/actions-" + str(steps), actions)
                        np.save("models/rewards-" + str(steps), rewards)
                        np.save("models/next_states-" + str(steps), next_states)
                        np.save("models/terminals-" + str(steps), terminals)
                        print "SEARCHING FOR REWARD MINIBATCH"
                        satisfied = False
                        while (not satisfied):
                            print "Size of history: " + str(hist_size_op.eval())
                            states, actions, rewards, next_states, terminals = sess.run(mini_op)
                            if np.any(rewards):
                                print "DUMPING REWARD MINIBATCH!"
                                np.save("models/success-states-" + str(steps), states)
                                np.save("models/success-actions-" + str(steps), actions)
                                np.save("models/success-rewards-" + str(steps), rewards)
                                np.save("models/success-next_states-" + str(steps), next_states)
                                np.save("models/success-terminals-" + str(steps), terminals)
                                satisfied = True
                        #DEBUGGING###
                    #dump summaries if needed
                    if (steps > params.agent_params.log_start) and (steps % params.agent_params.log_freq == 0):
                        writer.add_summary(summary, steps)
                        avg_loss /= params.agent_params.log_freq
                        end_time = time.time()
                        sys.stdout.write("Size of history: " + str(len(agent.experience)) + "; steps: " + str(global_step.eval()) +
                                                        "; epsilon " + str(agent.env.ep) + "; Loss " + str(avg_loss) +
                                                        "; Batch training time " + str((end_time-start_time)/params.agent_params.log_freq) + "\n")
                        start_time = time.time()
                        avg_loss = 0
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
    except Exception as e:
        traceback.print_exc()
        sys.exit()
