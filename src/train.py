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
import dqn


#create session and agent
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)      #can set how much memory to allocate on GPU
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
agent = dqn.dqn(sess, domains.fire_fighter)

#initialize everything
sess.run(tf.initialize_all_variables())

if params.summary > 0:
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("./logs", sess.graph_def)

saver = tf.train.Saver(max_to_keep=0)

steps = 0
successes = 0
failures = 0
successes_sampled = 0
failures_sampled = 0
try:
    with sess.as_default():
        #run random steps in the beginning to fill up experience
        print "STARTING RANDOM INITIALIZATIONS"
        while(steps < params.learn_start):
            r, t = agent.perceive()
            if r == 1:
                successes += 1
            elif r == -1:
                failures += 1
            steps += 1
            if steps % 10000==0:
                print "Size of history: " + str(len(agent.experience))
        #start training!
        print "DONE RANDOM PLAY"
        start_time = time.time()
        avg_loss = 0
        steps = 0
        while(steps < params.steps):
            steps += 1
            ######################################## draw a minibatch #########################################
            minibatch = random.sample(agent.experience, agent.batch_size)
            states = np.array([minibatch[i][0] for i in range(agent.batch_size)]).astype(np.float32)
            actions = np.array([minibatch[i][1] for i in range(agent.batch_size)]).astype(np.float32)
            rewards = np.array([minibatch[i][2] for i in range(agent.batch_size)]).astype(np.float32)
            successes_sampled += np.sum(rewards==1)
            failures_sampled += np.sum(rewards==-1)
            next_states = np.array([minibatch[i][3] for i in range(agent.batch_size)]).astype(np.float32)
            terminals = np.array([minibatch[i][4] for i in range(agent.batch_size)]).astype(np.float32)

            #calculate the bellman target = r + gamma*(1-terminal)*max(Q_target)
            Q_target = agent.target_net.logits.eval(feed_dict={agent.target_net.state_placeholder:next_states})
            Q_target_max = np.amax(Q_target, axis=1)
            Q_target_terminal = (1-terminals)*Q_target_max
            Q_target_gamma = agent.gamma*Q_target_terminal
            target = rewards + Q_target_gamma

            #################################### run the training step ########################################
            if params.summary > 0 and steps % params.log_freq == 0:
                (summary, result, loss) = sess.run([merged, agent.train_op, agent.loss],
                                                    feed_dict={agent.target_placeholder:target,
                                                                agent.train_net.state_placeholder:states,
                                                                agent.actions_placeholder: actions})
            else:
                (result, loss) = sess.run([agent.train_op, agent.loss],
                                            feed_dict={agent.target_placeholder:target,
                                            agent.train_net.state_placeholder:states,
                                            agent.actions_placeholder: actions})

            #perceive the next state
            r, t = agent.perceive()

            ############################## copy over target network if needed #################################
            if steps % params.target_q == 0:
                print "COPYING TARGET NETWORK OVER AT " + str(steps)
                agent.target_net.copy_weights(agent.train_net.var_dir, sess)

            ############################### all book keeping now ##############################################
            #for printing
            avg_loss += loss
            if r == 1:
                successes += 1
            elif r == -1:
                failures += 1
            #save model
            if (steps % params.save_freq == 0):
                print "SAVING MODEL AFTER " + str(steps) + " ..."
                saver.save(sess, "./models-grad/model", global_step = steps)
            #log summaries if needed
            if (steps > params.log_start) and (steps % params.log_freq == 0):
                if params.summary > 0:
                    writer.add_summary(summary, steps)
                #print
                avg_loss /= params.log_freq
                end_time = time.time()
                print "Size of history: ", len(agent.experience), "; Training steps: ", steps,\
                                                "; epsilon ", agent.env.ep, "; Successes perceived: ", successes,\
                                                "; Failures perceived ", failures, "; Successes sampled ", successes_sampled,\
                                                "; Failures sampled ", failures_sampled,\
                                                "; Average batch loss ", avg_loss,\
                                                "; Batch training time ", (end_time-start_time)/params.log_freq
                start_time = time.time()
                avg_loss = 0

except Exception as e:
    traceback.print_exc()
    sys.exit()
