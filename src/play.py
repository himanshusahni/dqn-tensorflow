import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys

import dqn
import domains

def unycbcr(state):
    new_state = np.zeros(state.shape+(3,))
    agent = np.around(0.587*255)
    water = np.around(0.114*255)
    fire = np.around(0.299*255)
    agent_water = np.around(1*255)
    # print np.where(state==fire)[0]
    new_state[state==fire,0] = 1
    new_state[state==agent,1] = 1
    new_state[state==water,2] = 1
    new_state[state==agent_water,0] = 1
    new_state[state==agent_water,1] = 1
    new_state[state==agent_water,2] = 1
    return new_state

if __name__ == "__main__":
    #create session and agent
    sess = tf.Session()
    global_step = tf.Variable(0, trainable=False)
    agent = dqn.dqn(sess, domains.fire_fighter, global_step)
    actions = agent.env.get_actions()
    #load previous agent
    saver = tf.train.Saver()
    # model = tf.train.get_checkpoint_state("./models-new/")
    # if model and model.model_checkpoint_path:
    #     print "Loading model " + model.model_checkpoint_path
    #     saver.restore(sess, model.model_checkpoint_path)

    model_num = 2120000
    print "Loading model model-" + str(model_num)
    saver.restore(sess, "models-taxi/model-" + str(model_num))
    #set up GUI
    plt.ion()
    f = plt.figure()
    a = f.add_subplot(131)
    plt.title("state-2")
    b = f.add_subplot(132)
    plt.title("state-1")
    c = f.add_subplot(133)
    plt.title("state")
    a_show = a.imshow(np.zeros(agent.img_size))
    b_show = b.imshow(np.zeros(agent.img_size))
    c_show = c.imshow(np.zeros(agent.img_size))
    plt.show()

    input_map = {'a': 0, 'd': 1, 'w': 2, 's': 3, 'z': 4, 'x': 5}
    inp = ""
    with sess.as_default():
        while (inp != "q"):
            print "Starting a validation run!"
            agent.state = agent.env.new_game()  #terminate current game and set up a new validation game
            terminal = False
            old_state = agent.state
            while not terminal:
                print "########################################################################################"
                #show current state
                a_show.set_data(unycbcr(old_state[:,:,0]))
                a_show.autoscale()
                b_show.set_data(unycbcr(agent.state[:,:,0]))
                b_show.autoscale()
                c_show.set_data(unycbcr(agent.state[:,:,0]))
                c_show.autoscale()
                #pick best action according to convnet on current state
                action_values = agent.train_net.logits.eval(feed_dict={agent.train_net.state_placeholder: np.expand_dims(agent.state, axis=0)})

                action_values = np.squeeze(action_values)
                max_a = np.argmax(action_values)
                for i,value in enumerate(action_values):
                    print(actions[i] + " value: " + str(value))
                print "Action preferred by net: " + actions[max_a]
                plt.draw()
                invalid = True
                while invalid:
                    invalid = False
                    inp = raw_input("Selection action to take or q to exit. To auto-pilot press y\n")
                    if inp == 'q':
                        break
                    elif inp == 'y':
                        chosen_a = max_a
                    elif (inp in input_map):
                        chosen_a = input_map[inp]
                    else:
                        print "Invalid selection, please select: w,s,a,d,z,x,q, or y only!"
                        invalid = True
                #execute that action in the environment,
                (next_state, reward, terminal) = agent.env.take_action(chosen_a, "False")

                print "REWARD IS " + str(reward)
                print "Terminal is " + str(terminal)
                # Q_target = agent.target_net.logits.eval(feed_dict={agent.target_net.state_placeholder:np.expand_dims(next_state, axis=0)})
                #
                # Q_target_max = np.amax(Q_target)
                # Q_target_terminal = (1-terminal)*Q_target_max
                # Q_target_gamma = agent.gamma*Q_target_terminal
                # target = reward + Q_target_gamma
                # print "Q_target is " + str(Q_target)
                # print "Q_target_max is " + str(Q_target_max)
                # print "Q_target_terminal is " + str(Q_target_terminal)
                # print "Q_target_gamma is " + str(Q_target_gamma)
                # print "target is " + str(target)
                # action_one_hot = np.zeros(agent.num_actions)
                # action_one_hot[chosen_a] = 1.0
                # (loss, Q_train, Q_train_actions, clipped_loss_vec) = sess.run([agent.loss, agent.Q_train, agent.Q_train_actions, agent.clipped_loss_vec],
                #                                                         feed_dict={agent.target_placeholder:np.expand_dims(target, axis=0),
                #                                                                             agent.train_net.state_placeholder:np.expand_dims(agent.state, axis=0),
                #                                                                             agent.train_actions_placeholder: np.expand_dims(action_one_hot, axis=0)})
                # print "Q_train is " + str(Q_train)
                # print "Q_train_actions is " + str(Q_train_actions)
                # print "clipped_loss_vec is " + str(clipped_loss_vec)
                # print "loss is " + str(loss)
                # print "just checking " + str(np.mean(clipped_loss_vec))

                # gradients={}
                # for var in agent.grads_and_vars:
                #     gradients[var] = var[0].eval(feed_dict={agent.target_placeholder:np.expand_dims(target, axis=0),
                #                         agent.train_net.state_placeholder:np.expand_dims(agent.state, axis=0),
                #                         agent.train_actions_placeholder: np.expand_dims(action_one_hot, axis=0)})
                #     if gradients[var].shape[0] == 4 and len(gradients[var].shape) == 1:
                #         print gradients[var]
                #         bias_old = agent.train_net.var_dir[var[1].name].eval()
                #         print "OLD BIAS TERM " + str(bias_old)
                #         break
                # print "APPLYING THE GRADIENT"
                # result = sess.run([agent.opt_op],feed_dict={agent.target_placeholder:np.expand_dims(target, axis=0),
                #                                                                             agent.train_net.state_placeholder:np.expand_dims(agent.state, axis=0),
                #                                                                             agent.train_actions_placeholder: np.expand_dims(action_one_hot, axis=0)})
                # Q_train_new = agent.Q_train.eval(feed_dict={agent.train_net.state_placeholder:np.expand_dims(agent.state, axis=0)})
                # print "NEW Q_train is " + str(Q_train_new)
                # print "Difference is " + str(Q_train_new - Q_train)
                # bias_new = agent.train_net.var_dir[var[1].name].eval()
                # print "NEW BIAS TERM " + str(bias_new)
                # print "Difference is " + str(bias_new - bias_old)
                # print "COPYING OVER TRAIN NET"
                # agent.target_net.copy_weights(agent.train_net.var_dir, sess)
                # print "NEW TRAIN VALUES ON THIS STATE ARE " + str(agent.Q_train.eval(feed_dict={agent.train_net.state_placeholder:np.expand_dims(agent.state, axis=0)}))
                # print "NEW TARGET VALUES ON NEXT STATE ARE " + str(agent.target_net.logits.eval(feed_dict={agent.target_net.state_placeholder:np.expand_dims(next_state, axis=0)}))
                # print "NEW TARGET VALUES ON OLD STATE ARE " + str(agent.target_net.logits.eval(feed_dict={agent.target_net.state_placeholder:np.expand_dims(agent.state, axis=0)}))
                old_state = agent.state
                agent.state = next_state
