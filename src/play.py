import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys

import dqn
import domains

if __name__ == "__main__":
    #create session and agent
    sess = tf.Session()
    agent = dqn.dqn(sess, domains.fire_fighter)
    actions = agent.env.get_actions()
    #load previous agent
    saver = tf.train.Saver()
    model = tf.train.get_checkpoint_state("./models/")
    if model and model.model_checkpoint_path:
        print "Loading model " + model.model_checkpoint_path
        saver.restore(sess, model.model_checkpoint_path)

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
            agent.requestNewGame()  #terminate current game and set up a new validation game
            terminal = False
            while not terminal:
                #show current state
                a_show.set_data(agent.state[:,:,0])
                a_show.autoscale()
                b_show.set_data(agent.state[:,:,1])
                b_show.autoscale()
                c_show.set_data(agent.state[:,:,2])
                c_show.autoscale()
                #pick best action according to convnet on current state
                action_values = agent.train_net.logits.eval(feed_dict={agent.batch_state_placeholder: np.expand_dims(agent.state, axis=0)})
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
                agent.state = next_state
