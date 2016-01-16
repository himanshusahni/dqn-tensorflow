import tensorflow as tf
import numpy as np

import params
import domains
import convnet

class dqn(object):
    """deep q learning agent for arbitrary domains."""
    def __init__(self, params, game_env):
        super(dqn, self).__init__()
        self.available_threads = params.agent.threads
        self.games = [game_env(params.game) for i in range(self.available_threads - 1)]
        #keep track of history
        self.len_history = net_params.history   #TODO: look for a better way to buffer history
        self.len_buffer = net_params.len_buffer
        #create the history buffer
        self.history = [[tf.zeros([self.img_height,self.img_width, 1]) for i in range(self.len_buffer)]
                            for j in range(self.available_threads - 1)]  #TODO:better initialization
        self.buffer_pos = [-1 for i in range(self.available_threads - 1)]
        self.steps = 0
        #create experience buffer
        self.replay_memory = params.agent.replay_memory     #max length of memory queue
        self.min_replay = params.agent.min_replay       #min length of memory queue
        self.experience = tf.RandomShuffleQueue(self.replay_memory,
                                    self.min_replay, tf.float32,
                                    shapes = [self.img_height,self.img_width, self.len_history, 1, 1],  #image(2), history, reward, terminal_flag
                                    name = 'experience replay')
        #spawn threads to start playing the game and collecting experience
        self.coord = tf.train.Coordinator()
        self.experience_runner = tf.train.QueueRunner(self.experience,
                                    [self.perceive()]*(self.available_threads - 1))

        #create the convnet
        net_params = params.net(game_env)
        self.net = convnet.ConvNetGenerator(net_params)
        self.batch_size = net_params.batch_size
        self.img_height = net_params.img_height
        self.img_width = net_params.img_width



    def perceive(self):
        """
        handles insertion of state in recall memory.
        """
        #TODO:figure out how to get thread number
        #get game state
        rawstate = self.games[thread_num].grab_screen()
        state = self.get_complete_state(rawstate, thread_num)
        reward = self.games[thread_num].grab_reward()
        is_terminal = self.games[thread_num].is_terminal()
        aug_state = tf.expand_dims(tf.expand_dims(state, -1), -1)   #expand to store reward and terminal
        aug_state[:,:,:,0,:] = reward   #TODO:yeah this might not work
        aug_state[:,:,:,:,0] = terminal
        #insert current (state,reward,terminal) into experience memory
        enq = self.experience.enqueue(aug_state)
        #pick action according to current network
        action = self.get_action(state)
        #take appropriate action to get next state
        self.take_game_action(action, game)
        self.steps += 1
        return enq



    def take_game_action(self, action, game):
        """tells the game environment to execute an action. does not return anything"""

    def get_action(self, state):
        """returns action recommended by target network"""


    def train(self):
        """draws minibatch from experience queue and updates current net"""

    def get_complete_state(self, rawstate, thread_num):
        """
        adds new screen grab to history buffer and returns complete state.
        In the very beginning just appends tensors of all zeros to make up the
        self.len_history channel tensors.
        """
        #increment buffer position
        self.buffer_pos[thread_num] += 1
        self.buffer_pos[thread_num] %= self.len_buffer
        #add screen to buffer
        self.history[thread_num][self.buffer_pos[thread_num]] = tf.expand_dims(rawstate, -1)
        #grab last 'len_history' screens into a state
        state = tf.identity(self.history[thread_num][self.buffer_pos[thread_num]])  #TODO:can defiitely do better than this concaatination at each step
        back_counter = self.buffer_pos[thread_num]
        for i in range(self.len_history - 1):
            if back_counter == 0:
                back_counter = self.len_buffer
            back_counter -= 1
            state = tf.concat(2, [self.history[back_counter], state])       #concatenate in the channel dimension
        return state



game = domains.fire_fighter(params.game)
agent = dqn(params, game)
sess = tf.Session()
sess.run(tf.initialize_all_variables())

#start the experience collection!
enqueue_threads = agent.experience_runner.create_threads(sess, coord = agent.coord, start = True)
for i in range(64):
    print "iter " + str(i)
    if agent.coord.should_stop():
        break
    print sess.run(agent.train())
    print state.get_shape()
