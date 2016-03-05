"""
contains all training parameters for the full dqn+rnn for transfer
may also contain parameters for the domain
"""

class net_params(object):
    """parameters for the convnet"""
    n_units       = [32, 64, 64]      #convolution filters at each layer
    filter_size   = [8, 4, 4]         #size at each layer
    filter_stride = [4, 2, 1]         #stride at each layer
    n_hid         = [512]             #size of fully connected layers
    batch_size    = 10                #states in minibatch
    lr            = 0.0025            #learning rate

class agent_params(object):
    """paramaters for agent behavior"""
    replay_memory        = 1000
    min_replay           = 10
    num_gameplay_threads = 4
    steps                = 1000
    history              = 3
    gamma           = 0.99

class game_params:
    grid_size = [5,5]
    grid_to_pixel = 6
    img_size = [g*grid_to_pixel for g in grid_size] + [3]
    agent_color = [0,1,0]
    water_color = [0,0,1]
    fire_color = [1,0,0]
    agent_water_color = [1,1,1]
    num_fires = 2
    num_waters = 3
