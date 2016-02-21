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
    batch_size    = 32                #states in minibatch
    lr            = 0.0025            #learning rate
    clip_delta    = 1                 #gradient clipping
class agent_params(object):
    """paramaters for agent behavior"""
    num_gameplay_threads = 4
    history              = 3
    gamma                = 0.99
    target_q             = 10000
    learn_start          = 10000
    replay_memory        = 100000
    min_replay           = 1000
    steps                = 100000
    
class game_params:
    grid_size         = [5,5]
    grid_to_pixel     = 6
    fire_color        = [1.0,0,0]    #rgb
    agent_color       = [0,1.0,0]
    water_color       = [0,0,1.0]
    agent_water_color = [1.0,1.0,1.0]
    img_size =  [g*grid_to_pixel for g in grid_size]
