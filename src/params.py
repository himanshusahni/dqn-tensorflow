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
    len_buffer    = 100
    batch_size    = 32

class agent_params(object):
    """paramaters for agent behavior"""
    replay_memory        = 1e6
    min_replay           = 10
    num_gameplay_threads = 4
    steps                = 1000
    history              = 3

class game_params:
    img_size = [32,32]
