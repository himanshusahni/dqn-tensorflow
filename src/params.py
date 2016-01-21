"""
contains all training parameters for the full dqn+rnn for transfer
may also contain parameters for the domain
"""

class net(object):
    """parameters for the convnet"""
    n_units       = [32, 64, 64]      #convolution filters at each layer
    filter_size   = [8, 4, 4]         #size at each layer
    filter_stride = [4, 2, 1]         #stride at each layer
    n_hid         = [512]             #size of fully connected layers
    history       = 3                 #num channels in input
    len_buffer    = 100
    batch_size    = 32
    img_height = 3
    img_width = 3
    img_size = [img_height, img_width]

    # def __init__(self, game_env):
    #     """sets parameters local to a game"""
    #     self.output_dims = len(game_env.actions)


class agent(object):
    """paramaters for agent behavior"""
    replay_memory = 1e6
    min_replay = 10
    available_threads = 4
    steps = 1000
