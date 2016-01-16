"""
contains all training parameters for the full dqn+rnn for transfer
may also contain parameters for the domain
"""

class game(object):
    """parameters for game"""
    img_size = [30,30]  #height, width

class net(object):
    """parameters for the convnet"""
    n_units       = [32, 64, 64]      #convolution filters at each layer
    filter_size   = [8, 4, 4]         #size at each layer
    filter_stride = [4, 2, 1]         #stride at each layer
    n_hid         = [512]             #size of fully connected layers
    history       = 3                 #num channels in input
    len_buffer    = 100
    batch_size    = 32

    def __init__(self, game_env):
        """sets parameters local to a game"""
        self.output_dims = len(game_env.actions)
        self.img_height = game.img_size[0]
        self.img_width = game.img_size[1]


class agent(object):
    """paramaters for agent behavior"""
    replay_memory = 1e6
    min_replay = 1e4
    threads = 4
