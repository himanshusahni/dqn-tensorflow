"""
contains all training parameters for the full dqn+rnn for transfer
may also contain parameters for the domain
"""

"""parameters for the convnet"""
n_units       = [32, 64, 64]      #convolution filters at each layer
filter_size   = [8, 4, 4]         #size at each layer
filter_stride = [4, 2, 1]         #stride at each layer
n_hid         = [512]             #size of fully connected layers
batch_size    = 64                #states in minibatch
lr            = 1e-4             #learning rate
lr_step       = 2e4               #step size of lr annealing
clip_delta    = 1                 #gradient clipping


"""paramaters for agent behavior"""
num_gameplay_threads = 3            #unused in serial
history              = 1            #frames stored in history buffer
gamma                = 0.99         #discount factor
target_q             = 1000         #frequency of copying training network
learn_start          = 10000         #steps of random play in beginning
replay_memory        = 50000        #maximum number of states in replay
min_replay           = 10000         #minimum number of states in replay
steps                = 5e5        #maximum training steps
ep                   = 1          #starting epsilon
ep_end               = 0.1          #final epsilon
ep_endt              = 1e6       #number of steps after which epsilon stops annealing
valid_ep             = 0.5         #epsilon for validation runs
valid_start          = 2e6         #steps after which validation starts
valid_episodes       = 200          #number of episodes validation run averaged over
save_freq            = 5000        #frequency of saving convnet
valid_freq           = 50000         #frequency of validations
log_freq             = 100          #frequency of logging loss and gradient histograms
log_start            = 100         #when to start logging loss

"""gridworld domain params"""
grid_size         = [8,8]
grid_to_pixel     = 10              #pixels per grid location
fire_color        = [1.0,0,0]       #rgb
agent_color       = [0,1.0,0]
water_color       = [0,0,1.0]
agent_water_color = [1.0,1.0,1.0]
