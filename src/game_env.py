from collections import deque
import numpy as np
import params

class Environment(object):
    """Shell for simplyfying certain tasks for agent"""
    def __init__(self, game, thread_num):
        #game parameters
        self.thread_num = thread_num
        self.game = game(params.game_params)
        self.img_size = params.game_params.img_size
        self.history = params.agent_params.history
        self.screen_history = deque(maxlen=self.history)
        self.counter = 0    #keeps track of number of steps taken in domain
        self.episodes = 0    #keeps track of number of episodes elapsed
        self.ep_reward = 0     #keeps track of epsodic reward
        self.ep = params.agent_params.ep                           #starting exploration randomness
        self.new_game()

    def flush_history(self):
        """fill history buffer with zeros"""
        for _ in range(self.screen_history.maxlen):
            self.screen_history.append(np.expand_dims(np.zeros(self.img_size), -1))

    def get_actions(self):
        return self.game.actions

    def get_num_actions(self):
        return len(self.game.actions)

    def get_img_size(self):
        return self.img_size

    def new_game(self):
        """reset the domain to start new episode and prepare the history"""
        self.game.reset()
        #reset episodic reward
        self.ep_reward = 0
        self.episodes += 1
        #flush history
        self.flush_history()
        #grab new screen and add to history
        screen = self.game.grab_screen()
        #preprocess screen
        screen = self.preprocess(screen)
        self.screen_history.append(screen)
        return self.get_state()

    def get_state(self):
        """current state of the agent in the game (concatenation of the last self.history frames)"""
        return np.concatenate([self.screen_history[hist] for hist in range(0, self.history)], axis=2)

    def take_action(self, a, valid):
        """take the action in the game, update history and return new state, reward and terminal"""
        #get new game screen
        (screen, reward, terminal) = self.game.execute_action(a)
        #preprocess screen
        screen = self.preprocess(screen)
        self.screen_history.append(screen)
        if not valid:
            self.counter += 1
            self.ep_reward += reward
        return (self.get_state(), reward, terminal)

    def preprocess(self, screen):
        """turn into grayscale and expand dimensions to get history"""
        screen = self.ycbcr(screen)
        screen = screen[:,:,0]  #taking only y channel for now
        screen = np.expand_dims(screen, -1)
        return screen

    def ycbcr(self, rgb_array):
        """convert rgb array to ycbcr (array values in 0-1)"""
        ycbcr = np.empty_like(rgb_array)
        ycbcr[:,:,0] = .299*rgb_array[:,:,0] + .587*rgb_array[:,:,1] + .114*rgb_array[:,:,2] #y
        ycbcr[:,:,1] = 0.5 -.168736*rgb_array[:,:,0] -.331364*rgb_array[:,:,1] + .5*rgb_array[:,:,2]
        ycbcr[:,:,2] = 0.5 +.5*rgb_array[:,:,0] - .418688*rgb_array[:,:,1] - .081312*rgb_array[:,:,2]
        return ycbcr

if __name__ == "__main__":
    g = Environment(domains.fire_fighter(params.game_params), params.agent_params)
    g.new_game()
    print g.take_action(5)
    print g.take_action(5)
    print g.take_action(5)
    print g.new_game()
    print g.take_action(5)
    print g.take_action(5)
    print g.take_action(5)
