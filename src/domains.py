"""
Test domains for deep transfer
"""
import numpy as np
import random
from collections import deque
import params


class fire_fighter(object):
    """
    Taxi cab style domain
    """
    actions = ['Left', 'Right', 'Up', 'Down', 'Pick', 'Drop']

    def __init__(self):
        self.screen_size = [g*params.grid_to_pixel for g in params.grid_size]
        self.grid_size = params.grid_size
        self.grid_to_pixel = params.grid_to_pixel
        self.agent_color = params.agent_color
        self.water_color = params.water_color
        self.fire_color = params.fire_color
        self.agent_water_color = params.agent_water_color
        if params.num_fires > params.num_waters:
            raise(ValueError("More fires than waters, please edit params"))
        #initialize domain
        self.reset()

    def get_actions(self):
        return self.actions

    def reset(self):
        """initialize locations of agent, fire and water"""
        #TODO: possible to make this more efficient (in larger domains it's wasteful computation)
        possible_coordinates = [(x, y) for x in range(self.grid_size[0]) for y in range(self.grid_size[1])]
        coord_pool = random.sample(possible_coordinates, params.num_fires + params.num_waters + 1)
        self.agent = coord_pool[0]
        self.fire = [(3,3),(5,3)]
        # self.fire = coord_pool[1:params.num_fires + 1]
        self.water = coord_pool[params.num_fires + 1:]
        self.has_water = False
        self.picked_at = None

    def grab_screen(self):
        """current screen of the game
        returns: numpy nd-array of shape screen_size"""
        #TODO: possible to make this more efficient by not generating a new array every time grab_screen is called.
        self.screen = np.zeros(self.screen_size + [3])

        #temp array to test display
        self.disp_arr = np.zeros(self.grid_size)
        agent = 1
        water = 2
        fire = 3
        if self.has_water:
            agent = 12
        elif self.agent in self.water:
            agent = 21

        #set color of water location
        _color = self.agent_water_color if self.has_water else self.water_color
        for _ in range(3):
            for w in self.water:
                if self.has_water and w == agent:
                    water = 12
                self.screen[w[0]*self.grid_to_pixel:(w[0]+1)*self.grid_to_pixel,
                        w[1]*self.grid_to_pixel:(w[1]+1)*self.grid_to_pixel, _] = _color[_]
                self.disp_arr[w[0]][w[1]] = water
                water = 2

        #set color of agent location
        _color = self.agent_water_color if self.has_water else self.agent_color
        for _ in range(3):
            self.screen[self.agent[0]*self.grid_to_pixel:(self.agent[0]+1)*self.grid_to_pixel,
                        self.agent[1]*self.grid_to_pixel:(self.agent[1]+1)*self.grid_to_pixel, _] = _color[_]
            self.disp_arr[self.agent[0]][self.agent[1]] = agent

        #set color of fire
        for _ in range(3):
            for f in self.fire:
                self.screen[f[0]*self.grid_to_pixel:(f[0]+1)*self.grid_to_pixel,
                            f[1]*self.grid_to_pixel:(f[1]+1)*self.grid_to_pixel, _] = self.fire_color[_]
                self.disp_arr[f[0]][f[1]] = fire
        return self.screen

    def get_dims(self):
        """row and column of screen in pixels"""
        return self.screen_size

    def execute_action(self, a):
        """takes action a in the game and gives new screen, reward and terminal
        if the new state is terminal, then start a new game.
        a: index of action to be executed
        returns: numpy.ndarray, shape=self.screen_size: new game screen (from grab_screen)
                 float32: reward for executing given action
                 boolean: whether the new state is a terminal state or not
        """
        reward = 0
        if a == 4 or a == 5: #same square
            if self.agent in self.water: #on water or has water
                if self.has_water: #can drop
                    if a == 5:
                        # print('Dropping Water')
                        self.has_water = False
                        adj_to_fire = None
                        for coord in self.fire:
                            if self.isAdjacent(coord, self.agent):
                                adj_to_fire = coord
                        if adj_to_fire: #agent is adjacent to fire
                            # print('Dousing. Win.')
                            print adj_to_fire
                            self.water.remove(self.agent)
                            self.agent = adj_to_fire
                            self.fire.remove(adj_to_fire)
                            reward = 1 #positive reward for each fire doused
                else: #can pick
                    if a == 4:
                        # print('Picking Water Up')
                        self.has_water = True #reward for picking up?
        else:
            if self.has_water:
                self.water.remove(self.agent)

            if a == 0:
                self.agent = (self.agent[0], self.agent[1] - 1) if self.agent[1] > 0 else self.agent
            elif a == 1:
                self.agent = (self.agent[0], self.agent[1] + 1) if self.agent[1] < self.grid_size[1] - 1 else self.agent
            elif a == 2:
                self.agent = (self.agent[0] - 1, self.agent[1]) if self.agent[0] > 0 else self.agent
            elif a == 3:
                self.agent = (self.agent[0] + 1, self.agent[1]) if self.agent[0] < self.grid_size[0] - 1 else self.agent


            if self.has_water:
                self.water.append(self.agent)

            if self.agent in self.fire: #die
                reward = -1 #death, negative reward

        return (self.grab_screen(), reward, self.isTerminal())

    def isTerminal(self):
        """
        Decides if terminal condition has been reached
        :return: True if water douses fire or agent walks into fire
        """
        return set(self.fire).issubset(set(self.water)) or (self.agent in self.fire)

    def isAdjacent(self, coord, check_coord):
        adjCoords = [(coord[0] - 1, coord[1]), (coord[0] + 1, coord[1]), (coord[0], coord[1] - 1), (coord[0], coord[1] + 1)]
        adjCoords = [some_coord for some_coord in adjCoords if self.isLegal(coord)]
        return check_coord in adjCoords

    def isLegal(self, coord):
        x = coord[0]
        y = coord[1]
        return 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]
