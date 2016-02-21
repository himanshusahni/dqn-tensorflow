"""
Test domains for deep transfer
"""
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

from params import game_params


class fire_fighter(object):
    """
    Taxi cab style domain
    """
    actions = ['Left', 'Right', 'Up', 'Down', 'Pick', 'Drop']

    def __init__(self, params):
        self.screen_size = params.img_size
        self.grid_size = params.grid_size
        self.grid_to_pixel = params.grid_to_pixel
        self.agent_color = params.agent_color
        self.water_color = params.water_color
        self.fire_color = params.fire_color
        self.agent_water_color = params.agent_water_color
        #initialize domain
        self.reset()
        self.counter = 0    #set step counter

    def reset(self):
        """initialize locations of agent, fire and water"""
        #TODO: possible to make this more efficient (in larger domains it's wasteful computation)
        possible_coordinates = [(x, y) for x in range(self.grid_size[0]) for y in range(self.grid_size[1])]
        coord_pool = random.sample(possible_coordinates, 3)
        self.agent = coord_pool[0]
        self.fire = coord_pool[1]
        self.water = coord_pool[2]
        self.has_water = False
        # print("Agent: " , self.agent, "Fire: ", self.fire , " Water: ", self.water, " Has Water: ", self.has_water)


    def grab_screen(self):
        """current screen of the game
        returns: numpy nd-array of shape screen_size"""
        #TODO: possible to make this more efficient by not generating a new array every time grab_screen is called.
        self.screen = self.counter*np.zeros(self.screen_size + [3])
        #set color of agent location
        _color = self.agent_water_color if self.has_water else self.agent_color
        for _ in range(3):
            self.screen[self.agent[0]*self.grid_to_pixel:(self.agent[0]+1)*self.grid_to_pixel,
                        self.agent[1]*self.grid_to_pixel:(self.agent[1]+1)*self.grid_to_pixel, _] =  _color[_]

        #set color of water location
        _color = self.agent_water_color if self.has_water else self.water_color
        for _ in range(3):
            self.screen[self.water[0]*self.grid_to_pixel:(self.water[0]+1)*self.grid_to_pixel,
                        self.water[1]*self.grid_to_pixel:(self.water[1]+1)*self.grid_to_pixel, _] = _color[_]

        #set color of fire
        for _ in range(3):
            self.screen[self.fire[0]*self.grid_to_pixel:(self.fire[0]+1)*self.grid_to_pixel,
                        self.fire[1]*self.grid_to_pixel:(self.fire[1]+1)*self.grid_to_pixel, _] = self.fire_color[_]
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
        self.counter += 1
        reward = 0
        if a == 4 or a == 5: #same square

            if self.agent == self.water: #on water or has water

                if self.has_water: #can drop
                    if a == 5:
                        # print('Dropping Water')
                        self.has_water = False
                        if self.isAdjacent(self.fire, self.agent): #agent is adjacent to fire
                            # print('Dousing. Win.')
                            self.agent = self.fire
                            self.water = self.fire
                            reward = 1
                else: #can pick
                    if a == 4:
                        # print('Picking Water Up')
                        self.has_water = True #positive reward
        else:
            if a == 0:
                self.agent = (self.agent[0], self.agent[1] - 1) if self.agent[1] > 0 else self.agent
            elif a == 1:
                self.agent = (self.agent[0], self.agent[1] + 1) if self.agent[1] < self.grid_size[1] - 1 else self.agent
            elif a == 2:
                self.agent = (self.agent[0] - 1, self.agent[1]) if self.agent[0] > 0 else self.agent
            elif a == 3:
                self.agent = (self.agent[0] + 1, self.agent[1]) if self.agent[0] < self.grid_size[0] - 1 else self.agent


            if self.has_water:
                self.water = self.agent

            if self.agent == self.fire: #die
                # print('Death.')
                reward = -1 #negative reward

        return (self.grab_screen(), reward, self.isTerminal())

    def isTerminal(self):
        """
        Decides if terminal condition has been reached
        :return: True if water douses fire or agent walks into fire
        """
        return ((self.fire == self.water) and not self.has_water) or (self.agent == self.fire)

    def isAdjacent(self, coord, check_coord):
        # print("Fire: ", coord)
        # print("Agent: ", check_coord)
        adjCoords = [(coord[0] - 1, coord[1]), (coord[0] + 1, coord[1]), (coord[0], coord[1] - 1), (coord[0], coord[1] + 1)]
        adjCoords = [some_coord for some_coord in adjCoords if self.isLegal(coord)]
        # print("Fire: ", coord)
        # print("Agent: ", check_coord)
        # print("Adj to fire: ", adjCoords)
        return check_coord in adjCoords

    def isLegal(self, coord):
        x = coord[0]
        y = coord[1]
        return 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]


def main():
     """
     Test Method - wasd to move. z to pickup, x to drop, l to end.
     """
     fighter = fire_fighter(game_params)
     print("New Instance: Agent: " , fighter.agent, "Fire: ", fighter.fire , " Water: ", fighter.water)
     inp = None
     while (inp != "l" and not fighter.isTerminal()):
         inp = raw_input("Input: ")
         mapping = {"a": 0, "d": 1, "w": 2, "s": 3, "z": 4, "x": 5}
         if inp in mapping:
             fighter.execute_action(mapping[inp])
         else:
             print("Illegal")
         # print("Agent: " , fighter.agent, "Fire: ", fighter.fire , " Water: ", fighter.water, " Has Water: ", fighter.has_water)

if __name__ == "__main__":
     main()
