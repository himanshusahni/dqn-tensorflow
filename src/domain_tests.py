import unittest
import numpy as np

import domains
import params

class test_fire_fighter(unittest.TestCase):

    def setUp(self):
        self.ff = domains.fire_fighter(params.game_params)

    def test_init(self):
        #spawn state not same
        self.assertNotEqual(self.ff.agent, self.ff.fire)
        self.assertNotEqual(self.ff.agent, self.ff.water)
        self.assertNotEqual(self.ff.water, self.ff.fire)
        cur_screen = self.ff.grab_screen()
        #init colors set properly
        self.assertTrue(cur_screen[self.ff.agent[0]][self.ff.agent[1]] == params.game_params.agent_color)
        self.assertTrue(cur_screen[self.ff.water[0]][self.ff.water[1]] == params.game_params.water_color)
        self.assertTrue(cur_screen[self.ff.fire[0]][self.ff.fire[1]] == params.game_params.fire_color)


    def test_motion(self):
        #set the agent, water and fire at suitable locations
        self.ff.agent = (0,0)
        self.ff.fire = (2,2)
        self.ff.water = (4,4)
        cur_screen = self.ff.grab_screen()
        #another color check
        self.assertTrue(cur_screen[self.ff.agent[0]][self.ff.agent[1]] == params.game_params.agent_color)
        self.assertTrue(cur_screen[self.ff.water[0]][self.ff.water[1]] == params.game_params.water_color)
        self.assertTrue(cur_screen[self.ff.fire[0]][self.ff.fire[1]] == params.game_params.fire_color)
        #move to right end of board
        self.ff.execute_action(0) #shouldn't move
        self.assertEqual(self.ff.agent,(0,0))
        self.ff.execute_action(2) #shouldn't move
        self.assertEqual(self.ff.agent,(0,0))
        for i in range(1,5):
            self.ff.execute_action(1) #move right
            self.assertEqual(self.ff.agent,(0,i))
        self.ff.execute_action(1) #shouldn't move
        self.assertEqual(self.ff.agent,(0,4))
        cur_screen = self.ff.grab_screen()
        #another color check
        self.assertTrue(cur_screen[self.ff.agent[0]][self.ff.agent[1]] == params.game_params.agent_color)
        self.assertTrue(cur_screen[self.ff.water[0]][self.ff.water[1]] == params.game_params.water_color)
        self.assertTrue(cur_screen[self.ff.fire[0]][self.ff.fire[1]] == params.game_params.fire_color)
        #move to bottom of board
        for i in range(1,5):
            self.ff.execute_action(3) #move down
            self.assertEqual(self.ff.agent,(i,4))
        self.ff.execute_action(3) #shouldn't move
        self.assertEqual(self.ff.agent,(4,4))

    def test_water_pick_drop(self):
        #set the agent, water and fire at suitable locations
        self.ff.agent = (0,0)
        self.ff.fire = (2,2)
        self.ff.water = (0,0)
        self.assertFalse(self.ff.has_water)
        self.ff.execute_action(4)
        self.assertTrue(self.ff.has_water)
        cur_screen = self.ff.grab_screen()
        #water color check
        self.assertTrue(cur_screen[self.ff.agent[0]][self.ff.agent[1]] == params.game_params.agent_water_color)
        #drop water back to same location
        self.ff.execute_action(5)
        self.assertFalse(self.ff.has_water)
        cur_screen = self.ff.grab_screen()
        #water color check
        self.assertTrue(cur_screen[self.ff.agent[0]][self.ff.agent[1]] == params.game_params.agent_color or
                        cur_screen[self.ff.agent[0]][self.ff.agent[1]] == params.game_params.water_color)
        #pick up water again
        self.ff.execute_action(4)
        self.assertTrue(self.ff.has_water)
        #move to next spot
        self.ff.execute_action(1)
        #check if still has water
        self.assertTrue(self.ff.has_water)
        #move next to fire
        self.ff.execute_action(1)
        self.ff.execute_action(3)
        self.assertEqual(self.ff.agent,(1,2))
        #try to douse
        self.ff.execute_action(5)
        self.assertEqual(self.ff.agent,self.ff.fire)
        self.assertEqual(self.ff.water,self.ff.fire)

    def test_rewards(self):
        #set the agent, water and fire at suitable locations
        self.ff.agent = (0,0)
        self.ff.fire = (2,2)
        self.ff.water = (0,0)
        self.ff.has_water = False
        #pick up water (no reward)
        (screen, reward, terminal) = self.ff.execute_action(4)
        self.assertEqual(reward, 0)
        #drop water (no reward)
        (screen, reward, terminal) = self.ff.execute_action(5)
        self.assertEqual(reward, 0)
        #move around the map (no reward)
        (screen, reward, terminal) = self.ff.execute_action(1)
        self.assertEqual(reward, 0)
        (screen, reward, terminal) = self.ff.execute_action(3)
        self.assertEqual(reward, 0)
        (screen, reward, terminal) = self.ff.execute_action(0)
        self.assertEqual(reward, 0)
        (screen, reward, terminal) = self.ff.execute_action(2)
        self.assertEqual(reward, 0) #back to original location
        #pick up water again
        (screen, reward, terminal) = self.ff.execute_action(4)
        #move around the map again (no reward)
        (screen, reward, terminal) = self.ff.execute_action(1)
        self.assertEqual(reward, 0)
        (screen, reward, terminal) = self.ff.execute_action(3)
        self.assertEqual(reward, 0)
        (screen, reward, terminal) = self.ff.execute_action(0)
        self.assertEqual(reward, 0)
        (screen, reward, terminal) = self.ff.execute_action(2)
        self.assertEqual(reward, 0) #back to original location
        #move next to fire
        (screen, reward, terminal) = self.ff.execute_action(1)
        (screen, reward, terminal) = self.ff.execute_action(1)
        (screen, reward, terminal) = self.ff.execute_action(3)
        #drop water
        (screen, reward, terminal) = self.ff.execute_action(5)
        self.assertEqual(reward, 1)
        #reset to a different location and test again
        self.ff.agent = (2,1)
        self.ff.water = (2,1)
        self.ff.has_water = True
        #drop water
        (screen, reward, terminal) = self.ff.execute_action(5)
        self.assertEqual(reward, 1)
        #reset to a different location and test again
        self.ff.agent = (2,4)
        self.ff.water = (2,4)
        self.ff.has_water = True
        #drop water
        (screen, reward, terminal) = self.ff.execute_action(5)
        self.assertEqual(reward, 0)


        
    def test_terminal(self):
        self.ff.agent = (0,0)
        self.ff.fire = (2,2)
        self.ff.water = (0,0)
        self.ff.has_water = False
        #pick up water (not terminal)
        (screen, reward, terminal) = self.ff.execute_action(4)
        self.assertFalse(terminal)
        #drop water (not terminal)
        (screen, reward, terminal) = self.ff.execute_action(5)
        self.assertFalse(terminal)
        #move around the map (not terminal)
        (screen, reward, terminal) = self.ff.execute_action(1)
        self.assertFalse(terminal)
        (screen, reward, terminal) = self.ff.execute_action(3)
        self.assertFalse(terminal)
        (screen, reward, terminal) = self.ff.execute_action(0)
        self.assertFalse(terminal)
        (screen, reward, terminal) = self.ff.execute_action(2)
        self.assertFalse(terminal) #back to original location
        #pick up water again
        (screen, reward, terminal) = self.ff.execute_action(4)
        #move around the map again (not terminal)
        (screen, reward, terminal) = self.ff.execute_action(1)
        self.assertFalse(terminal)
        (screen, reward, terminal) = self.ff.execute_action(3)
        self.assertFalse(terminal)
        (screen, reward, terminal) = self.ff.execute_action(0)
        self.assertFalse(terminal)
        (screen, reward, terminal) = self.ff.execute_action(2)
        self.assertFalse(terminal) #back to original location
        #move next to fire
        (screen, reward, terminal) = self.ff.execute_action(1)
        (screen, reward, terminal) = self.ff.execute_action(1)
        (screen, reward, terminal) = self.ff.execute_action(3)
        #drop water
        (screen, reward, terminal) = self.ff.execute_action(5)
        self.assertTrue(terminal)
        #reset to a different location and test again
        self.ff.agent = (2,3)
        self.ff.water = (2,3)
        self.ff.has_water = True
        #drop water
        (screen, reward, terminal) = self.ff.execute_action(5)
        self.assertTrue(terminal)
        #reset to a different location and test again
        self.ff.agent = (2,4)
        self.ff.water = (2,4)
        self.ff.has_water = True
        #drop water
        (screen, reward, terminal) = self.ff.execute_action(5)
        self.assertFalse(terminal)
if __name__ == '__main__':
    unittest.main()
