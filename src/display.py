import pygame
from params import game_params


class display(object):
    # Defining colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    AGENT = 255 * [0,1,0]
    WATER = 255 * [0,0,1]
    FIRE = 255 * [1,0,0]
    AGENT_W_WATER = 255 * [1, 1, 1]
    AGENT_ON_WATER = 255 * [0.5, 0.5, 0.5]

    MARGIN = 5 # margin between each cell

    def __init__(self, grid_size, grid_to_pixel):
        self.grid = []
        self.grid_size = grid_size
        for row in range(grid_size[0]):
            self.grid.append([])
            for column in range(grid_size[1]):
                self.grid[row].append(0)

        self.ratio = grid_to_pixel

        # Initialize pygame
        pygame.init()
        done = False

        # Set the HEIGHT and WIDTH of the screen
        WINDOW_SIZE = [255, 255]

        # WIDTH and HEIGHT of each grid location
        self.WIDTH = WINDOW_SIZE[0]/grid_size[0] - display.MARGIN - display.MARGIN/grid_size[0]
        self.HEIGHT = WINDOW_SIZE[1]/grid_size[1] - display.MARGIN - display.MARGIN/grid_size[1]

        self.screen = pygame.display.set_mode(WINDOW_SIZE)

        # Set title of screen
        pygame.display.set_caption("Fire-Fighter")

        # Loop until the user clicks the close button.
        self.done = False

        # Used to manage how fast the screen updates
        self.clock = pygame.time.Clock()


    def draw(self, screen_arr):
        # Set the screen background
        self.screen.fill(display.BLACK)

        print(screen_arr)

        # Draw the grid
        for row in range(self.grid_size[0]):
            for column in range(self.grid_size[1]):
                color = display.WHITE
                if self.grid[row][column] == 1:
                    color = display.GREEN
                pygame.draw.rect(self.screen,
                                 color,
                                 [(display.MARGIN + self.WIDTH) * column + display.MARGIN,
                                  (display.MARGIN + self.HEIGHT) * row + display.MARGIN,
                                  self.WIDTH,
                                  self.HEIGHT])

        # Limit to 60 frames per second
        self.clock.tick(60)

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

    def quit(self):
        pygame.quit()


# ref - http://programarcadegames.com/index.php?chapter=array_backed_grids
