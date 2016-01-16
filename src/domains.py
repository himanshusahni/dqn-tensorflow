"""
Test domains for deep transfer
"""

class fire_fighter(object):
    """
    Taxi cab style domain
    """
    actions = ['Left', 'Right', 'Up', 'Down']
    def __init__(self, params):
        self.screen_size = params.img_size
