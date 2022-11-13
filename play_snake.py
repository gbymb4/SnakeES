# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 19:26:06 2022

@author: Gavin
"""

import time, random

import numpy as np

import matplotlib.pyplot as plt

from env.snake.lib.game import (
    Game, create_listener, ACTION,
    contextualise_pointing_direction,
    InvalidPointingContextException,
    OutsideRegionException,
    AteSelfException,
    TICK_DELAY,
    MoveDirection
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)



def display(game):...


def lose(reason):
    print(f'lost due to {reason}!')
    
    

def main():
    set_seed(0)
    
    snake_game = Game()
    _ = create_listener()
    
    display(snake_game)
    time.sleep(TICK_DELAY)
    
    while True:
        try:
            snake = snake_game.snake
            
            move_dir = snake.move_direction
            
            #move_dir = MoveDirection(np.random.choice(3) - 1) 
            
            if ACTION is not None:
                try: move_dir = contextualise_pointing_direction(ACTION, snake)
                except InvalidPointingContextException:...
            
            snake_game.step(move_dir)    
            
            display(snake_game)
            
            time.sleep(TICK_DELAY)
            
        except OutsideRegionException as e: 
            lose(e)
            break
        except AteSelfException as e:
            lose(e)
            break
    
if __name__ == '__main__':
    main()