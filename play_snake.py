# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 19:26:06 2022

@author: Gavin
"""

import time, random

import numpy as np

import matplotlib.pyplot as plt

from env.snake.lib.game import (
    Game, KeyListener,
    contextualise_pointing_direction,
    InvalidPointingContextException,
    OutsideRegionException,
    AteSelfException,
    TICK_DELAY,
    MoveDirection
)

from env.snake.lib.renderer import (
    SnakeWindow, 
    initialise_renderer, 
    update_visualisation
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)



def lose(reason, renderer, listener):
    print(f'lost due to {reason}!')
    
    renderer.stop()
    
    listener.stop()
    
    

def main():
    set_seed(0)
    
    snake_game = Game()
    listener = KeyListener()
    
    try:
        renderer, visualisation = initialise_renderer(snake_game)
        
        time.sleep(TICK_DELAY)
        
        while True:
            try:
                snake = snake_game.snake
                
                move_dir = snake.move_direction
                print(f'Reading Action: {listener.action}')
                if listener.action is not None:
                    try: 
                        move_dir = contextualise_pointing_direction(
                            listener.consume_action(), 
                            snake
                        )
                    except InvalidPointingContextException:...
                
                snake_game.step(move_dir)    
                
                update_visualisation(renderer, visualisation, snake_game)
                
                time.sleep(TICK_DELAY)
                
            except OutsideRegionException as e: 
                lose(e, renderer, listener)
                break
            except AteSelfException as e:
                lose(e, renderer, listener)
                break
        
    except:
        renderer.stop()
        listener.stop()
    
if __name__ == '__main__':
    main()