import gymnasium as gym
import flappy_bird_gymnasium
import pygame
# creating the env
env=gym.make("FlappyBird-v0",render_mode="human")
state,info=env.reset()
done=False
# initialize pygame keyboard
pygame.init()
screen=pygame.display.get_surface() # gym has alredy created the window
while not done:
    action=0 # default -> 0 no flap and 1 is flap
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            done=True
        elif event.type==pygame.KEYDOWN:
            if event.key==pygame.K_SPACE:
                action=1 # flap
    state,reward,done,truncated,info=env.step(action)
    env.render()
env.close()
pygame.quit()
