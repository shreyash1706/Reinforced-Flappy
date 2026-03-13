import numpy as np
import pygame, random 
from pygame.locals import *

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
SPEED = 20
GRAVITY = 2.5
GAME_SPEED = 15
GROUND_WIDTH = 2 * SCREEN_WIDTH
GROUND_HEIGHT = 100
PIPE_WIDTH = 80
PIPE_HEIGHT = 500
PIPE_GAP = 150

# Global Assets for the Test/Display
# We load these once here so they are available for rendering
pygame.init()
BACKGROUND = pygame.image.load('assets/sprites/background-day.png')
BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDTH, SCREEN_HEIGHT))

class Bird(pygame.sprite.Sprite):

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)

        self.images =  [pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha(),
                        pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha(),
                        pygame.image.load('assets/sprites/bluebird-downflap.png').convert_alpha()]

        self.speed = SPEED

        self.current_image = 0
        self.image = pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
        self.rect[0] = SCREEN_WIDTH / 6
        self.rect[1] = SCREEN_HEIGHT / 2

    def update(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        self.speed += GRAVITY

        #UPDATE HEIGHT
        self.rect[1] += self.speed

    def bump(self):
        self.speed = -SPEED

    def begin(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]




class Pipe(pygame.sprite.Sprite):

    def __init__(self, inverted, xpos, ysize):
        pygame.sprite.Sprite.__init__(self)

        self. image = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDTH, PIPE_HEIGHT))


        self.rect = self.image.get_rect()
        self.rect[0] = xpos

        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect[1] = - (self.rect[3] - ysize)
        else:
            self.rect[1] = SCREEN_HEIGHT - ysize


        self.mask = pygame.mask.from_surface(self.image)


    def update(self):
        self.rect[0] -= GAME_SPEED

        

class Ground(pygame.sprite.Sprite):
    
    def __init__(self, xpos):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/base.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDTH, GROUND_HEIGHT))

        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT
    def update(self):
        self.rect[0] -= GAME_SPEED

def is_off_screen(sprite):
    return sprite.rect[0] < -(sprite.rect[2])

def get_random_pipes(xpos):
    size = random.randint(100, 300)
    pipe = Pipe(False, xpos, size)
    pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)
    return pipe, pipe_inverted

class FlappyEnv:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.max_frames = 10000
        self.frames_survived = 0

    def reset(self):
        
        self.frames_survived = 0
        self.bird_group = pygame.sprite.Group()
        self.bird = Bird()
        self.bird_group.add(self.bird)

        self.ground_group = pygame.sprite.Group()

        for i in range (2):
            ground = Ground(GROUND_WIDTH * i)
            self.ground_group.add(ground)

        self.pipe_group = pygame.sprite.Group()
        for i in range (2):
            pipes = get_random_pipes(SCREEN_WIDTH * i + 800)
            self.pipe_group.add(pipes[0])
            self.pipe_group.add(pipes[1])

        return self.get_state()


    def get_state(self):
        """
        get birds position ,
        check pipe pair position, check if it has passed?
        return NumPy array
        """

        bird_y = self.bird.rect[1]
        bird_vel = self.bird.speed

        pipe_x = SCREEN_WIDTH
        top_pipe_bottom_edge = 0
        bottom_pipe_top_edge = SCREEN_HEIGHT

        if len(self.pipe_group.sprites())>=2:

            #check first pipe postions
            bottom_pipe = self.pipe_group.sprites()[0]
            top_pipe = self.pipe_group.sprites()[1]
            #as pipes are in order bottom,top

            if self.bird.rect[0]>bottom_pipe.rect[0] + PIPE_WIDTH:
                if len(self.pipe_group.sprites())>=4:

                    bottom_pipe = self.pipe_group.sprites()[2]
                    top_pipe = self.pipe_group.sprites()[3]

            pipe_x = bottom_pipe.rect[0]

            bottom_pipe_top_edge = bottom_pipe.rect[1]
            top_pipe_bottom_edge = top_pipe.rect[1] + top_pipe.rect[3]


        horizontal_distance = pipe_x - self.bird.rect[0]

        state = [
                bird_y,
                bird_vel,
                horizontal_distance,
                top_pipe_bottom_edge,
                bottom_pipe_top_edge
                ]

        return np.array(state,dtype = np.float32)

    def step(self, action):
        if action == 1:
            self.bird.bump()

        # --- 2. THE PHYSICS UPDATE (Advance time by 1 frame) ---
        self.bird_group.update()
        self.ground_group.update()
        self.pipe_group.update()
        
        self.frames_survived += 1 # Tick our internal clock

        # Handle endless ground scrolling
        if is_off_screen(self.ground_group.sprites()[0]):
            self.ground_group.remove(self.ground_group.sprites()[0])
            new_ground = Ground(GROUND_WIDHT - 20)
            self.ground_group.add(new_ground)

        # Handle endless pipe spawning
        if is_off_screen(self.pipe_group.sprites()[0]):
            self.pipe_group.remove(self.pipe_group.sprites()[0])
            self.pipe_group.remove(self.pipe_group.sprites()[0]) # Remove both top and bottom
            pipes = get_random_pipes(SCREEN_WIDHT * 2)
            self.pipe_group.add(pipes[0])
            self.pipe_group.add(pipes[1])
        
        reward =0.1
        done=False

        if (pygame.sprite.groupcollide(self.bird_group, self.ground_group, False, False, pygame.sprite.collide_mask) or pygame.sprite.groupcollide(self.bird_group, self.pipe_group, False, False, pygame.sprite.collide_mask)):
            reward =-100
            done = True

        elif self.frames_survived>=self.max_frames:
            reward = 100
            done= True


        next_state = self.get_state()

        return next_state,reward,done

if __name__ == "__main__":
    env = FlappyEnv()
    state = env.reset()
    done = False
    
    print("Initial State:", state)
    
    while not done:
        # Pick a random action: 0 (nothing) or 1 (flap)
        action = 1 if random.random() > 0.9 else 0 
        
        next_state, reward, done = env.step(action)
        
        # We still need to draw it so we can see if it's working!
        env.screen.blit(BACKGROUND, (0, 0))
        env.bird_group.draw(env.screen)
        env.pipe_group.draw(env.screen)
        env.ground_group.draw(env.screen)
        pygame.display.update()
        env.clock.tick(15) # Keep it slow just for this test

    print("Game Over! Final Reward:", reward)
    pygame.quit()







