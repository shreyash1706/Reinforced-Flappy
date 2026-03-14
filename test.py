import pygame
import torch
import sys
from flappy_env import FlappyEnv, SCREEN_WIDTH, SCREEN_HEIGHT, BACKGROUND
from dqn_agent import DQNAgent

# --- CONFIGURATION ---
MODEL_PATH = "flappy_dqn_best.pth"
FPS = 15
MAX_FRAMES = 10000

def test():
    # Initialize Pygame and Mixer for Audio
    pygame.init()
    pygame.mixer.init()
    env = FlappyEnv()
    # Load the original audio and start screen assets
    try:
        wing_sound = pygame.mixer.Sound('assets/audio/wing.wav')
        hit_sound = pygame.mixer.Sound('assets/audio/hit.wav')
        begin_image = pygame.image.load('assets/sprites/message.png').convert_alpha()
    except FileNotFoundError:
        print("❌ Warning: Could not find audio or image assets in 'assets/' folder. Running without them.")
        wing_sound, hit_sound, begin_image = None, None, None

    
    
    # --- LOAD THE AI BRAIN ---
    agent = DQNAgent(state_size=5, action_size=2)
    try:
        # Load the weights into the model
        agent.model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        agent.model.eval()  # Lock the brain into evaluation mode (no learning)
        print(f"✅ Successfully loaded '{MODEL_PATH}'")
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{MODEL_PATH}'. Check the file name!")
        return

    # 100% EXPLOITATION - ZERO RANDOMNESS
    agent.epsilon = 0.0 

    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 28, bold=True)

    # --- WAITING SCREEN ---
    waiting = True
    print("Waiting for user to start...")
    
    while waiting:
        env.screen.blit(BACKGROUND, (0, 0))
        
        # Draw the original start message if it exists
        if begin_image:
            env.screen.blit(begin_image, (120, 150))
            
        # Draw text instructions
        text = font.render("AI is Ready! Press SPACE", True, (255, 255, 255))
        outline = font.render("AI is Ready! Press SPACE", True, (0, 0, 0))
        rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100))
        env.screen.blit(outline, (rect.x + 2, rect.y + 2))
        env.screen.blit(text, rect)
        
        pygame.display.update()
        clock.tick(15)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and (event.key == pygame.K_SPACE or event.key == pygame.K_UP):
                waiting = False

    # --- THE SHOWCASE LOOP ---
    state = env.reset()
    done = False
    score = 0

    print("🚀 AI is taking control...")

    while not done and score < MAX_FRAMES:
        clock.tick(FPS) # Lock to 15 FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Ask the AI for the perfect move
        action = agent.act(state)
        
        # Play the wing flap audio if the AI decides to jump
        if action == 1 and wing_sound:
            wing_sound.play()

        # Advance the environment
        next_state, reward, done = env.step(action)
        state = next_state
        score += 1

        # Render the game
        env.screen.blit(BACKGROUND, (0, 0))
        env.bird_group.draw(env.screen)
        env.pipe_group.draw(env.screen)
        env.ground_group.draw(env.screen)

        # Draw the live frame counter
        score_text = font.render(f"Frames Survived: {score}", True, (255, 255, 255))
        outline_text = font.render(f"Frames Survived: {score}", True, (0, 0, 0))
        env.screen.blit(outline_text, (12, 12))
        env.screen.blit(score_text, (10, 10))

        pygame.display.update()

        # If the AI crashes, play the hit sound and freeze for a second
        if done:
            if hit_sound:
                hit_sound.play()
            pygame.time.delay(1000)

    # --- END SCREEN ---
    if score >= MAX_FRAMES:
        print(f"🏆 Showcase Complete! The AI flawlessly survived {MAX_FRAMES} frames.")
    else:
        print(f"💥 The AI crashed after {score} frames.")

    pygame.quit()

if __name__ == "__main__":
    test()
