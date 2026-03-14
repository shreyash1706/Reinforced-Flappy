import torch
import numpy as np
import cv2
import os
import pygame
import matplotlib.pyplot as plt
from tqdm import tqdm
from flappy_env import FlappyEnv, SCREEN_WIDTH, SCREEN_HEIGHT, BACKGROUND
from dqn_agent import DQNAgent

# --- CONFIGURATION ---
EPISODES = 10000
RECORD_INTERVAL = 200  
MAX_FRAMES = 10000          # Changed to 10k
EARLY_STOPPING_SCORE = 9900 # Stop if the average hits 9900 (99% optimality)

if not os.path.exists("training_videos"):
    os.makedirs("training_videos")

def train():
    env = FlappyEnv()
    env.max_frames = MAX_FRAMES 
    agent = DQNAgent(state_size=5, action_size=2)
    
    best_score = 0
    recent_scores = []
    
    # Tracking for our graphs
    loss_history = []
    optimality_history = [] 

    print("🚀 Starting DQN Training...")

    # Wrap the loop in tqdm for a clean progress bar
    pbar = tqdm(range(1, EPISODES + 1), desc="Training", unit="ep")
    
    try: 
        for episode in pbar:
            state = env.reset()
            done = False
            score = 0
            episode_losses = [] # Track losses for this specific game
            
            # --- VIDEO RECORDING SETUP ---
            render_this_episode = (episode % RECORD_INTERVAL == 0)
            video_writer = None
            
            if render_this_episode:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                video_path = f"training_videos/episode_{episode}.mp4"
                video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (SCREEN_WIDTH, SCREEN_HEIGHT))

            # --- THE GAME LOOP ---
            while not done:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                
                agent.memory.push(state, action, reward, next_state, done)
                loss = agent.learn() # Make sure dqn_agent.py returns loss.item()
                
                if loss is not None:
                    episode_losses.append(loss)
                
                state = next_state
                score += 1 

                # --- VISUALS & RECORDING ---
                if render_this_episode:
                    env.screen.blit(BACKGROUND, (0, 0))
                    env.bird_group.draw(env.screen)
                    env.pipe_group.draw(env.screen)
                    env.ground_group.draw(env.screen)
                    pygame.display.update()
                    
                    view = pygame.surfarray.array3d(env.screen)
                    view = view.transpose([1, 0, 2])
                    img_bgr = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
                    video_writer.write(img_bgr)

            # --- END OF EPISODE LOGIC ---
            agent.decay_epsilon()
            
            if video_writer:
                video_writer.release()
                
            # Update our metrics
            recent_scores.append(score)
            if len(recent_scores) > 200: # We now require 400 consecutive games of proof
                recent_scores.pop(0) 
                
            avg_score = sum(recent_scores) / len(recent_scores)
            optimality_rate = (avg_score / MAX_FRAMES) * 100
            optimality_history.append(optimality_rate)
            
            if episode_losses:
                avg_episode_loss = sum(episode_losses) / len(episode_losses)
                loss_history.append(avg_episode_loss)
            else:
                loss_history.append(0)

            # 1. Update the TQDM progress bar silently
            pbar.set_postfix({
                "Avg Score": f"{avg_score:.0f}", 
                "Opt %": f"{optimality_rate:.1f}%", 
                "Eps": f"{agent.epsilon:.2f}"
            })

            # 2. Print a permanent log line every 100 episodes
            if episode % 100 == 0:
                pbar.write(f"📊 Ep: {episode} | Last Score: {score} | Avg: {avg_score:.0f} | Opt: {optimality_rate:.1f}% | Loss: {loss_history[-1]:.4f}")

            # Save the best model
            if score > best_score:
                best_score = score
                torch.save(agent.model.state_dict(), "flappy_dqn_best.pth")
                pbar.write(f"🌟 New High Score: {best_score}! Model saved.")

            # --- EARLY STOPPING ---
            # If it hits a 99% average over the last 400 games, it wins!
            if avg_score >= EARLY_STOPPING_SCORE and len(recent_scores) == 400:
                pbar.write(f"🏆 Solved! The AI mastered the game.")
                torch.save(agent.model.state_dict(), "flappy_dqn_final.pth")
                break
        
    except KeyboardInterrupt:
        print("Interrupted, saving progress.")

    pygame.quit()
    
# --- PLOT 1: OPTIMALITY RATE ---
    print("📈 Generating Optimality Curve...")
    plt.figure(figsize=(10, 5))
    plt.plot(optimality_history, color='tab:red')
    plt.title('DQN Flappy Bird: Optimality Rate over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Optimality Rate (%)')
    plt.grid(True, alpha=0.3)
    plt.savefig('optimality_curve.png')
    plt.close() # Close to start a fresh figure

    # --- PLOT 2: LOSS (LOG SCALE) ---
    print("📉 Generating Loss Curve...")
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color='tab:blue')
    plt.yscale('log') # This is the magic line for your millions-scale loss
    plt.title('DQN Flappy Bird: Training Loss (Log Scale)')
    plt.xlabel('Episodes')
    plt.ylabel('Average Loss')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig('loss_curve.png')
    plt.close()

    print("✅ Graphs saved successfully as 'optimality_curve.png' and 'loss_curve.png'")

if __name__ == "__main__":
    train()
