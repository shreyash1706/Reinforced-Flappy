import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import random
from collections import deque,namedtuple

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Grab a random batch of memories to train on
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(self,state_size,action_size):
        super(QNetwork,self).__init__()
        self.fc1 = nn.Linear(state_size,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.Linear(128,action_size)

    def forward(self,state):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size=5, action_size=2):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters (The dials you can tune later)
        self.gamma = 0.99           # How much it cares about future rewards
        self.lr = 1e-4              # Learning Rate
        self.batch_size = 256        # How many memories it reviews at once
        
        # Epsilon-greedy exploration (Controls randomness)
        self.epsilon = 1.0          # 1.0 means 100% random actions at the start
        self.epsilon_min = 0.01     # Lowest randomness allowed (1%)
        self.epsilon_decay = 0.995  # How fast it stops being random per episode
        
        # Initialize the Network and Optimizer
        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() # Mean Squared Error
        
        self.memory = ReplayBuffer(100000) # Remember the last 100,000 frames

    def act(self,state):

        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)

        return torch.argmax(q_values).items()

    def learn(self):

        if len(self.memory)< self.batch_size:
            return

        states, action, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        current_q_values = self.model(states).gather(1,actions)

        witrh torch,no_grad():
            max_next_q_values = self.model(next_states).max(1)[0].unsqueeze(1)

            target_q_values = rewards + (self.gamma*max_next_q_values*(1-dones))

        loss = self.criterion(current_q_values,target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def decay_epsilon(self):
        if self.epsilon> self.epsilon_min:
            self.epsilon *= self.epsilon_decay
