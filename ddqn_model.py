import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random
import numpy as np

class DDQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save(self, file_name='model_ddqn.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class DDQNTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
         # game_over = torch.tensor(game_over, dtype=torch.float)
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )

        pred = self.model(state)
        next_pred = self.model(next_state)
        q_values = pred.clone()

        # 2: Compute Q values using target network
        with torch.no_grad():
            next_pred = self.model(next_state)
            next_q_values = self.target_model(next_state)
            best_actions = torch.argmax(next_pred.unsqueeze(0), dim=1)
        
        # 3: Update Q values using DDQN update rule
        for i in range(len(game_over)):
            Q_new = reward[i]
            if not game_over[i]:
                Q_new += self.gamma * next_q_values[i][best_actions[i].item()]
                Q_new = reward[i] + self.gamma * torch.max(next_pred[i])
            q_values[i][action[i]] = Q_new

        # Zero gradients, perform backward pass, and update the weights
        self.optimizer.zero_grad()
        loss = self.criterion(q_values, pred)
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())