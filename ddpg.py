"""
一种把一堆东西传给class的方法
    params = {
        'env': env,
        'gamma': 0.99,
        'actor_lr': 0.001,
        'critic_lr': 0.001,
        'tau': 0.02,
        'capacity': 10000,
        'batch_size': 32,
    }

    agent = Agent(**params)  # 双星号（**）将参数以字典的形式导入 
class ...
    def __init__(self, **kwargs):
        
        for key, value in kwargs.items():
            setattr(self, key, value)
            ...
            
    def put(self, *transition):  #将所有参数以元组(tuple)的形式导入

        
"""


import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, s):
        x = F.relu(self.linear1(s))  # State -> Linear -> Relu -> activation
        x = F.relu(self.linear2(x))  # activation -> Linear -> Relu -> activation
        x = torch.tanh(self.linear3(x))  # activition -> Linear -> tanh (-1,1) -> action

        return x


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)  # Same Structure as actor
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class Agent(object):
    def __init__(self, **kwargs):
        
        for key, value in kwargs.items():
            setattr(self, key, value)
            # **kwargs是前面params字典里面所有的变量的名称以及他们的值，现在已经是self中的数据

        s_dim = self.env.observation_space.shape[0]
        a_dim = self.env.action_space.shape[0]

        self.actor = Actor(s_dim, 128, a_dim)  # input_size, hidden_size, output_size
        self.actor_target = Actor(s_dim, 128, a_dim)
        
        self.critic = Critic(s_dim+a_dim, 128, a_dim)  # crtic把状态的dim以及action的dim一起纳入输入范围
        self.critic_target = Critic(s_dim+a_dim, 128, a_dim)
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        self.buffer = []
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
    def forward(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action = self.actor(state).squeeze(0).detach().numpy()
        return action
    
    def put(self, *transition): 
        if len(self.buffer)== self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
    
    def learn(self):
        if len(self.buffer) < self.batch_size:
            return 
        
        samples = random.sample(self.buffer, self.batch_size)
        
        state, action, r1, state_new = zip(*samples)
        
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size,-1)
        state_new = torch.tensor(state_new, dtype=torch.float)
        
        def critic_learn():
            a1 = self.actor_target(state_new).detach()
            y_true = r1 + self.gamma * self.critic_target(state_new, a1).detach()
            
            y_pred = self.critic(state, action)
            
            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()
            
        def actor_learn():
            loss = -torch.mean( self.critic(state, self.actor(state)) )
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()
                                           
        def soft_update(net_target, net, tau):
            for target_param, param  in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
        critic_learn()
        actor_learn()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)