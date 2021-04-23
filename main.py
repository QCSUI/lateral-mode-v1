import gym
from ddpg import Agent
from LateralModeEnv import LateralModeEnv
import torch
import matplotlib.pyplot as plt
import time
import os 

if __name__ == '__main__':

    env = LateralModeEnv()
    env.reset()


    try:
        agent = torch.load('net.pkl')
        
        filemt= time.localtime(os.stat('net.pkl').st_mtime) #获取上一次修改时间
        os.rename('net.pkl','net'+time.strftime('%Y%m%d%H%M%S',filemt)+'.pkl')

        
    except FileNotFoundError:
        params = {
            'env': env,
            'gamma': 0.99,
            'actor_lr': 0.002,
            'critic_lr': 0.002,    
            'tau': 0.02,
            'capacity': 10000,
            'batch_size': 32,
        }
        agent = Agent(**params)
         
            
    

    for episode in range(400):
        state  = env.reset()
        episode_reward = 0
        
        for step in range(500):
            action = agent.forward(state)
            state_new, reward, done, _ = env.step(action)
            agent.put(state, action, reward, state_new)

            episode_reward += reward  # 积累式reward 除了能打出来积累值没什么用
            state = state_new

            agent.learn()
            if done: break

        print(episode, ': ',len(env.buffer),'Eps', episode_reward)
        
    env.render()# 与其他环境不同,只能在每一次epsisode结束之后才能render()
    plt.grid(True)
    time.sleep(1)        
    torch.save(agent, 'net.pkl')  # 保存整个网络
    
    
    