import numpy as np
import matplotlib.pyplot as plt

from LateralModeEnv import LateralModeEnv
if __name__ =='__main__':
    env = LateralModeEnv()
    env.dt = 0.01
    for episode in range(1):
        
        state  = env.reset()
        env.state = env.np_random.uniform(low=-env.high, high=env.high)
        episode_reward = 0
        
        for step in range(500):
            action = np.array([0,0])
            state_new, reward, done, _ = env.step(action)

            episode_reward += reward  # 积累式reward 除了能打出来积累值没什么用
            state = state_new

        print(episode, ': ', episode_reward)
        
    env.render()# 与其他环境不同,只能在每一次epsisode结束之后才能render()
    plt.grid(True)
    