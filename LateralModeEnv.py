# -*- coding: utf-8 -*-
"""
线性系统：小扰动理论，x_dot = AX+Bc
We are given 𝑝,𝑞,𝑟 in a body-fixed system to find the Euler angle rate  𝜙,θ,𝜓 .
Body axes components of angular velocity with respect to inertial axes 𝑝,𝑞,𝑟 are availed from rate gyros. 
神经网络控制的缺点:只能在特定的训练区才能实现，无法解决轨迹跟踪精度的问题。

"""
'''
X_i+1 = X_i + dt * A * X_i
X_i+1 = (I+dt*A) X_i
'''
import gym
from gym import spaces #Box是干嘛用的
from gym.utils import seeding
import numpy as np
from os import path
import matplotlib.pyplot as plt 
import time


class LateralModeEnv(gym.Env):
    def __init__(self):
        self.dt = 0.1

        self.A = np.array([[-0.0999,0.0000,0.1153,-1.0000], 
                            [-1.6038,-1.0932,0.0,0.2850 ],
                            [0.0,1.0,0.0,0.0],
                            [0.4089,-.0395,0.0,-.2454]] )
        self.B = np.array([[0.,0.0182],
                           [0.3215,0.0868],
                           [0.,0.],
                           [-0.0017, -0.2440]])
        
        self.Ap = np.eye(4)+self.dt*self.A
        self.App =self.Ap@self.Ap@self.Ap@self.Ap
        max_beta = 0.5
        max_p = 0.1
        max_phi = 0.5
        max_r = 0.1
        max_a_rad = 15/57.3
        max_r_rad = 15/57.3
        self.high = np.array([max_beta,max_p,max_phi,max_r])
        self.action_high = np.array([max_a_rad,max_r_rad])
        
        self.action_space = spaces.Box(low=-self.action_high, high=self.action_high, dtype=np.float32)
        # 用法 ： action_space.contains(action)
        # 用法 ： action_space.sample() 
        self.observation_space = spaces.Box(low=-self.high, high=self.high, dtype=np.float32)
        self.seed()
        self.buffer = []
        self.MAX_BUFFER = 500
        
        
    def step(self,action,ref_beta = 0,ref_p=0,ref_phi=0,ref_r=0):
        '''
        X Y Z | Coordinate Sys.
        p q r | Angular Velocity
        φ θ ψ | Angle
                
        x =β p φ r
        
        
        EE : X_i+1 = X_i + X_dot * dt
        RK4: k1 =
             k2 = 
             k3 = 
             k4 = 
             X_i+1 = X_i + (k1+2*k2+2*k3+k4) * dt / 6
             
        '''

        action = np.clip(action, -self.action_high,self.action_high)
        self.last_action = action # for rendering
        '''k1 =
        k2 = 
        k3 = 
        k4 ='''
        newstate = self.Ap @ self.state + self.B@action*self.dt
        
        self.ref = np.array([ref_beta,ref_p,ref_phi,ref_r])
        self.error = self.ref-newstate

        costs = (self.error) @ (self.error)+ 0.05*action@action
        costs = costs*10 
        if len(self.buffer)>self.MAX_BUFFER:
            self.buffer.pop(0)
        self.buffer.append(newstate)
        
        self.state = newstate
        return self.error,-costs,not self.observation_space.contains(self.state),{}
    
    def reset(self):
        self.state = self.np_random.uniform(low=-self.high/3, high=self.high/3)
        self.last_action = None
        self.buffer = []
        self.buffer.append(self.state)
        # plt.clf()
        plt.close()
        
        return self.state
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)  #这是干嘛用的？ 
        return [seed]
    
    def render(self):
        beta_serie = np.array([state[0] for state in self.buffer])
        p_serie = np.array([state[1] for state in self.buffer])
        phi_serie = np.array([state[2] for state in self.buffer])
        r_serie = np.array([state[3] for state in self.buffer])
        t_serie = np.linspace(0,(beta_serie.size-1)*self.dt,beta_serie.size)
        

        plt.subplot(411)
        plt.plot(t_serie,beta_serie)
        plt.ylabel('Beta')
        
        
        plt.subplot(412)
        plt.plot(t_serie,p_serie)
        plt.ylabel('p')
        
        plt.subplot(413)
        plt.plot(t_serie,phi_serie)
        plt.ylabel('Phi')
        plt.subplot(414)
        plt.plot(t_serie,r_serie)
        plt.ylabel('r')
        
      
        
    
if __name__ == '__main__':
    env = LateralModeEnv()
    env.reset()
    env.step(env.action_space.sample())

    
        
    
