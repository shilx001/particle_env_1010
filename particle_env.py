import numpy as np


class ParticleEnv:
    def __init__(self, n_agent=3, n_landmark=3):
        self.agents = []
        self.landmarks = []
        self.n_agent = n_agent
        self.n_landmark = n_landmark

    def reset(self):
        # 初始化每个agent
        agent_center = np.array([-1, -1])
        landmark_center = np.array([1, 1])
        for i in range(self.n_agent):
            self.agents.append(Agent(np.random.uniform(-1, -0.5, 2) * i + agent_center))
        for i in range(self.n_landmark):
            self.landmarks.append(Landmark(np.random.uniform(0.5, 1, 2) * i + landmark_center))
        #返回每个agent的位置和每个landmark的位置

    def step(self, action):
        #针对每个agent返回agent
        observation=[]
        reward=[]
        done=[]
        assert len(action) == self.n_agent
        # 根据每个agent的action,返回每个agent的观测值


class Agent:
    def __init__(self, pos, size=0.15):
        self.position = pos  # agent初试位置
        self.size = size


class Landmark:
    def __init__(self, pos, size=0.05):
        self.position = pos  # landmark的初试位置
        self.size = size
