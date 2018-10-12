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
        observation = []
        for i in range(self.n_agent):
            self.agents.append(Agent(np.random.uniform(-1 * (i + 1), -0.5 * (i + 1), 2) + agent_center))
        for i in range(self.n_landmark):
            self.landmarks.append(Landmark(np.random.uniform(0.5 * (i + 1), 1 * (i + 1), 2) + landmark_center))
        # 返回每个agent的位置和每个landmark的位置
        for i in range(self.n_agent):
            # agent observation format: [agent_pos, other_agent_pos, target_pos]
            agent_observation = [self.agents[i].position]
            for j in range(self.n_agent):
                if j == i:
                    continue
                agent_observation.append(self.agents[j].position - self.agents[i].position)
            agent_observation.append(self.landmarks[i].position)
            observation.append(agent_observation)
        return observation

    def step(self, action):
        # 针对每个agent返回agent
        # 每个action是10维
        observation = []
        reward = []
        done = []
        num_collision = 0
        # action number
        assert len(action) == self.n_agent
        # 根据每个agent的action,返回每个agent的观测值
        # agent action:
        # 0: move up
        # 1: move down
        # 2: move left
        # 3: move right
        # 4: move upper left
        # 5: move upper right
        # 6: move lower left
        # 7: move lower right
        # agent move step: 0.01, 0.1, 1
        # each action: range(25)
        for i in range(self.n_agent):  # 针对agent
            # 先计算distance
            if int(action[i] / 8) == 0:
                distance = 0.01
            elif int(action[i] / 8) == 1:
                distance = 0.1
            elif int(action[i] / 8) == 2:
                distance = 1
            else:
                distance = 0
            if action[i] % 8 == 0:
                self.agents[i].position[1] += distance
            elif action[i] % 8 == 1:
                self.agents[i].position[1] -= distance
            elif action[i] % 8 == 2:
                self.agents[i].position[0] -= distance
            elif action[i] % 8 == 3:
                self.agents[i].position[0] += distance
            elif action[i] % 8 == 4:
                self.agents[i].position[0] -= distance / np.sqrt(2)
                self.agents[i].position[1] += distance / np.sqrt(2)
            elif action[i] % 8 == 5:
                self.agents[i].position[0] += distance / np.sqrt(2)
                self.agents[i].position[1] += distance / np.sqrt(2)
            elif action[i] % 8 == 6:
                self.agents[i].position[0] -= distance / np.sqrt(2)
                self.agents[i].position[1] -= distance / np.sqrt(2)
            elif action[i] % 8 == 7:
                self.agents[i].position[0] += distance / np.sqrt(2)
                self.agents[i].position[1] -= distance / np.sqrt(2)
            else:
                continue
        for i in range(self.n_agent):  # 计算observation,reward, done
            # agent observation format: [agent_pos, other_agent_pos, target_pos]
            agent_observation = [self.agents[i].position]
            agent_reward = -np.sqrt(np.square(np.sum(self.agents[i].position - self.landmarks[i].position)))
            agent_done = np.sqrt(np.square(np.sum(self.agents[i].position - self.landmarks[i].position))) <= \
                         self.agents[i].size + self.landmarks[i].size
            for j in range(self.n_agent):
                if j == i:
                    continue
                agent_observation.append(self.agents[j].position - self.agents[i].position)
                # if collision then give penalty
                if np.sqrt(np.square(np.sum(self.agents[j].position - self.agents[i].position))) <= self.agents[
                    i].size + self.agents[j].size:
                    agent_reward -= 1
                    num_collision += 1
            agent_observation.append(self.landmarks[i].position)
            observation.append(agent_observation)
            done.append(agent_done)
            reward.append(agent_reward)
        return observation, reward, done, num_collision, -(agent_reward + num_collision)


class Agent:
    def __init__(self, pos, size=0.15):
        self.position = pos  # agent初试位置
        self.size = size


class Landmark:
    def __init__(self, pos, size=0.05):
        self.position = pos  # landmark的初试位置
        self.size = size
