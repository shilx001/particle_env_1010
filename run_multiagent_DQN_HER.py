import numpy as np
import matplotlib.pyplot as plt
import pickle
from DQN_HER_Agent import DeepQNetwork
from particle_env import *


def cal_distance(loc):
    return np.sqrt(np.square(np.sum(loc)))


def cal_reward(temp_state, temp_goal):
    # 根据state和goal计算是否达到目标
    collision = False
    other_pos = temp_state[2]
    for i in range(2):
        if cal_distance(other_pos[i]) <= 0.3:
            collision = True
            break
    return int(cal_distance(temp_state[0] - temp_goal) <= 0.2 and not collision)


env = ParticleEnv()

total_reward = []
total_collision = []
total_distance = []

agents = np.zeros(3, dtype=object)
k = 2

for i in range(3):
    agents[i] = DeepQNetwork(25, 8, name='agent_' + str(i))

for episode in range(1000):
    state = env.reset()
    cum_reward = 0
    collision_count = 0
    distance_count = 0
    # 针对每个episode建立trajectory list
    trajectory = []
    agent_status = [False, False, False]  # 记录agent是否完成
    goal = np.zeros([3, 2])
    for i in range(3):
        temp_state = state[i]
        goal[i] = temp_state[3]
    for step in range(200):  # execution
        action = []
        for i in range(3):
            if not agent_status[i]:  # 如果未完成
                agent_observation = state[i]
                agent_action, _ = agents[i].choose_action(agent_observation, np.reshape(goal[i], [1, 2]))
                action.append(agent_action)
            else:
                agent_action = 24
                action.append(agent_action)
        next_state, reward, done, num_collision, target_distance = env.step(action)
        trajectory.append([state, action, reward, next_state])
        for i in range(3):
            if done[i]:
                agent_status[i] = True
        cum_reward += np.sum(np.array(reward))
        collision_count += num_collision
        distance_count += target_distance
        state = next_state
        if step == 200 - 1:
            print('Episode ', episode, 'finished at reward ', cum_reward)
    total_reward.append(cum_reward)
    total_collision.append(collision_count)
    total_distance.append(distance_count)
    # 执行结束开始学习
    for step in range(200):  # Hindsight Experience Replay
        # reward设置要针对每个agent
        experience = trajectory[step]
        state = experience[0]
        action = experience[1]
        next_state = experience[3]
        for i in range(3):  # 针对每个agent
            agent_state = state[i]
            agent_action = action[i]
            agent_next_state = next_state[i]
            # cal reward:看看是否和其他agent碰撞，且是否达到目标
            agent_reward = cal_reward(agent_next_state, goal[i])
            agent_done = 0
            if agent_reward is 1:
                agent_done = 1
            agents[i].store_transition(agent_state, agent_action, agent_reward,
                                       agent_next_state, agent_done, goal[i])  # normal experience replay
            # sample a set of goals from historical trajectories
            index = np.random.choice(200, k)
            for j in index:
                experience = trajectory[j]
                state = experience[0]
                action = experience[1]
                next_state = experience[3]
                agent_state = state[i]
                agent_action = action[i]
                agent_next_state = next_state[i]
                # cal reward:看看是否和其他agent碰撞，且是否达到目标
                agent_goal = agent_next_state[0]
                agent_reward = cal_reward(agent_next_state, agent_goal)
                agent_done = 0
                if agent_reward is 1:
                    agent_done = 1
                agents[i].store_transition(agent_state, agent_action, agent_reward, agent_next_state, agent_done,
                                           agent_goal)
    for step in range(200):  # learning process
        for i in range(3):
            if not agent_status[i]:  # 如果未完成则学习
                agents[i].learn()
pickle.dump(total_reward, open('total_reward-Independent DQN HER no collision', 'wb'))
pickle.dump(total_collision, open('collision_count-Independent DQN HER no collision', 'wb'))
pickle.dump(total_distance, open('distance_count-Independent DQN HER no collision', 'wb'))

plt.plot(np.array(total_reward) / 200)
plt.xlabel('Episode')
plt.ylabel('Average reward')
plt.title('Average reward per step')
plt.savefig('Independent DQN HER no collision')
