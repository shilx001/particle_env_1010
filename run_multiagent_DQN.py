import numpy as np
import matplotlib.pyplot as plt
import pickle
from DQN_Agent import DeepQNetwork
from particle_env import *

env = ParticleEnv()

total_reward = []
total_collision = []
total_distance = []

agents = np.zeros(3, dtype=object)
for i in range(3):
    agents[i] = DeepQNetwork(25, 8, name='agent_' + str(i))

for episode in range(1000):
    state = env.reset()
    cum_reward = 0
    collision_count = 0
    distance_count = 0
    for step in range(200):
        action = []
        agent_status = [False, False, False]
        for i in range(3):
            if not agent_status[i]:  # 如果该agent未达到目标
                agent_observation = state[i]
                agent_action, _ = agents[i].choose_action(agent_observation)
                action.append(agent_action)
            else:#否则则保持静止
                agent_action = 24
                action.append(agent_action)
        next_state, reward, done, num_collision, target_distance = env.step(action)
        cum_reward += np.sum(np.array(reward))
        collision_count += num_collision
        distance_count += target_distance
        if step == 200 - 1:
            print('Episode ', episode, 'finished at reward ', cum_reward)
        for i in range(3):  # store transitions and learn
            agent_observation = state[i]
            agent_observation_ = next_state[i]
            agent_action = action[i]
            agent_reward = reward[i]
            agent_done = done[i]
            if agent_done:  # 如果该agent已完成任务，则设其为done，不再学习
                agent_status[i] = True
                continue
            # 否则存储当前，并进行学习
            agents[i].store_transition(agent_observation, agent_action, agent_reward, agent_observation_,
                                       int(agent_done))
            agents[i].learn()
        state = next_state
    total_reward.append(cum_reward)
    total_collision.append(collision_count)
    total_distance.append(distance_count)

pickle.dump(total_reward, open('total_reward-Independent DQN', 'wb'))
pickle.dump(total_collision, open('collision_count-Independent DQN', 'wb'))
pickle.dump(total_distance, open('distance_count-Independent DQN', 'wb'))

plt.plot(np.array(total_reward) / 200)
plt.xlabel('Episode')
plt.ylabel('Average reward')
plt.title('Average reward per step')
plt.savefig('Independent DQN')
