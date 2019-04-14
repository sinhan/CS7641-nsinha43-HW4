import gym
import time
from gym import wrappers
import os
import numpy as np
import matplotlib.pyplot as plt


# Plot results


# Environment initialization
def runql(myenv):
    print("Running qlearning for : ", myenv)
    def chunk_list(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'q_learning')
    env = wrappers.Monitor(gym.make(myenv), folder, force=True)

    # Q and rewards
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    iterations = []

    # Parameters
    alpha = 0.75
    discount = 0.90
    episodes = 500

    time1 = time.time()

    # Episodes
    for episode in range(episodes):
        # Refresh state
        state = env.reset()
        done = False
        t_reward = 0
        max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

        # Run episode
        for i in range(max_steps):
            if done:
                break

            current = state
            action = np.argmax(Q[current, :] + np.random.randn(1, env.action_space.n) * (1 / float(episode + 1)))

            state, reward, done, info = env.step(action)
            t_reward += reward
            Q[current, action] += alpha * (reward + discount * np.max(Q[state, :]) - Q[current, action])

        rewards.append(t_reward)
        iterations.append(i)

    # Close environment
    env.close()


    size = 5
    chunks = list(chunk_list(rewards, size))
    averages = [sum(chunk) / len(chunk) for chunk in chunks]

    plt.plot(range(0, len(rewards), size), averages)
    plt.title("Q-Learning for: " + myenv)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig("figs/QL-"+ myenv)
    plt.close()
    #plt.show()

    time2 = time.time()
    print('Elapsed time: %0.3f ms' % ((time2 - time1) * 1000.0))


# FROZEN LAKE

runql('FrozenLake-v0')
runql('FrozenLake8x8-v0')

# TAXI
runql('Taxi-v2')
