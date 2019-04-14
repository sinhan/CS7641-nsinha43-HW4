import numpy as np
import gym
import random
import pandas as pd
from time import time
from matplotlib import pyplot as ply
from gym.envs.registration import register

taxi = gym.make('Taxi-v2').env
taxi.render()

def value_iteration(env,descrip):
    V = np.zeros(env.nS,dtype='float64')  # initialize value-function
    max_iter = 2000
    theta = 1e-20
    timeV = []
    nIter = []
    rewards = []

    for g in np.linspace(0.5, 1, 11):
        valueDiff = []
        t0 = time()
        for i in range(max_iter):
            prev_V = np.copy(V)
            for state in range(env.nS):
                A = next_step(env, state, V, gamma =g)
                V[state] = max(A)

            vD = np.sum(np.fabs(prev_V - V))
            valueDiff.append(vD)
            if (vD <= theta):
                tD = time()-t0
                print ('Value-iteration converged at iteration# %d.' %(i+1))
                print ('Value-iteration convergence took %.2fs' %(tD))
                break

        ply.plot(valueDiff,label= 'gamma= %2.2g, iter= %3d' %(g,i+1))
        timeV.append(tD)
        nIter.append(i+1)

    ply.title(descrip + ': Value Iteration Convergence',fontsize=14, fontweight='bold')
    ply.ylabel('||v-v*||')
    ply.xlabel('# Iterations')
    ply.legend(loc = 'upper right')
    ply.ylim([0,.25])

    fig1 = ply.gcf()
    #ply.show()
    ply.draw()
    fig1.savefig('figs/VI_'+ descrip +'.png', bbox_inches='tight', dpi=200)
    ply.close()

    tim = np.asarray(timeV)
    # nIte = np.asarray(g)
    ply.plot(tim,np.linspace(0.5, 1, 11))
    ply.title(descrip + 'Value Iteration Convergence Time',fontsize=14, fontweight='bold')
    ply.xlabel('Time to convergence(s)')
    ply.ylabel('Gamma (discount rate)')

    fig1 = ply.gcf()
    #ply.show()
    ply.draw()
    fig1.savefig('figs/GammaTime_'+ descrip +'.png', bbox_inches='tight', dpi=200)
    ply.close()

    # Extract the policy given a value-function
    suc = []

    policy = np.zeros(env.nS,dtype='int16')
    for state in range(env.nS):
        A = next_step(env, state, V,gamma=0.90)
        policy[state] = np.argmax(A)

    sucess,steps = run_episodes(env,policy,descrip+': Value Iteration Optimal Policy')
    suc.append(sucess)

    # view_episode(env, policy)
    # displayResults(env,policy)

    return policy

def next_step(env,state, V,gamma):
    A = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, next_state, reward, done in env.P[state][a]:
            A[a] += prob * (reward + gamma * V[next_state])
    return A


def run_episodes(env, policy,descrip):
    numIter = 5000
    rewards = []
    numSteps = np.zeros(numIter,dtype='float32')
    numEpisodes = env.spec.max_episode_steps - 1
    env.reset()
    for iter in range(numIter):
        state = env.reset()
        step = 0
        done = False
        totR = 0;
        for step in range(numEpisodes):
            new_state, reward, done, _ = env.step(policy[state])
            totR+= reward
            if done:
               break
            state = new_state

        numSteps[iter] = step
        rewards.append(totR)
    env.close()

        ## measurePerformance(rewards, numSteps)
    sucessRatio = np.sum(np.asarray(rewards)) /numIter
    # avgSteps = np.mean(numSteps[rewards == 1])
    # stdSteps = np.std(numSteps[rewards == 1])
    tmp = pd.DataFrame(data = rewards)
    ply.plot(range(numIter), tmp.rolling(200).mean())
    ply.title(descrip + ': Mean Rewards' , fontsize=14, fontweight='bold')
    ply.xlabel('# of episodes')
    ply.ylabel('Rolling Mean Reward')
    ply.ylim([-50,50])
    fig1 = ply.gcf()
    #ply.show()
    ply.draw()
    fig1.savefig('figs/MeanRewards_' + descrip + '.png', bbox_inches='tight', dpi=200)
    ply.close()

    #print(policy.reshape(np.int(np.sqrt(env.nS)),np.int(np.sqrt(env.nS))))
    mapping={0: "S", 1: "N", 2: "E", 3: "W", 4: "P", 5: "D"}
    print(np.array([mapping[action] for action in policy]), '\n')
    taxi.render

    return sucessRatio, numSteps

def view_episode(env, policy):
    obs = env.reset()
    step_idx = 0
    while True:
        env.render()
        obs, _, done , _ = env.step(int(policy[obs]))
        step_idx += 1
        if done:
            break
    return


## Policy Iteration
def policy_iteration(env, descrip,gamma=.90):

    policy = np.random.choice(env.nA, size=(env.nS)) # Random policies
    maxIter = 2000
    V = np.zeros(env.nS)
    print("ENV ns:",env.nS)
    idx = 0
    ix = 0
    theta = 1e-20
    timeV = []
    nVIter = []
    nPIter = []
    ncIter = []

    for g in np.linspace(0.5, 1, 11):
         #for c in range(maxIter):
            t0 = time()
            prev_sum=0
            while True:
                prev_V = np.copy(V)
                for state in range(env.nS):
                    currPolicy = policy[state]
                    # for prb, nextS, r,_ in env.P[state][currPolicy]:
                    #     V[state]= sum(prb * (r + gamma * prev_V[nextS]))
                    V[state] = sum([p * (r + g * prev_V[s_]) for p, s_, r, _ in env.P[state][currPolicy]])
                curr_sum=np.sum((np.fabs(prev_V - V)))
                #if (np.sum((np.fabs(prev_V - V))) <= theta):
                if (curr_sum == prev_sum):
                   print('Not converging anymore.At iteraration# %d.' % (idx))
                   break
                if (curr_sum <= theta):
                   print('Max policy value converged at iteration# %d.' % (idx))
                   nVIter.append(idx)
                   break
                idx+=1
                prev_sum = curr_sum
                #print("At iteration :", idx)
                #print(np.sum((np.fabs(prev_V - V))))

            while True:
                prev_policy = np.copy(policy)
                policy = np.zeros(env.nS,dtype='int16')


                for state in range(env.nS):
                    value = next_step(env, state, V, gamma=g)  # Extract the policy for value
                    currBest = np.argmax(value)

                    # if (currBest != policy[state]):
                    #     policy_stable = False
                    policy[state] = currBest


                if np.all(prev_policy == policy)  :
                   print('Policy-iteration converged at iteration# %d.' % (ix))
                   nPIter.append(ix)
                   break
                   # return policy, V
                if (curr_sum == prev_sum):
                    break
                ix+=1
            if (curr_sum != prev_sum):
                timeV.append(time()-t0)
         #ncIter.append(c)

    tim = np.asarray(timeV)
    nIte = np.asarray(nPIter)
    print(tim.shape)
    print(nIte.shape)
    ply.plot(tim, nIte)
    ply.title(descrip + ': Policy Iteration Convergence Time' , fontsize=14, fontweight='bold')
    ply.xlabel('Time to convergence(s)')
    ply.ylabel('# Iterations')

    fig1 = ply.gcf()
    #ply.show()
    ply.draw()
    fig1.savefig('figs/Time_PI_' + descrip + '.png', bbox_inches='tight', dpi=200)
    ply.close()

    ##
    #ply.plot(tim,np.linspace(0.5, 1, 10))
    ply.plot(tim,np.linspace(0.5, 1, ix))
    ply.title(descrip + ': Policy Iteration Convergence Time',fontsize=14, fontweight='bold')
    ply.xlabel('Time to convergence(s)')
    ply.ylabel('Gamma (discount rate)')

    # ply.plot(tim, np.asarray(nVIter))
    # ply.title('Policy Iteration Value Convergence Time_' + descrip + 'MDP', fontsize=14, fontweight='bold')
    # ply.xlabel('Time to convergence(s)')
    # ply.ylabel('# Iterations')
    fig1 = ply.gcf()
    #ply.show()
    ply.draw()
    fig1.savefig('figs/GammaTime_PI_Value_' + descrip + '.png', bbox_inches='tight', dpi=200)
    ply.close()

##
    #ply.plot(np.linspace(0.5, 1, 10), nIte)
    ply.plot(np.linspace(0.5, 1, ix), nIte)
    ply.title(descrip + ': Policy Iteration Convergence Time', fontsize=14, fontweight='bold')
    ply.xlabel('Discount rate(gamma)')
    ply.ylabel('# Iterations')

    fig1 = ply.gcf()
    #ply.show()
    ply.draw()
    fig1.savefig('figs/IterNum_PI_' + descrip + '.png', bbox_inches='tight', dpi=200)
    ply.close()

    sucess,steps = run_episodes(env,policy,descrip+': Policy Iteration Optimal Policy ')
    # view_episode(env, policy)
    # displayResults(env,policy)


def QLearningRL(env, descrip,gamma = .75):
    import sys
    max_steps = env.spec.max_episode_steps # Max steps per episode
    print(max_steps)
    Qtbl = np.zeros([env.nS, env.nA])
    num_episodes = 20000
    #num_episodes = 20000 
    lr = 0.75

    steps = []
    timeV = []
    epsi = []

    # Exploration parameters
      # Exploration rate
    max_epsilon = 1.0  # Exploration probability at start
    min_epsilon = 0.1  # Minimum exploration probability
    dr = 0.005  # Exponential decay rate for exploration prob
    rewards = []  # np.zeros(num_episodes) # List of rewards
    epsilon = 1.0
    # for newdr in np.linspace(0.01,0.005,5):

    for episode in range(num_episodes):
        # Reset the environment
        #print('-------------------------------- episode ', episode)
        state = env.reset()
        done = False
        totalR = 0
        t0 = time()
        epsi.append(epsilon)

        for step in range(max_steps):
            #print('-------> step ', step)
            thresh = random.uniform(0, 1) ## First we randomize a number
            # If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
            # if thresh > epsilon:
            #    a = np.argmax(Qtbl[state, :])
            # else:              # Else doing a random choice --> exploration
            #    a = env.action_space.sample()

            #Choose an action by greedily (with noise) picking from Q table
            a = np.argmax(Qtbl[state,:] + np.random.randn(1,env.nA)*(1./(episode+1)))
            nextSt, r, done,_ = env.step(a)  #new state and reward
            #print(nextSt, r, done)

            # Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            # Update Q-Table with new knowledge
            Qtbl[state, a]+= lr*((r + gamma * np.max(Qtbl[nextSt,:])) - Qtbl[state, a])

            state = nextSt  # new state is state
            totalR+=  r
            # tD = time() - t0

            if done == True: # finish episode
               break

       # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon-min_epsilon) * np.exp(-dr * episode)
        rewards.append(totalR)
        timeV.append(time() - t0)

     
    #ply.title(descrip + ': QL Mean Rewards With Varying Decay Rates', fontsize=14, fontweight='bold')
    #ply.xlabel('# of episodes')
    #ply.ylabel('Rolling Mean Reward')
    #ply.legend(loc = 'upper right')
    #ply.ylim([0,1])
    #fig1 = ply.gcf()
    ##ply.show()
    #ply.draw()
    #fig1.savefig('figs/RollingMean_QL_varyingDR' + descrip + '.png', bbox_inches='tight', dpi=200)
    #ply.close()



    optimalPolicy = np.argmax(Qtbl,axis=1)
    # print("Score over time: " + str(sum(rewards) / num_episodes))
    # # print(optimalPolicy)
    # # displayResults(env, optimalPolicy)
    #
    # Calculate and print the average reward per thousand episodes
    rolling_reward = np.split(np.array(rewards), num_episodes / 500)
    count = 500
    print("*****Average reward per thousand episode*****\n")
    for rr in rolling_reward:
        print(count, ": ", str(sum(rr / 500)))
        count += 500


    tmp = pd.DataFrame(data = rewards)
    print(tmp.head())
    print(tmp.rolling(500).mean()) 
    tmp2 = pd.DataFrame(data=epsi)

    ply.plot(range(num_episodes - 49 ), tmp.rolling(50).mean().dropna())
    ply.title(descrip + ': MeanRewards without DecayRate' , fontsize=14, fontweight='bold')
    ply.xlabel('# of episodes')
    ply.ylabel('Rolling Mean Reward')
    fig1 = ply.gcf()
    #ply.show()
    ply.draw()
    fig1.savefig('figs/RollingMean_woEpsilon_' + descrip + '.png', bbox_inches='tight', dpi=200)
    ply.close()

    ply.plot(tmp2.rolling(50).mean().dropna(), tmp.rolling(50).mean().dropna(),'-',  color="g")
    ply.title(descrip + ': QL Epsilon vs. Rewards', fontsize=14, fontweight='bold')
    ply.xlabel('Epsilon')
    ply.ylabel('Rolling Mean Reward')
    #ply.ylim([0, 1])
    ply.xlim([1, 0])
    fig1 = ply.gcf()
    #ply.show()
    ply.draw()
    fig1.savefig('figs/Epsilon_' + descrip + '.png', bbox_inches='tight', dpi=200)
    ply.close()
    sucess,steps = run_episodes(env,optimalPolicy,descrip+': QLearning Optimal Policy')

def displayResults(env,optimalPolicy):
    ## Display results
    env.reset()
    for episode in range(5):
        state = env.reset()
        step = 0
        done = False
        print("****************************************************")
        print("EPISODE ", episode)

        for step in range(env.spec.max_episode_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            new_state, reward, done, _ = env.step(optimalPolicy[state])
            if done:
                env.render()         # print the last state
                print("Number of steps", step)  # number of steps it took.
                break
            state = new_state
    env.close()


if __name__ == '__main__':
     print("------------------------------------running VI Taxi")
     value_iteration(taxi,'Taxi')

     print("------------------------------------running PI Taxi")
     policy_iteration(taxi,'Taxi')

     print("------------------------------------running QL Taxi")
     QLearningRL(taxi,'Taxi')

