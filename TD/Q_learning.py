##Compare with sara(0),the only difference the greedy policy for target

from cmath import pi
from email import policy
import itertools
from os import environ, stat
from turtle import done
import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict



if "/home/uif15692/RL_exercise/libRL/envs" not in sys.path:
  sys.path.append("/home/uif15692/RL_exercise/")
sys.path.append("/home/uif15692/RL_exercise/") 
from Lib.envs.cliff_walking import CliffWalkingEnv
from Lib import plotting

matplotlib.style.use('ggplot')


env = CliffWalkingEnv()


def make_epsilon_greedy_policy(Q,epsilon,nA):
        """
    Creates an GELIE epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 1,also decay with iteration
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """
        def policy_fn(state):
            A = np.ones(nA,dtype =float)*epsilon/nA   ##random
            best_action = np.argmax(Q[state])     ##greedy
            A[best_action] +=( 1-epsilon)   #   += the random action can be greedy action
            return A
        return policy_fn


def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    #policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    policy = defaultdict(lambda:np.zeros(env.action_space.n))

    for i_episode in range(1,num_episodes+1):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        epsilon = 1/i_episode   #decay
        Policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
        # Implement this!
        state = env.reset()   #Ini state
        """ probs = Policy(state)
        action = np.random.choice(np.arange(env.action_space.n),p = probs)  # sample action with respect to policy """
        for t in itertools.count():     #each time step
            probs = Policy(state)
            action = np.random.choice(np.arange(env.action_space.n),p = probs)  #mui epsilo policy for environment
                                                                                #move inside the loop update by itself
            next_state,reward,done,_ = env.step(action)
            next_action_probs = Policy(state)
            best_next_action = np.argmax(Q[next_state]) #sample next action
            policy[state] = next_action_probs  #update policy
            # Update statistics,does not matter
            stats.episode_rewards[i_episode-1] += reward
            stats.episode_lengths[i_episode-1] = t
            #Update process
            td_target = reward + discount_factor * Q[next_state][best_next_action]  # pi greedy policy for target
            td_error = td_target - Q[state][action]
            Q[state][action]+= alpha * td_error
            if done:
                break
            state = next_state
    return Q,stats




Q,stats = sarsa(env, 200)



plotting.plot_episode_stats(stats)




