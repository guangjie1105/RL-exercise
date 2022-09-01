from cmath import pi
import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict

if "/home/uif15692/RL_exercise/libRL/envs" not in sys.path:
  sys.path.append("/home/uif15692/RL_exercise/")
sys.path.append("/home/uif15692/RL_exercise/") 
from Lib.envs.blackjack import BlackjackEnv
from Lib import plotting
matplotlib.style.use('ggplot')
env = BlackjackEnv()

def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final value function
    V = defaultdict(float)

    print('default V',V)

        
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        ### Generate an episode by pi.
        # An episode is an array of (state, action, reward) tuples , state is tuple 2. 
        episode = []
        state = env.reset()
        for t in range(100):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)      
            episode.append((state, action, reward))    #append as tuple
            if done:
                break
            state = next_state

        ### Extract stae from episode
        # We convert each state to a tuple so that we can use it as a dict key
        print('episode',episode)
        states_in_episode = set([x[0] for x in episode])    #Take the first element of ith element in episode,compose a set.
        print('states_in_episode',states_in_episode)
        
        ### Update valuse function for each state
        for state in states_in_episode:
            # Find the first occurance of the state in the episode
            first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == state)   #Because use reward afterwards so need to find where this state starts
                                       #need index so make it as an enumerate
                                       #next(iter，stopdef),need an iterator  like lier(list),this this case i is from for loop ,so it is iterator
                                       #return i for i in enumerate(episode) if x[0] == state
            """for i,x in enumerate(episode):
                if x[0] == state:
                    first_occurence_idx = i"""
                 
            print('first_occurence_idx',first_occurence_idx)
            # Sum up all rewards since the first occurance
            G = sum(x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:]))  #Accumulated r from first_occurence_idx to end
            # Calculate average return for this state over all sampled episodes
            returns_sum[state] += G
            returns_count[state] += 1.0
            V[state] = returns_sum[state] / returns_count[state]

    return V    




def sample_policy(observation):
    """
    A policy that sticks if the player score is >= 20 and hits otherwise.
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1

V_10k = mc_prediction(sample_policy, env, num_episodes=10000)
plotting.plot_value_function(V_10k, title="10,000 Steps")

V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
plotting.plot_value_function(V_500k, title="500,000 Steps")