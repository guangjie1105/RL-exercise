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

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn    

def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities
    """
    
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))    #Ini Q dic:key is the state value the Q value for each action
    policy =defaultdict(lambda: np.zeros(env.action_space.n))  #Ini policy dic: key is the state value is np vector each state prob
    """ 
    Syntax: defaultdict(default_factory)
    Parameters:  

        default_factory: A function returning the default value for the dictionary defined. The key does not exist trun the value of default_factory.
    Advantage:Can add value with key,when is missing key before"""
    
    """lambda: Syntaxlambda arguments : expression
    in this case no matter what is given return is np.zeros(env.action_space.n)    """


    # The policy we're following
    #policy = make_epsilon_greedy_policy(Q, epsilon , env.action_space.n)
    
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        epsilon = 1/i_episode          ###update to GELI epsilon-greedy
        Policy = make_epsilon_greedy_policy(Q, epsilon ,env.action_space.n)
        ###### Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        for t in range(100):
            probs = Policy(state)
            policy[state] = probs
            action = np.random.choice(np.arange(len(probs)), p=probs)   #sample action with respect to probs
            next_state, reward, done, _ = env.step(action)   #sample
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # Find all (state, action) pairs we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        sa_in_episode = set((x[0], x[1]) for x in episode)
        print(sa_in_episode)
        ######Predection
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            # Find the first occurance of the (state, action) pair in the episode
            first_occurence_idx = next(i for i,x in enumerate(episode)
                                       if x[0] == state and x[1] == action)
            # Sum up all rewards since the first occurance
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])]) #No matter for Q or V,return is afterwards reward
            # Calculate average return for this state over all sampled episodes
            #returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            #Q[state][action] =  returns_sum[sa_pair]/ returns_count[sa_pair]
            Q[state][action] = Q[state][action]+(G-Q[state][action]) / returns_count[sa_pair]   ##Incremental update    
        
        # The policy is improved implicitly by changing the Q dictionary
    
    return Q, policy
Q, policy = mc_control_epsilon_greedy(env, num_episodes=40000)


# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")
