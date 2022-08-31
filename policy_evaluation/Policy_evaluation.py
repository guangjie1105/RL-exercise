from IPython.core.debugger import set_trace
import numpy as np
import pprint
import sys
if "/home/uif15692/RL_exercise/libRL/envs" not in sys.path:
  sys.path.append("/home/uif15692/RL_exercise/")
sys.path.append("/home/uif15692/RL_exercise/")      ## add the path that contains Lib not the paht of Lib !!!!!!! 
                                                    ## And check if the pck is with __init__.py,if it is empty add manully by sys.path.append
from Lib.envs.gridworld import GridworldEnv   ##gym version 0.21.0


pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).P(s|a)
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta & stop condition: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    N_Converge = True
    while N_Converge :
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for  prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value. Ref: Sutton book eq. 4.6.
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            N_Converge = False
    return V.reshape(env.shape)



random_policy = np.ones([env.nS, env.nA]) / env.nA   #1/4 for each action 
v = policy_eval(random_policy, env)



