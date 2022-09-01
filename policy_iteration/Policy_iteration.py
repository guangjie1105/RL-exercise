from IPython.core.debugger import set_trace
import numpy as np
import pprint
import sys
if "/home/uif15692/RL_exercise/libRL/envs" not in sys.path:
  sys.path.append("/home/uif15692/RL_exercise/")
sys.path.append("/home/uif15692/RL_exercise/")      ## add the path that contains Lib not the path of Lib !!!!!!! 
                                                    ## And check if the pck is with __init__.py,if it is empty add manully by sys.path.append
from Lib.envs.gridworld import GridworldEnv   ##gym version 0.21.0

from policy_evaluation.Policy_evaluation import policy_eval

"""
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    def cal_Q(state,V):
        """Objective:
        calcule Q(s,a) by bellman equation for a specific state each action
        Args:
        state:the state need to be estimated
        V:state value funtion vector with env.nS length
        return:
        A vector with length env.nA , the state-action value funtion for each action in specified state
    """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob,next_state,reward,done in env.P[state][a]:   #prob=P(s'|s,a)
                A[a] += prob*(reward+discount_factor*V[next_state])
        return A
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        #Evaluate current policy
        V = policy_eval_fn(policy,env,discount_factor)
        policy_stable = np.full((env.nS,), True)    # boolean vector with length env.nS to check if policy is stable for all state
        for s in range(env.nS):
            #chose the action that has max prob in pi(s)
            stable =  False
            chosen_a = np.argmax(policy[s])
            #chose the action that max Q value
            best_a = np.argmax(cal_Q(s,V))
            if chosen_a != best_a:
                stable = False
            else:
                stable = True
            policy_stable[s] = stable
            policy[s] =  np.eye(env.nA)[best_a]   # greedy policy
        print(policy)
        if policy_stable.all(): # end condition all state is true
            return policy,V
            break


policy, V = policy_improvement(env)
print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(V)
print("")

print("Reshaped Grid Value Function:")
print(V.reshape(env.shape))
print("")





# Test the value function
""" expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(V, expected_v, decimal=2) """
