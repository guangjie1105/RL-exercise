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

## the main idea :use bellman optimality equation pick the max Q as V 

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.
    
    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment. 
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.        
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
    stable_table = np.full((env.nS),True)  
    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])
    delta = 0
    # For value iteration
    while True:
        for s in range(env.nS):
            Q = cal_Q(s,V)
            print(Q)
            v = np.max(Q)
            print(v)
            delta = np.abs(v-V[s])
            if delta <theta:
                stable_table[s] = True
            else : 
                stable_table[s] = False
            V[s] = v
        if stable_table.all():         #Check for all states
            break
    #Greedy policy pick max Q
    for s in range(env.nS):
        a = np.argmax(cal_Q)
        policy [s] = np.eye(env.nA)[a]
    return policy, V