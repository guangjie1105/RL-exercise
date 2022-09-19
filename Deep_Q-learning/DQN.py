#####1.Expercience replay
#####2.Fixed network


##Run avec vitural python 3.6env
#from msilib.schema import Class
from gettext import translation
from re import A
from xml.etree.ElementTree import QName
import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf   #with python lower than 3.7 and tf V1


if "/home/uif15692/RL_exercise/libRL/envs" not in sys.path:
  sys.path.append("/home/uif15692/RL_exercise/")
sys.path.append("/home/uif15692/RL_exercise/") 

from Lib import plotting
from collections import deque, namedtuple

env = gym.envs.make("Breakout-v0")
VALID_ACTIONS = [0, 1, 2, 3]
# Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions
class StateProcessor():
    """
    Processes a raw Atari images. Resizes it and converts it to grayscale.
    """
    def __init__(self):
        # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)  #[210,160,3]
            self.output = tf.image.rgb_to_grayscale(self.input_state) #210*160*1
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)     #top-letf corner ,H,W
            self.output = tf.image.resize_images(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)#84*84*1
            self.output = tf.squeeze(self.output)#84*84

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State

        Returns:
            A processed [84, 84] state representing grayscale values.
        """
        return sess.run(self.output, { self.input_state: state })


class Estimator():
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
        with tf.variable_scope(scope):
            
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)
            
    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """

        # Placeholders for our input
        # Our input are 4 grayscale frames of shape 84, 84 each
        #self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X") #batch of image
        # The TD target value
        #self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")  #for each action
        # Integer id of which action was selected
        #self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.X_pl) / 255.0  #normalization  None*84*84*4
        batch_size = tf.shape(self.X_pl)[0] 

        # Three convolutional layers
        conv1 = tf.contrib.layers.conv2d(
            X, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl#First start + action number
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices) #Gather the Q faltten by index

        # Calculate the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)  #custom loss
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])

    def predict(self, sess, s):
        """
        Predicts action values.

        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 4, 84, 84, 1]

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated 
          action values.
        """
        return sess.run(self.predictions, { self.X_pl: s })

    def update(self, sess, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 4, 84, 84, 1]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss




# For Testing....

tf.reset_default_graph()
global_step = tf.Variable(0, name="global_step", trainable=False)

e = Estimator(scope="test")
sp = StateProcessor()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # Example observation batch
    observation = env.reset()
    
    observation_p = sp.process(sess, observation)  #84*84
    observation = np.stack([observation_p] * 4, axis=2)   #add along axis 2 84*84*4
    observations = np.array([observation] * 2)    # outside value*2,inside first dimension*2
    
    # Test Prediction
    print(e.predict(sess, observations))

    # Test training step
    y = np.array([10.0, 10.0])
    a = np.array([1, 3])
    print(e.update(sess, observations, a, y))



class ModelParamettersCopier():
    """Copy model para from train network to target network"""

    def __init__(self,estimaor1,estimator2):
        self.update_ops = []
        e1_params =[t for t in tf.trainable_variables() if t.name.startwith(estimaor1.scope)]
        e2_params = [t for t in tf.trainable_variables() if t.name.startwith(estimator2.scope)]
        e1_params = sorted(e1_params,key=lambda v:v.name)  #sort by name 
        e2_params = sorted(e2_params,key = lambda v : v.name)

        for e1,e2 in zip(e1_params,e2_params):    #zip:make tuple for each element in two groups
            op = e2.assign(e1)            #return array
            self.update_ops.append(op)


    def make(sess,self):
        sess.run(self.update_ops)


def make_epsilon_greedy_policy(estimator, nA):
    """argmax for the estimator prediction for the state
    A:nA array:epsilon/nA prob random action ,epsilon/nA+(1-eplision) prob greedy action"""

    def policy_fn(sess,observation,epsilon):     # ? observation 84*84*4 ?
        
        ''' return action prob in vector for observation '''
        
        A = np.ones(nA,dtype=float)*epsilon/nA
        q_value = estimator.predict(sess,np.expand_dims(observation,0))[0] #self.X_pl need None*84*84*4
        greedy_action_index = np.argmax(q_value)
        A[greedy_action_index] += (1-epsilon)
        return A
    return policy_fn




def q_learning(sess,env,epsilon_start,epsilon_end,epsilon_decay_steps,reply_memory_seize,make_epsilon_gredy_policy\
    ,q_estimator,target_estimator,StateProcessor,num_episodes,update_step,batch_size,discount_factor):
    total_t = sess.run(tf.contrib.framework.get_global_step())
    reply_memory = []
    epsilons = np.linspace(epsilon_start,epsilon_end,epsilon_decay_steps)
    policy = make_epsilon_gredy_policy(q_estimator,len(VALID_ACTIONS))
    state = env.reset()
    state = StateProcessor.process(sess,state) #84 *84
    state = np.stack([state]*4,axis = 2) #84*84*4
    Transition = namedtuple('transition',['state','action','next_state','reward','done'])

    ##### store experience with reply_memory_size
    for i in range(reply_memory_seize):
        action_probs = policy(sess,state,epsilons[min(total_t,epsilon-1)])
        action_index = np.random.choice(np.arange(len(VALID_ACTIONS)),action_probs)
        next_state,reward,done,_ = env.step(VALID_ACTIONS[action_index])
        next_state = StateProcessor.process(sess,next_state)
        next_state = np.append(state[:,:,1:],np.expend_dims(next_state,2),axis = 2)  
        #84*84*4 ,8*84*1 ,axis = 2--->the shape excluding specific axis should be same
        reply_memory.append()
        if done:
            state = env.reset()
            state = StateProcessor.process(sess,state)
            state = np.stack([state]*4, axis=2)
        else:
            state = next_state

    estimator_copy = ModelParamettersCopier(q_estimator, target_estimator)
    ##### Main traing
    for i_episode in range(num_episodes):
        ##Experience replay+robot running
        state = env.reset()
        state = StateProcessor.process(sess,state)
        state = np.stack([state]*4,axis=2)
        for t in itertools.count():
            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]
            ##update Q target
            if total_t%update_step==0:
                estimator_copy.make(sess)
                print("update fixed Q target")
            ##update replay_memory
            action_probs = policy(sess,state,epsilon)
            action  = np.random.choice(np.arrange(len(VALID_ACTIONS)),action_probs)
            next_state,reward,done,_ = env.step(VALID_ACTIONS[action])
            next_state = StateProcessor.process(sess,next_state)
            next_state = np.append(state[:,:,1:],np.expend_dims(next_state,2),axis=2)
            update_index = t%reply_memory_seize
            transition =Transition(state, action,next_state,reward,done) 
            reply_memory[update_index]=transition
            ##Feed with batch size data from replay_memory
            samples = random.sample(reply_memory,batch_size)
            state_batch,action_batch,next_state_batch,reward_batch,done_batch = map(np.array,zip(*samples))
            #unpack
            ###Target
            q_valuse_next = target_estimator.predict(sess,next_state_batch)
            target_batch = reward_batch + discount_factor*np.invert(done_batch).astype(np.float32)*np.argmax(q_valuse_next)

            
            ###SGD
            loss = q_estimator.update(sess,state_batch,action_batch,target_batch)

            if done:
                break
            state = next_state
            total_t+=1



        









    


        

        
