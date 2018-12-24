import numpy as np
import pandas as pd
import random
import pickle
import os.path
import threading

import tensorflow as tf
import os

from utils import ParamsBase

from multiprocessing import Lock
from utils import EmptyLock


# dqn params
class ORIG_PARAMS(ParamsBase):
    def __init__(self, frameSize, gameVarsSize, numActions, discountFactor=0.99, batchSize = 64, maxReplaySize = 10000, minReplaySize = 64, 
                learning_rate=0.00025, numTrials2CmpResults=1000, outputGraph=True, accumulateHistory=True, numTrials2Learn=None, numTrials2Save=100):

        super(ORIG_PARAMS, self).__init__(frameSize=frameSize, gameVarsSize=gameVarsSize,numActions=numActions, discountFactor=discountFactor, numTrials2Learn=numTrials2Learn,
                                        numTrials2Save=numTrials2Save, maxReplaySize=maxReplaySize, minReplaySize=minReplaySize, accumulateHistory=accumulateHistory)
        

        self.learning_rate = learning_rate
        self.batchSize = batchSize
        self.epochSize = batchSize

        self.type = "orig"
        self.numTrials2CmpResults = numTrials2CmpResults
        self.outputGraph = outputGraph
        self.normalizeState = False

class ORIG_MODEL:
    def __init__(self, modelParams, nnName, nnDirectory, isMultiThreaded = False, agentName = ""):
        # Create the input variables
        self.params = modelParams

        self.summaryDirectoryName = "./logs" + nnDirectory.replace(".", "") + "/"
        self.directoryName = nnDirectory + nnName
        self.agentName = agentName

        self.states = tf.placeholder(tf.float32, [None] + list(modelParams.stateSize) + [1], name="State")
        self.actions = tf.placeholder(tf.int32, [None], name="Action")
        self.target_q = tf.placeholder(tf.float32, [None, modelParams.numActions], name="TargetQ")

        self.numRuns =tf.get_variable("numRuns", shape=(), initializer=tf.zeros_initializer(), dtype=tf.int32)

        # Add 2 convolutional layers with ReLu activation
        conv1 = tf.contrib.layers.convolution2d(self.states, num_outputs=8, kernel_size=[6, 6], stride=[3, 3],
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1))
        conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=8, kernel_size=[3, 3], stride=[2, 2],
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                biases_initializer=tf.constant_initializer(0.1))
        conv2_flat = tf.contrib.layers.flatten(conv2)
        fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=128, activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                biases_initializer=tf.constant_initializer(0.1))

        
        self.output = tf.contrib.layers.fully_connected(fc1, num_outputs=modelParams.numActions, activation_fn=None,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.1))
        self.bestAction = tf.argmax(self.output, 1)

        self.loss = tf.losses.mean_squared_error(self.output, self.target_q)

        optimizer = tf.train.RMSPropOptimizer(modelParams.learning_rate)
        # Update the parameters according to the computed gradient using RMSProp.
        self.train_step = optimizer.minimize(self.loss)

    def InitModel(self, session, resetModel=False):
        self.sess = session

        if self.params.outputGraph:
            # $ tensorboard --logdir=logs/directory
            tf.summary.FileWriter(self.summaryDirectoryName, self.sess.graph)

        self.init_op = tf.global_variables_initializer()
        self.sess.run(self.init_op)  

        self.saver = tf.train.Saver()
        fnameNNMeta = self.directoryName + ".meta"
        if os.path.isfile(fnameNNMeta) and not resetModel:
            self.saver.restore(self.sess, self.directoryName)
            loadedDM = True
        else:
            self.Save()
            loadedDM = False
        
        return loadedDM

    def NumRuns(self):
        return self.numRuns.eval(session = self.sess)

    def Save(self, numRuns2Save = None, toPrint = True):
        self.saver.save(self.sess, self.directoryName)
        numRuns2Save = self.NumRuns()

        if toPrint:
            print("\n\t", threading.current_thread().getName(), " : ", self.agentName, "->save dqn with", numRuns2Save, "runs to:", self.directoryName)


    def StatesValue(self, states, size):
        return self.sess.run(self.output, feed_dict={self.states: states.reshape(size, self.params.stateSize[0], self.params.stateSize[1], 1)})

    def choose_action(self, state, validActions, targetValues=False):
        values = self.sess.run(self.output, feed_dict={self.states: state.reshape(1, self.params.stateSize[0], self.params.stateSize[1], 1)})[0]
        
        filteredActions = values.copy()
        filteredActions.fill(np.nan)
        
        filteredActions[validActions] = values[validActions]
        
        action = np.nanargmax(filteredActions)

        return action




    def learn(self, s, a, r, s_, terminal, numToLearn=None):
        """ Learns from a single transition (making use of replay memory).
        s2 is ignored if s2_isterminal """

        # Get a random minibatch from the replay memory and learns from it.
        size = len(a)

        qNext = np.max(self.StatesValue(s_, size), axis=1)
        target_q = self.StatesValue(s, size)
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if not isterminal else r
        target_q[np.arange(target_q.shape[0]), a] = r + self.params.discountFactor * (1 - terminal) * qNext
        feed_dict = {self.states: s.reshape(size, self.params.stateSize[0], self.params.stateSize[1], 1), 
                    self.target_q: target_q}
        l, _ = self.sess.run([self.loss, self.train_step], feed_dict=feed_dict)

    def DecisionMakerType(self):
        return "orig"

    def TakeDfltValues(self):
        return False
    
    def end_run(self, reward):
        assign = self.numRuns.assign_add(1)
        self.sess.run(assign)
        self.Save(toPrint=False)

    def Reset(self):
        self.sess.run(self.init_op) 

    def DiscountFactor(self):
        return self.params.discountFactor
