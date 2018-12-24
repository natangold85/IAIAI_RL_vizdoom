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
class A2C_PARAMS(ParamsBase):
    def __init__(self, frameSize, gameVarsSize, numActions, discountFactor=0.99, batchSize=64, maxReplaySize = 10000, minReplaySize=64, 
                learning_rate=0.0001, numTrials2CmpResults=1000, outputGraph=True, accumulateHistory=True, numTrials2Learn=None, 
                numTrials2Save=100, layersNum=1, neuronsInLayerNum=256, withConvolution=True, includeEmbedding=False):

        super(A2C_PARAMS, self).__init__(frameSize=frameSize, gameVarsSize=gameVarsSize, numActions=numActions, discountFactor=discountFactor, numTrials2Learn=numTrials2Learn,
                                        numTrials2Save=numTrials2Save, maxReplaySize=maxReplaySize, minReplaySize=minReplaySize, accumulateHistory=accumulateHistory, 
                                        withConvolution=withConvolution, includeEmbedding=includeEmbedding)
        

        self.learning_rate = learning_rate
        self.batchSize = batchSize
        self.epochSize = maxReplaySize
        self.type = "A2C"
        self.numTrials2CmpResults = numTrials2CmpResults
        self.outputGraph = outputGraph
        self.layersNum = layersNum
        self.neuronsInLayerNum = neuronsInLayerNum
        self.normalizeState = True
        self.normalizeRewards = True


class A2C:
    def __init__(self, modelParams, nnName, nnDirectory, isMultiThreaded = False, agentName = ""):
        self.params = modelParams
        self.summaryDirectoryName = "./logs" + nnDirectory.replace(".", "") + "/"

        self.directoryName = nnDirectory + nnName
        self.agentName = agentName


        with tf.variable_scope("meta_data"):
            self.numRuns =tf.get_variable("numRuns", shape=(), initializer=tf.zeros_initializer(), dtype=tf.int32)

        with tf.variable_scope("critic"):
            self.critic = AC_Critic(modelParams)
        
        with tf.variable_scope("actor"):
            self.actor = AC_Actor(modelParams)

        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        self.summaries = [s for s in summary_ops if self.directoryName in s.name]


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
        
        if numRuns2Save == None:
            self.saver.save(self.sess, self.directoryName)
            numRuns2Save = self.NumRuns()
        else:
            currNumRuns = self.NumRuns()
            assign4Save = self.numRuns.assign(numRuns2Save)
            self.sess.run(assign4Save)

            self.saver.save(self.sess, self.directoryName)

            assign4Repeat2Train = self.numRuns.assign(currNumRuns)
            self.sess.run(assign4Repeat2Train)

        if toPrint:
            print("\n\t", threading.current_thread().getName(), " : ", self.agentName, "->save dqn with", numRuns2Save, "runs to:", self.directoryName)

    def choose_action(self, state, validActions, targetValues=False):
        actionProbs = self.ActionsValues(state, validActions)        
        action = np.random.choice(np.arange(len(actionProbs)), p=actionProbs)

        return action

    def State2Layers(self, s, size=1):
        sFrame = s[0].reshape(size, self.params.frameSize[0], self.params.frameSize[1], 1)
        sVars = s[1].reshape(size, self.params.gameVarsSize)
        return sFrame, sVars
    
    def State2LayersByIdx(self, s, idx):
        size = len(idx)
        sFrame = s[0][idx].reshape(size, self.params.frameSize[0], self.params.frameSize[1], 1)
        sVars = s[1][idx].reshape(size, self.params.gameVarsSize)
        return sFrame, sVars

    def DisperseNonValidValues(self, values, validActions):
        # clean non-valid actions from prob
        validValues = np.zeros(len(values), float)

        #take only valid values
        validValues[validActions] = values[validActions]
        # return values normalize to 1 
        return validValues / validValues.sum()


    def ActionsValues(self, state, validActions, targetVals=False):
        sFrame, sVars = self.State2Layers(state)
        actionProbs = self.actor.ActionsValues(sFrame, sVars, self.sess) 
        return self.DisperseNonValidValues(actionProbs, validActions)
    
    def learn(self, s, a, r, s_, terminal, numRuns2Save = None):  
        size = len(a)

        sFrame, sVars = self.State2Layers(s, size)
        s_Frame, s_Vars = self.State2Layers(s_, size)

        rNextState = self.critic.StatesValue(s_Frame, s_Vars, self.sess)
        
        # calculating critic value(s, s_)
        criticTargets = r + self.params.discountFactor * np.invert(terminal) * rNextState
        # estimating advantage function
        actorTarget = criticTargets - self.critic.StatesValue(sFrame, sVars, self.sess)

        for i in range(int(size  / self.params.batchSize)):
            chosen = np.arange(i * self.params.batchSize, (i + 1) * self.params.batchSize)

            feedDict = {self.actor.statesFrame: sFrame[chosen], self.actor.statesVars: sVars[chosen],
                        self.critic.statesFrame: sFrame[chosen], self.critic.statesVars: sVars[chosen],
                        self.actor.actions: a[chosen],
                        self.actor.targets: actorTarget[chosen],
                        self.critic.targets: criticTargets[chosen]}

            self.sess.run([self.actor.loss, self.critic.loss, self.actor.train_op, self.critic.train_op], feedDict)
    
    
    def DecisionMakerType(self):
        return "A2C"

    def TakeDfltValues(self):
        return False
    
    def end_run(self, reward):
        assign = self.numRuns.assign_add(1)
        self.sess.run(assign)

    def Reset(self):
        self.sess.run(self.init_op) 

    def DiscountFactor(self):
        return self.params.discountFactor

class AC_Actor:
    def __init__(self, params):
        # Network Parameters
        self.params = params
    

        self.statesFrame = tf.placeholder(tf.float32, shape=[None] + list(params.frameSize) + [1], name="statesFrame")
        
        if self.params.gameVarsSize > 0:
            self.statesVars = tf.placeholder(tf.float32, shape=[None, params.gameVarsSize], name="statesVars")

        # The TD target value
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="advantage")
        # Integer id of which action was selected
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        with tf.variable_scope("NeuralNet"):
            self.actionProb = self.create_actor_nn()

        # We add entropy to the loss to encourage exploration
        with tf.variable_scope("entropy"):
            self.entropy = -1 * tf.reduce_sum(self.actionProb * tf.log(self.actionProb), 1, name="entropy")
            self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")

        with tf.variable_scope("picked_action_prob"):
            batch_size = tf.shape(self.actions)[0]
            gather_indices = tf.range(batch_size) * tf.shape(self.actionProb)[1] + self.actions
            self.picked_action_probs = tf.gather(tf.reshape(self.actionProb, [-1]), gather_indices)
        
        
        self.losses = - ( tf.log(self.picked_action_probs) * self.targets + 0.01 * self.entropy)

        self.loss = tf.reduce_sum(self.losses, name="loss")
        self.optimizer = tf.train.RMSPropOptimizer(self.params.learning_rate, 0.99, 0.0, 1e-6)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]

        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)


        tf.summary.scalar(self.loss.op.name, self.loss)
        tf.summary.scalar(self.entropy_mean.op.name, self.entropy_mean)
        tf.summary.histogram(self.entropy.op.name, self.entropy)


    # Define the neural network
    def create_actor_nn(self, numLayers=2, numNeuronsInLayer=256):
        
        if self.params.withConvolution:
            # Add 2 convolutional layers with ReLu activation
            conv1 = tf.contrib.layers.convolution2d(self.statesFrame, num_outputs=8, kernel_size=[6, 6], stride=[3, 3],
                                                    activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                    biases_initializer=tf.constant_initializer(0.1))
            
            conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=8, kernel_size=[3, 3], stride=[2, 2],
                                                    activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                    biases_initializer=tf.constant_initializer(0.1))
            
            currInput = tf.contrib.layers.flatten(conv2)
        else:
            currInput = self.statesFrame

        for _ in range(self.params.layersNum):
            fc = tf.contrib.layers.fully_connected(currInput, self.params.neuronsInLayerNum)
            currInput = fc   

        if self.params.gameVarsSize > 0 and self.params.includeEmbedding:
            currInput = tf.concat([currInput, self.statesVars], axis = 1)

        output = tf.contrib.layers.fully_connected(currInput, self.params.numActions, activation_fn=None)
        softmax = tf.nn.softmax(output) + 1e-8

        return softmax

    def ActionsValues(self, sFrame, sVars, sess):
        probs = self.actionProb.eval({ self.statesFrame: sFrame, self.statesVars: sVars }, session=sess)        
        return probs[0]


class AC_Critic:
    def __init__(self, params):
        # Network Parameters
        self.params = params
             
        self.statesFrame = tf.placeholder(tf.float32, shape=[None] + list(params.frameSize) + [1], name="stateFrame") 
        self.statesVars = tf.placeholder(tf.float32, shape=[None, params.gameVarsSize], name="statesVars")

        # The TD target value
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="stateValues")

        with tf.variable_scope("NeuralNet"):
            self.output = self.create_critic_nn()

        self.losses = tf.squared_difference(self.output, self.targets)
        self.loss = tf.reduce_sum(self.losses, name="loss")
        
        self.optimizer = tf.train.RMSPropOptimizer(self.params.learning_rate, 0.99, 0.0, 1e-6)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)

    # Define the neural network
    def create_critic_nn(self, numLayers=2, numNeuronsInLayer=256):
        
        if self.params.withConvolution:
            # Add 2 convolutional layers with ReLu activation
            conv1 = tf.contrib.layers.convolution2d(self.statesFrame, num_outputs=8, kernel_size=[6, 6], stride=[3, 3],
                                                    activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                    biases_initializer=tf.constant_initializer(0.1))
            
            conv2 = tf.contrib.layers.convolution2d(conv1, num_outputs=8, kernel_size=[3, 3], stride=[2, 2],
                                                    activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                                    biases_initializer=tf.constant_initializer(0.1))
            
            currInput = tf.contrib.layers.flatten(conv2)
        else:
            currInput = self.statesFrame
            
        for _ in range(self.params.layersNum):
            fc = tf.contrib.layers.fully_connected(currInput, self.params.neuronsInLayerNum)
            currInput = fc   

        if self.params.gameVarsSize > 0 and self.params.includeEmbedding:
            currInput = tf.concat([currInput, self.statesVars], axis = 1)

        output = tf.contrib.layers.fully_connected(currInput, 1, activation_fn=None)
        output = tf.squeeze(output, squeeze_dims=[1], name="logits")

        return output

    
    def StatesValue(self, sFrames, sVars, sess):
        return self.output.eval({self.statesFrame: sFrames, self.statesVars: sVars}, session=sess)

