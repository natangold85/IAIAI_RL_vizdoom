import math
import numpy as np
import pandas as pd
import pickle
import os.path
import time
import datetime
import sys

import matplotlib.pyplot as plt

from multiprocessing import Lock

from utils import EmptyLock

from utils import ParamsBase

#qtable params
class QTableParams(ParamsBase):
    def __init__(self, stateSize, numActions, learning_rate=0.01, discountFactor=0.95, explorationProb=0.1, maxReplaySize = 500000, minReplaySize = 1000):
        super(QTableParams, self).__init__(stateSize, numActions, discountFactor, maxReplaySize, minReplaySize)    
        self.learningRate = learning_rate
        self.explorationProb = explorationProb        
    
    def ExploreProb(self, numRuns):
        return self.explorationProb

class QTableParamsExplorationDecay(ParamsBase):
    def __init__(self, stateSize, numActions, learning_rate=0.01, discountFactor=0.95, exploreRate = 0.001, exploreStop = 0.1, maxReplaySize = 50000, minReplaySize = 1000):
        super(QTableParamsExplorationDecay, self).__init__(stateSize, numActions, discountFactor, maxReplaySize, minReplaySize) 

        self.learningRate = learning_rate        
        self.exploreStart = 1
        self.exploreStop = exploreStop
        self.exploreRate = exploreRate

    def ExploreProb(self, numRuns):
        return self.exploreStop + (self.exploreStart - self.exploreStop) * np.exp(-self.exploreRate * numRuns)

class QLearningTable:

    def __init__(self, modelParams, qTableName, qTableDirectory, loadTable = True, isMultiThreaded = False):
        self.qTableFullName = qTableDirectory + qTableName
        
        if isMultiThreaded:
            self.checkStateLoc = Lock()
        else:
            self.checkStateLoc = EmptyLock()

        self.TrialsData = "TrialsData"
        self.NumRunsTotalSlot = 0
        self.NumRunsExperimentSlot = 1
        self.AvgRewardSlot = 2
        self.AvgRewardExperimentSlot = 3

        slotsInTable = max(4, modelParams.numActions)
        self.actions = list(range(modelParams.numActions))
        self.slots = list(range(slotsInTable))  # a list
        self.table = pd.DataFrame(columns=self.slots, dtype=np.float)
        if os.path.isfile(self.qTableFullName + '.gz') and loadTable:
            self.ReadTable()
        
        self.params = modelParams
        

        self.check_state_exist(self.TrialsData)

        self.numTotRuns = self.table.ix[self.TrialsData, self.NumRunsTotalSlot]
        self.avgTotReward = self.table.ix[self.TrialsData, self.AvgRewardSlot]
        self.numExpRuns = 0
        self.avgExpReward = 0

        self.table.ix[self.TrialsData, self.AvgRewardExperimentSlot] = 0
        self.table.ix[self.TrialsData, self.NumRunsExperimentSlot] = 0

    def InitTTable(self, ttable):
        self.ttable = ttable
        self.reverseTable = ttable.reverseKey
        self.normalTable = ttable.normalKey
        self.timeoutPropogation = 10

    def ReadTable(self):
        self.table = pd.read_pickle(self.qTableFullName + '.gz', compression='gzip')

    def SaveTable(self):
        self.table.to_pickle(self.qTableFullName + '.gz', 'gzip') 
    
    def choose_absolute_action(self, observation):
        state = str(observation)
        self.check_state_exist(state)
        state_action = self.table.ix[state, self.actions]
        
        state_actionReindex = state_action.reindex(np.random.permutation(state_action.index))
        action = state_actionReindex.idxmax()

        return action, state_action[action]

    def ExploreProb(self):
        return self.params.ExploreProb(self.numTotRuns)

    def choose_action(self, state, validActions, targetValues=False):
        state = str(state)
        exploreProb = self.params.ExploreProb(self.numTotRuns)

        if np.random.uniform() > exploreProb:
            # choose best action
            state_action = self.table.ix[state, self.actions]
            
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            
            # todo : not checked properly (should disregard non valid actions)
            maxArgs = list(np.argwhere(state_action == np.amax(state_action[validActions])))
            maxArgsValid = [x for x in maxArgs if x in validActions]    
            action = np.random.choice(maxArgsValid) 
        else:
            # choose random action
            action = np.random.choice(self.actions)
            
        return action


    def ActionsValues(self, state, validActions, targetValues = False):
        s = str(state)
        self.check_state_exist(s)
        state_action = self.table.ix[s, :]
        # todo: insert valid actions calculations
        vals = np.zeros(len(self.actions), dtype=float)
        for a in range(len(self.actions)):
            vals[a] = state_action[self.actions[a]]

        return vals

    def NumRuns(self):
        return self.numTotRuns

    def learn(self, statesVec, actionsVec, rewardsVec, nextStateVec, terminal):
        for i in range(len(rewardsVec)):
            s = str(statesVec[i])
            s_ = str(nextStateVec[i])
            self.check_state_exist(s)
            self.check_state_exist(s_)
            self.learnIMP(s, actionsVec[i], rewardsVec[i], s_, terminal[i])

    def learnIMP(self, s, a, r, s_, terminal):
        q_predict = self.table.ix[s, a]
        
        if not terminal:
            q_target = r + self.params.discountFactor * self.table.ix[s_, :].max()
        else:
            q_target = r  # next state is terminal
        
        # update
        self.table.ix[s, a] += self.params.learningRate * (q_target - q_predict)

    def end_run(self, r, saveTable = False):
        self.avgTotReward = (self.numTotRuns * self.avgTotReward + r) / (self.numTotRuns + 1)
        self.avgExpReward = (self.numExpRuns * self.avgExpReward + r) / (self.numExpRuns + 1)
        
        self.numTotRuns += 1
        self.numExpRuns += 1

        self.table.ix[self.TrialsData, self.AvgRewardSlot] = self.avgTotReward
        self.table.ix[self.TrialsData, self.AvgRewardExperimentSlot] = self.avgExpReward

        
        self.table.ix[self.TrialsData, self.NumRunsTotalSlot] = self.numTotRuns
        self.table.ix[self.TrialsData, self.NumRunsExperimentSlot] = self.numExpRuns

        # print("num total runs = ", self.numTotRuns, "avg total = ", self.avgTotReward)
        # print("num experiment runs = ", self.numExpRuns, "avg experiment = ", self.avgExpReward)

        if saveTable:
            self.SaveTable()

    def Reset(self):
        self.table = pd.DataFrame(columns=self.slots, dtype=np.float)
        self.check_state_exist(self.TrialsData)

        self.numTotRuns = self.table.ix[self.TrialsData, self.NumRunsTotalSlot]
        self.avgTotReward = self.table.ix[self.TrialsData, self.AvgRewardSlot]
        self.numExpRuns = 0
        self.avgExpReward = 0

        self.table.ix[self.TrialsData, self.AvgRewardExperimentSlot] = 0
        self.table.ix[self.TrialsData, self.NumRunsExperimentSlot] = 0

    def check_state_exist(self, state, stateToInitValues = None):
        self.checkStateLoc.acquire()
        newState = False
        if state not in self.table.index:
            # append new state to q table
            self.table = self.table.append(pd.Series([0] * len(self.slots), index=self.table.columns, name=state))
            
            if stateToInitValues in self.table.index:
                self.table.ix[state,:] = self.table.ix[stateToInitValues, :]
            newState = True

        self.checkStateLoc.release()
        return newState