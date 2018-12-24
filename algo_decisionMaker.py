import numpy as np
import pandas as pd
import pickle
import os.path
import datetime
import sys
import threading
import math

from time import time
import tensorflow as tf

from multiprocessing import Lock

from utils import EmptyLock

#decision makers
from algo_dqn import DQN
from algo_dqn import DQN_WithTarget

from algo_a2c import A2C
from algo_a3c import A3C

from algo_orig import ORIG_MODEL

from algo_qtable import QLearningTable

from utils_history import History

# model builders:
from utils_ttable import TransitionTable

# results handlers
from utils_results import ResultFile

def CreateDecisionMaker(agentName, configDict, isMultiThreaded, dmCopy, heuristicClass=None):
    from agentRunTypes import GetRunType    
    from agentStatesAndActions import NumActions2Agent
    from agentStatesAndActions import StatesParams2Agent

    dmCopy = "" if dmCopy==None else "_" + str(dmCopy)

    if configDict[agentName] == "none":
        return BaseDecisionMaker(agentName), []

    runType = GetRunType(agentName, configDict)

    directory = configDict["directory"] + "/" + agentName + "/" + runType["directory"] + dmCopy

    if configDict[agentName] == "heuristic":
        decisionMaker = heuristicClass(resultFName=runType["results"], directory=directory)
    else:        
        numActions = NumActions2Agent(agentName)
        stateParams = StatesParams2Agent(agentName)
        dmClass = eval(runType["dm_type"])

        decisionMaker = dmClass(modelType=runType["algo_type"], modelParams = runType["params"], decisionMakerName = runType["dm_name"], agentName=agentName,  
                            resultFileName=runType["results"], historyFileName=runType["history"], directory=directory, isMultiThreaded=isMultiThreaded)

    return decisionMaker, runType


class BaseDecisionMaker:
    def __init__(self, agentName):
        self.trainFlag = False
        self.switchFlag = False

        self.resultFile = None
        self.secondResultFile = None
        self.decisionMaker = None
        self.historyMngr = None

        self.subAgentsDecisionMakers = {}
        self.switchCount = {}
        self.agentName = agentName

        self.copyTargetLock = Lock()
        self.CopyTarget2ModelNumRuns = -1

    def SetSubAgentDecisionMaker(self, key, decisionMaker):
        self.subAgentsDecisionMakers[key] = decisionMaker
        self.switchCount[key] = 0

    def GetSubAgentDecisionMaker(self, key):
        if key in self.subAgentsDecisionMakers.keys():
            return self.subAgentsDecisionMakers[key]
        else:
            return None

    def GetDecisionMakerByName(self, name):
        if self.agentName == name:
            return self
        else:
            for saDM in self.subAgentsDecisionMakers.values():
                if saDM != None:
                    dm = saDM.GetDecisionMakerByName(name)
                    if dm != None:
                        return dm
                
            return None
            
    def TrainAll(self):
        numTrial2Learn = -1
    
        for subDM in self.subAgentsDecisionMakers.values():
            if subDM != None:
                numTrial2LearnSa = subDM.TrainAll()
                if numTrial2LearnSa >= 0:
                    numTrial2Learn = numTrial2LearnSa
        
        return numTrial2Learn
        
    def InitModel(self, sess, resetModel):
        if self.decisionMaker != None:
            self.decisionMaker.InitModel(sess, resetModel)

        if self.resultFile != None and not resetModel:
            self.resultFile.Load()

        if self.secondResultFile != None and not resetModel:
            self.secondResultFile.Load()

        if self.historyMngr != None and not resetModel:
            self.historyMngr.Load()

        for subAgentDm in self.subAgentsDecisionMakers.values():
            if subAgentDm != None:
                subAgentDm.InitModel(sess, resetModel)

    def AddSwitch(self, idx, numSwitch, name, resultFile):
        if resultFile != None:
            if idx not in self.switchCount:
                self.switchCount[idx] = 0

            if self.switchCount[idx] <= numSwitch:
                self.switchCount[idx] = numSwitch + 1
                slotName = name + "_" + str(numSwitch)
                resultFile.AddSlot(slotName)

    def AddSlot(self, slotName):
        if self.resultFile != None:
            self.resultFile.AddSlot(slotName)

    def GoToNextResultFile(self, numFile):
        self.resultFile.GoToNextFile(numFile)

    def AddResultFile(self, resultFile):
        self.secondResultFile = resultFile
        
    def GetResultFile(self):
        return self.secondResultFile

    def GetHistoryInstance(self):
        return self.historyMngr

    def Train(self):
        pass     

    def NumRuns(self):
        pass

    def choose_action(self, state, validActions, targetValues=False):
        pass

    def learn(self, s, a, r, s_, terminal = False):
        pass

    def ActionsValues(self,state, validActions, targetValues = False):
        pass

    def end_run(self, r, score = 0 ,steps = 0):
        pass

    def TrimHistory(self):
        pass
    
    def AddHistory(self):
        return None
    
    def DrawStateFromHist(self, realState=True):
        return None
    
    def TakeTargetDM(self, numRunsEnd):
        self.copyTargetLock.acquire()
        self.CopyTarget2ModelNumRuns = numRunsEnd    
        self.copyTargetLock.release()

    def CopyTarget2Model(self, numRuns):
        pass

    def GetMinReward(self):
        return 0.0
    
    def SetMinReward(self, r):
        pass

    def GetMaxReward(self):
        return 0.01
    
    def SetMaxReward(self, r):
        pass

    def DecisionMakerType(self):
        return "Base"

    def NumDfltRuns(self):
        return 0

class BaseNaiveDecisionMaker(BaseDecisionMaker):
    def __init__(self, numTrials2Save, agentName = "", resultFName = None, directory = None):
        super(BaseNaiveDecisionMaker, self).__init__(agentName)
        self.resultFName = resultFName
        self.trialNum = 0
        self.numTrials2Save = numTrials2Save

        if resultFName != None:
            self.lock = Lock()
            if directory != None:
                fullDirectoryName = "./" + directory +"/"
                if not os.path.isdir(fullDirectoryName):
                    os.makedirs(fullDirectoryName)
            else:
                fullDirectoryName = "./"

            self.resultFile = ResultFile(fullDirectoryName + resultFName, numTrials2Save, agentName)



    def end_run(self, r, score, steps):
        saveFile = False
        self.trialNum += 1
        
        print("\t", threading.current_thread().getName(), ":", self.agentName," #trials =", self.trialNum, "reward =", r)
        if self.resultFName != None:
            self.lock.acquire()
            if self.trialNum % self.numTrials2Save == 0:
                saveFile = True

            self.resultFile.end_run(r, score, steps, saveFile)
            self.lock.release()
       
        return saveFile 


class UserPlay(BaseDecisionMaker):
    def __init__(self, agentName = "", playWithInput = True, numActions = 1, actionDoNothing = 0):
        super(UserPlay, self).__init__(agentName)
        self.playWithInput = playWithInput
        self.numActions = numActions
        self.actionDoNothing = actionDoNothing

    def choose_action(self, state, validActions, targetValues=False):
        if self.playWithInput:
            a = input("insert action: ")
        else:
            a = self.actionDoNothing
        return int(a)

    def learn(self, s, a, r, s_, terminal = False):
        return None

    def ActionsValues(self,state, validActions, targetValues = False):
        return np.zeros(self.numActions,dtype = float)

    def end_run(self, r, score = 0 ,steps = 0):
        return False



class DecisionMakerAlgoBase(BaseDecisionMaker):
    def __init__(self, modelType, modelParams, agentName='', decisionMakerName='', resultFileName='', historyFileName='', directory='', isMultiThreaded=False):
        super(DecisionMakerAlgoBase, self).__init__(agentName)

        # params
        self.params = modelParams
        
        # count of trials that model is active and not learning to trim history
        self.nonTrainingHistCount = 0
        # trial to DQN to learn(insert to model as learn trial num)
        self.trial2LearnModel = -1
        # sync mechanism
        self.endRunLock = Lock() if isMultiThreaded else EmptyLock()
        
        self.printTrain = False

        # create directory        
        if directory != "":
            fullDirectoryName = "./" + directory +"/"
            if not os.path.isdir(fullDirectoryName):
                os.makedirs(fullDirectoryName)
        else:
            fullDirectoryName = "./"


        # create result file
        if resultFileName != '':
            self.resultFile = ResultFile(fullDirectoryName + resultFileName, modelParams.numTrials2Save, self.agentName)
        
        # create history mngr class
        self.historyMngr = History(modelParams, historyFileName, fullDirectoryName, isMultiThreaded)
    	
        self.startScope = fullDirectoryName
        # create decision maker class
        decisionClass = eval(modelType)

        with tf.variable_scope(self.startScope):
            self.decisionMaker = decisionClass(modelParams, decisionMakerName, fullDirectoryName, isMultiThreaded=isMultiThreaded, agentName=self.agentName)
        
    def choose_action(self, state, validActions, targetValues=False):
        return self.decisionMaker.choose_action(state, validActions, targetValues)     

    def NumRuns(self):
        return self.decisionMaker.NumRuns()

    def ResetHistory(self, dump2Old=True, save=False):
        self.historyMngr.Reset(dump2Old, save)

    def ResetAllData(self, resetDecisionMaker=True, resetHistory=True, resetResults=True):
        if resetDecisionMaker:
            self.decisionMaker.Reset()

        if resetHistory:
            self.historyMngr.Reset()

        if resetResults and self.resultFile != None:
            self.resultFile.Reset()


    def ActionsValues(self, state, validActions, targetValues = False):
        return self.decisionMaker.ActionsValues(state, validActions, targetValues)

    def DiscountFactor(self):
        return self.decisionMaker.DiscountFactor()

    def DrawStateFromHist(self, realState=True):
        return self.historyMngr.DrawState(realState)

    def GetMinReward(self):
        return self.historyMngr.GetMinReward()
    
    def SetMinReward(self, r):
        self.historyMngr.SetMinReward(r)

    def GetMaxReward(self):
        return self.historyMngr.GetMaxReward()
    
    def SetMaxReward(self, r):
        self.historyMngr.SetMaxReward(r)

    def DecisionMakerType(self):
        return self.decisionMaker.DecisionMakerType()

    def NumDfltRuns(self):
        return self.decisionMaker.NumDfltRuns()

    def DfltValueInitialized(self):
        return self.decisionMaker.DfltValueInitialized()

    def CheckModel(self, plotGraphs):
        pass

    def Save(self):
        self.decisionMaker.Save()
        self.historyMngr.Save()
        if self.resultFile != None:
            self.resultFile.Save()

    def end_test_run(self, r, score, steps):
        if self.resultFile != None:
            self.resultFile.end_run(r, score, steps, True)

    def NumTestRuns(self):
        numRuns = 0
        if self.resultFile != None:
            numRuns = self.resultFile.NumRuns()
        
        return numRuns
          
class DecisionMakerExperienceReplay(DecisionMakerAlgoBase):
    def __init__(self, modelType, modelParams, agentName='', decisionMakerName='', resultFileName='', historyFileName='', directory='', isMultiThreaded = False):
        super(DecisionMakerExperienceReplay, self).__init__( modelType=modelType, modelParams=modelParams, agentName=agentName, decisionMakerName=decisionMakerName, 
                                                            resultFileName=resultFileName, historyFileName=historyFileName, directory=directory, 
                                                            isMultiThreaded=isMultiThreaded)

    def end_run(self, r, score, steps):
        self.endRunLock.acquire()

        numRun = int(self.NumRuns())

        save = True if (numRun + 1) % self.params.numTrials2Save == 0 else False
        train = True if (numRun + 1) % self.params.numTrials2Learn == 0 else False

        if self.resultFile != None:
            self.resultFile.end_run(r, score, steps, save)

        if save:
            self.historyMngr.Save()

        if train:
            self.Train()
            self.decisionMaker.end_run(r)
            self.decisionMaker.Save()
        else:        
            self.decisionMaker.end_run(r)
            
        self.endRunLock.release()
        
        return save 

    def Train(self):

        start = time()
        if self.historyMngr.Size() > self.params.minReplaySize:
            sizeHist = min(self.params.epochSize, self.historyMngr.Size())
            s, a, r, s_, terminal = self.historyMngr.get_sample(sizeHist)

            self.decisionMaker.learn(s, a, r, s_, terminal)
           #if self.printTrain:
            print("\t", threading.current_thread().getName(), ":", self.agentName,"->learn on", sizeHist, "transitions, duration =", (time() - start) / 60.0)
            
            if self.params.type == "A2C":
                self.historyMngr.Reset()

        return (time() - start) / 60.0

    def ActionsValues(self, state, validActions, targetValues = False):
        return self.decisionMaker.ActionsValues(state, targetValues)


class DecisionMakerOnlineAsync(DecisionMakerAlgoBase):
    def __init__(self, modelType, modelParams, agentName='', decisionMakerName='', resultFileName='', historyFileName='', directory='', isMultiThreaded = False):        
        super(DecisionMakerOnlineAsync, self).__init__( modelType=modelType, modelParams=modelParams, agentName=agentName, decisionMakerName=decisionMakerName, 
                                                            resultFileName=resultFileName, historyFileName=historyFileName, directory=directory, 
                                                            isMultiThreaded=isMultiThreaded)

        self.agentsHistory = {}
        
        self.counter4HistFiles = 0
        self.histFileName = directory + "/histTest_"
        self.histFileLock = Lock()

    def AddHistory(self):
        idxWorker = len(self.agentsHistory)
        agentName = "GameThread_" + str(idxWorker)

        with tf.variable_scope(self.startScope):
            self.decisionMaker.AddWorker(agentName)

        hist = self.historyMngr.AddHistory()
        self.agentsHistory[agentName] = hist
       
        return hist

    def NumRunsAgent(self, agentName):
        return self.decisionMaker.NumRunsAgent(agentName)

    def NumRuns(self):
        runs = 0
        for name in self.agentsHistory.keys():
            runs += self.NumRunsAgent(name)

        return runs

    def end_run(self, r, score, steps):

        agentName = threading.current_thread().getName()
        numAgentRuns = self.NumRunsAgent(agentName)
        numTotRun = self.NumRuns()

        print(agentName, ":", self.agentName,"->for trial#", numTotRun, " agent trial#", numAgentRuns, ": reward =", r, "score =", score, "steps =", steps)

        self.decisionMaker.end_run(r)
        if (numAgentRuns + 1) % self.params.numTrials2Learn == 0:
            self.Train()            
             
        # for end run only shared files between agents are needed for lock
        self.endRunLock.acquire()

        save = True if (numTotRun + 1) % self.params.numTrials2Save == 0 else False
        if self.resultFile != None:
            self.resultFile.end_run(r, score, steps, save)
        if save:
            self.decisionMaker.Save()
            self.historyMngr.Save()

        self.endRunLock.release()

        return True

    def SaveLocalHistFile(self, s, a, r, s_, terminal):
        self.histFileLock.acquire()
        
        transitions = {}
        self.historyMngr.AddMetaDataFields2Dict(transitions)
        transitions["s"] = (s  + 1) * transitions["maxStateVals"] / 2
        transitions["a"] = a
        transitions["r"] = r
        transitions["s_"] = (s_  + 1) * transitions["maxStateVals"] / 2
        transitions["terminal"] = terminal
        transitions["worker"] = threading.current_thread().getName()

        fName = self.histFileName + str(self.counter4HistFiles)
        self.counter4HistFiles += 1

        self.histFileLock.release()

        pd.to_pickle(transitions, fName + '.gz', 'gzip') 

    def GetHistory(self, agentName):
        hist = self.agentsHistory[agentName]   
        s, a, r, s_, terminal = self.historyMngr.GetHistory(hist, shuffle=False)
        # if self.counter4HistFiles < 2000:
        #     self.SaveLocalHistFile(s, a, r, s_, terminal)

        return s, a, r, s_, terminal

    def Train(self):
        agentName = threading.current_thread().getName()

        s,a,r,s_, terminal = self.GetHistory(agentName)

        start = datetime.datetime.now()      
        self.decisionMaker.learn(s, a, r, s_, terminal, insert2Graph=True)
        diff = datetime.datetime.now() - start
        msDiff = diff.seconds * 1000 + diff.microseconds / 1000
        
        #print("\t", threading.current_thread().getName(), ":", self.agentName,"-> training with hist size = ", len(r), ", last", msDiff, "milliseconds")

        return self.decisionMaker.NumRuns()

    def CheckModel(self, agent, plotGraphs=False, withDfltModel=False, statesIdx2Check=[], actions2Check=[]):
        np.set_printoptions(precision=2, suppress=True)

        numStates2Check = 1000

        states2Check = []
        vals = []

        validActionsResults = {}
        for i in range(numStates2Check):
            s = self.DrawStateFromHist(realState=False)
            if len(s) > 0:
                states2Check.append(s)
                validActions = agent.ValidActions(s)
                v = self.ActionsValues(s, validActions, targetValues=True)
                
                keyValidActions = str(validActions)
                if keyValidActions not in validActionsResults:
                    validActionsResults[keyValidActions] = []

                validActionsResults[keyValidActions].append(v[validActions])

                vWithNan = np.zeros(len(v), float)
                vWithNan.fill(np.nan)
                for i in validActions:
                    vWithNan[i] = v[i]
                vals.append(vWithNan)
        print("average network state:")
        print(self.agentName, "maxVals =", self.historyMngr.transitions["maxStateVals"])
        print(self.agentName, ": current a3c num runs = ", self.decisionMaker.NumRuns()," avg values =", np.nanmean(vals, axis=0))

        print("\n\nnetwork state according to valid actions:")
        for key, value in validActionsResults.items():
            print("for valid actions:", key, "avg val =", np.average(value, axis=0))
