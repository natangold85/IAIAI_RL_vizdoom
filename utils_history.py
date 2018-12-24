import numpy as np
import pandas as pd
import pickle
import os.path
import datetime
import sys

from multiprocessing import Lock
from utils import EmptyLock

from random import sample

def GetHistoryFromFile(name):
    transitions = None
    if os.path.isfile(name + '.gz') and os.path.getsize(name + '.gz') > 0:
        transitions = pd.read_pickle(name + '.gz', compression='gzip')

    return transitions

def JoinTransitions(target, toJoin):
    keyTransitions = ["s", "a", "r", "s_", "terminal"]
    for k in keyTransitions:
        if k in toJoin.keys():
            if k in target.keys():
                target[k] += toJoin [k]
            else:
                target[k] = toJoin [k]

    if "maxStateVals" in toJoin.keys():
        if "maxStateVals" in target.keys():
            target["maxStateVals"] = np.maximum(target["maxStateVals"], toJoin["maxStateVals"])
        else:
            target["maxStateVals"] = toJoin["maxStateVals"]

    if "rewardMax" in toJoin.keys():
        if "rewardMax" in target.keys():
            target["rewardMax"] = max(target["rewardMax"], toJoin["rewardMax"])
        else:
            target["rewardMax"] = toJoin["rewardMax"]

    if "rewardMin" in toJoin.keys():
        if "rewardMin" in target.keys():
            target["rewardMin"] = min(target["rewardMin"], toJoin["rewardMin"])
        else:
            target["rewardMin"] = toJoin["rewardMin"]


class History:
    def __init__(self, params, historyFileName = '', directory = '', isMultiThreaded=False, createAllHistFiles=False):

        self.params = params
        self.transitions = {}
        self.transitionKeys = ["sFrame", "sVars", "a", "r", "s_Frame", "s_Vars", "terminal"]

        self.transitions["currIdx"] = 0
        self.transitions["size"] = 0

        if len(params.frameSize) == 1:
            frame_shape = (params.maxReplaySize, params.frameSize[0], 1)
            self.copyState = self.CopyState1D
        else:
            frame_shape = (params.maxReplaySize, params.frameSize[0], params.frameSize[1], 1)
            self.copyState = self.CopyState2D
        
        self.transitions["sFrame"] = np.zeros(frame_shape, dtype=np.float)
        self.transitions["s_Frame"] = np.zeros(frame_shape, dtype=np.float)

        self.transitions["sVars"] = np.zeros((params.maxReplaySize, params.gameVarsSize), dtype=np.float)
        self.transitions["s_Vars"] = np.zeros((params.maxReplaySize, params.gameVarsSize), dtype=np.float)

        self.transitions["a"] = np.zeros(params.maxReplaySize, dtype=np.int32)
        self.transitions["r"] = np.zeros(params.maxReplaySize, dtype=np.float)
        self.transitions["terminal"] = np.zeros(params.maxReplaySize, dtype=np.bool)

        self.transitions["maxAbsReward"] = 0.0

        self.transitions["maxFrameVals"] = np.ones(params.frameSize, float)
        self.transitions["maxVarsVals"] = np.ones(params.gameVarsSize, float)

        self.isMultiThreaded = isMultiThreaded
        if isMultiThreaded:
            self.histLock = Lock()
        else:
            self.histLock = EmptyLock()
            
        self.metaDataFields = ["maxVarsVals", "maxFrameVals", "maxAbsReward"]

        if historyFileName != '':
            self.histFileName = directory + historyFileName
        else:
            self.histFileName = historyFileName
    def CopyState1D(self, key, s, idx):
        self.transitions[key][idx, :, 0] = s.copy()
    def CopyState2D(self, key, s, idx):
        self.transitions[key][idx, :, :, 0] = s.copy()

    def add_transition(self, s, a, r, s_, terminal = False):
        self.histLock.acquire()
        currIdx = self.transitions["currIdx"]
        self.transitions["currIdx"] = (self.transitions["currIdx"] + 1) % self.params.maxReplaySize
        self.transitions["size"] = min(self.transitions["size"] + 1, self.params.maxReplaySize)
        self.histLock.release()

        self.transitions["maxAbsReward"] = max(self.transitions["maxAbsReward"], abs(r))

        sFrame = s[0]
        s_Frame = s_[0]

        self.copyState("sFrame", sFrame, currIdx)
        self.copyState("s_Frame", s_Frame, currIdx)

        self.transitions["a"][currIdx] = a
        self.transitions["r"][currIdx] = r
        self.transitions["terminal"][currIdx] = terminal

        if self.params.gameVarsSize > 0:
            sVars = s[1]
            s_Vars = s_[1]

            self.transitions["maxVarsVals"] = np.maximum(self.transitions["maxVarsVals"], np.maximum(abs(sVars), abs(s_Vars)))
            self.transitions["sVars"][currIdx, :] = sVars.copy()
            self.transitions["s_Vars"][currIdx, :] = s_Vars.copy()

    def Load(self):
        if os.path.isfile(self.histFileName + '.gz') and os.path.getsize(self.histFileName + '.gz') > 0:
            self.transitions = pd.read_pickle(self.histFileName + '.gz', compression='gzip')

    def Save(self):
        pd.to_pickle(self.transitions, self.histFileName + '.gz', 'gzip') 

    def Reset(self):
        self.transitions["currIdx"] = 0
        self.transitions["size"] = 0

    def CleanHistory(self):
        for key in self.transitionKeys:
            self.transitions[key] = []

    def get_sample(self, sample_size):
        i = sample(range(0, self.transitions["size"]), sample_size)
        
        r = self.transitions["r"][i]
        
        r = r / self.transitions["maxAbsReward"] if self.params.normalizeRewards else r

        sVars = self.transitions["sVars"][i]
        sVars = sVars / self.transitions["maxVarsVals"] if self.params.normalizeState else sVars

        s_Vars = self.transitions["sVars"][i]
        s_Vars = s_Vars / self.transitions["maxVarsVals"] if self.params.normalizeState else s_Vars

        s = [self.transitions["sFrame"][i], sVars]
        s_ = [self.transitions["s_Frame"][i], s_Vars]
        
        return s, self.transitions["a"][i], r, s_, self.transitions["terminal"][i]

    def DrawState(self, realState=True):
        if realState:
            s, _, _, _, _ = self.get_sample(1)
            return s

    def Size(self):
        return self.transitions["size"]
