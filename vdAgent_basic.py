#!/usr/bin/python3

from __future__ import division
from __future__ import print_function


import os 
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf

from algo_decisionMaker import DecisionMakerExperienceReplay
from algo_decisionMaker import DecisionMakerOnlineAsync

from algo_dqn import DQN_PARAMS
from algo_a2c import A2C_PARAMS
from algo_a3c import A3C_PARAMS
from algo_orig import ORIG_PARAMS

from utils import VD_Env

AGENT_DIR = "VD_Simple/"
AGENT_NAME = "VD_Simple"

FRAME_REPEAT = 12
FRAME_RESOLUTION = (30, 45)
VARIABLES_4LEARNING = ["health", "sel_ammo"]


NUM_TRIALS_2_SAVE = 50

DQN = "dqn"
DQN_EMBEDDING = "dqnEmbedding"
A2C = 'A2C'
A2C_EMBEDDING = 'A2CEmbedding'
A3C = 'A3C'
ORIG = 'orig'

# data for run type
TYPE = "type"
DECISION_MAKER_TYPE = "dm_type"
DECISION_MAKER_NAME = "dm_name"
HISTORY = "history"
RESULTS = "results"
PARAMS = 'params'
DIRECTORY = 'directory'

RUN_TYPES = {}

RUN_TYPES[DQN] = {}
RUN_TYPES[DQN][DECISION_MAKER_TYPE] = "DecisionMakerExperienceReplay"
RUN_TYPES[DQN][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN][DIRECTORY] = "VDAgent_dqn"
RUN_TYPES[DQN][PARAMS] = DQN_PARAMS(frameSize=FRAME_RESOLUTION, gameVarsSize=len(VARIABLES_4LEARNING), numActions=0, numTrials2Save=NUM_TRIALS_2_SAVE, includeEmbedding=False)
RUN_TYPES[DQN][DECISION_MAKER_NAME] = "dqn"
RUN_TYPES[DQN][HISTORY] = "replayHistory"
RUN_TYPES[DQN][RESULTS] = "results"

RUN_TYPES[DQN_EMBEDDING] = {}
RUN_TYPES[DQN_EMBEDDING][DECISION_MAKER_TYPE] = "DecisionMakerExperienceReplay"
RUN_TYPES[DQN_EMBEDDING][TYPE] = "DQN_WithTarget"
RUN_TYPES[DQN_EMBEDDING][DIRECTORY] = "VDAgent_dqnEmbedding"
RUN_TYPES[DQN_EMBEDDING][PARAMS] = DQN_PARAMS(frameSize=FRAME_RESOLUTION, gameVarsSize=len(VARIABLES_4LEARNING), numActions=0, numTrials2Save=100, includeEmbedding=True, learning_rate=0.0001)
RUN_TYPES[DQN_EMBEDDING][DECISION_MAKER_NAME] = "dqnEmbedding"
RUN_TYPES[DQN_EMBEDDING][HISTORY] = "replayHistory"
RUN_TYPES[DQN_EMBEDDING][RESULTS] = "results"

RUN_TYPES[A2C] = {}
RUN_TYPES[A2C][DECISION_MAKER_TYPE] = "DecisionMakerExperienceReplay"
RUN_TYPES[A2C][TYPE] = "A2C"
RUN_TYPES[A2C][DIRECTORY] = "VDAgent_A2C"
RUN_TYPES[A2C][PARAMS] = A2C_PARAMS(frameSize=FRAME_RESOLUTION, gameVarsSize=len(VARIABLES_4LEARNING), numActions=0, numTrials2Save=NUM_TRIALS_2_SAVE, includeEmbedding=False)
RUN_TYPES[A2C][DECISION_MAKER_NAME] = "A2C"
RUN_TYPES[A2C][HISTORY] = "replayHistory"
RUN_TYPES[A2C][RESULTS] = "results"

RUN_TYPES[A2C_EMBEDDING] = {}
RUN_TYPES[A2C_EMBEDDING][DECISION_MAKER_TYPE] = "DecisionMakerExperienceReplay"
RUN_TYPES[A2C_EMBEDDING][TYPE] = "A2C"
RUN_TYPES[A2C_EMBEDDING][DIRECTORY] = "VDAgent_A2CEmbedding"
RUN_TYPES[A2C_EMBEDDING][PARAMS] = A2C_PARAMS(frameSize=FRAME_RESOLUTION, gameVarsSize=len(VARIABLES_4LEARNING), numActions=0, numTrials2Save=NUM_TRIALS_2_SAVE, includeEmbedding=True)
RUN_TYPES[A2C_EMBEDDING][DECISION_MAKER_NAME] = "A2CEmbedding"
RUN_TYPES[A2C_EMBEDDING][HISTORY] = "replayHistory"
RUN_TYPES[A2C_EMBEDDING][RESULTS] = "results"


RUN_TYPES[ORIG] = {}
RUN_TYPES[ORIG][DECISION_MAKER_TYPE] = "DecisionMakerExperienceReplay"
RUN_TYPES[ORIG][TYPE] = "ORIG_MODEL"
RUN_TYPES[ORIG][DIRECTORY] = "VDAgent_ORIG"
RUN_TYPES[ORIG][PARAMS] = ORIG_PARAMS(frameSize=FRAME_RESOLUTION, gameVarsSize=len(VARIABLES_4LEARNING), numActions=0, numTrials2Learn=NUM_TRIALS_2_SAVE, numTrials2Save=NUM_TRIALS_2_SAVE)
RUN_TYPES[ORIG][DECISION_MAKER_NAME] = "ORIG"
RUN_TYPES[ORIG][HISTORY] = "replayHistory"
RUN_TYPES[ORIG][RESULTS] = "results"

BUTTONS_ORDER = ["MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT", "MOVE_LEFT", "MOVE_RIGHT", "ATTACK", "SPEED", "CROUCH"]
ACTION_LIST = [['MOVE_FORWARD', 'MOVE_BACKWARD'], ['TURN_LEFT', 'TURN_RIGHT'], ['ATTACK']]

WEAPONS_PREFERENCES = [
    ('bfg9000', 'cells', 7), ('shotgun', 'shells', 3),
    ('chaingun', 'bullets', 4), ('plasmarifle', 'cells', 6),
    ('pistol', 'bullets', 2), ('rocketlauncher', 'rockets', 5)
]

def GetRunTypeVDAgent(configDict):
    if configDict[AGENT_NAME] == "none":
        return {}
    else:
        return RUN_TYPES[configDict[AGENT_NAME]]

def CreateDecisionMakerVDAgent(configDict, env, isMultiThreaded, dmCopy=None, hyperParamsDict=None):
    dmCopy = "" if dmCopy==None else "_" + str(dmCopy)
    
    runType = GetRunTypeVDAgent(configDict)
 
    # create agent dir
    directory = configDict["directory"] + "/" + AGENT_DIR
    if not os.path.isdir("./" + directory):
        os.makedirs("./" + directory)
    directory += runType[DIRECTORY] + dmCopy

    dmClass = eval(runType[DECISION_MAKER_TYPE])

    runType[PARAMS].numActions = env.NumActions()

    decisionMaker = dmClass(modelType=runType[TYPE], modelParams = runType[PARAMS], decisionMakerName = runType[DECISION_MAKER_NAME], agentName=AGENT_NAME,
                                    resultFileName=runType[RESULTS], historyFileName=runType[HISTORY], directory=directory, isMultiThreaded=isMultiThreaded)
                                
    if "printTrain" in  configDict and configDict["printTrain"]:
        decisionMaker.printTrain = True

    return decisionMaker, runType


class VDAgent:
    def __init__(self, configDict, decisionMaker, isMultiThreaded=False, dmCopy=None, training=True, playMode=False):
        
        self.env = VD_Env(configDict["mapPath"], FRAME_REPEAT, buttonsOrder=BUTTONS_ORDER, actionList=ACTION_LIST, playMode=playMode)

        self.training = training
        if decisionMaker == None:
            decisionMaker, _ = CreateDecisionMakerVDAgent(configDict, self.env, isMultiThreaded, dmCopy=dmCopy)
        
        self.decisionMaker = decisionMaker

        self.history = self.decisionMaker.GetHistoryInstance()
        
        self.terminalState = [np.zeros(FRAME_RESOLUTION, float), np.zeros(len(VARIABLES_4LEARNING), float) ] 
        
        if "learn4Transition" in configDict and self.training:
            self.onlineLearning = configDict["learn4Transition"]
        else:
            self.onlineLearning = False

        self.idx2Check = 0

    def GetDecisionMaker(self):
        return self.decisionMaker

    def RunEpisode(self):      
        self.env.new_episode()

        r = 0
        step = 0
        
        while not self.env.is_episode_finished():
            step += 1
            r = self.Step()
            if self.onlineLearning:
                self.decisionMaker.Train()

        
        acumR = self.env.get_total_reward()
        if self.training:
            self.decisionMaker.end_run(r, acumR, step)
        else:
            self.decisionMaker.end_test_run(r, acumR, step)
                
    def PlayEpisode(self):
    
        self.env.new_episode()

        r = 0
        while not self.env.is_episode_finished():
            s = self.GetState()
            action = self.ChooseAction(s)            
            self.env.PlayAction(action)

        r = 1.0 if r > 0 else -1.0
        print("\tterminal reward =", r)


    def Close(self):
        self.env.close()

    def GetState(self):
        state = self.env.get_state()
        img = state.screen_buffer
        img = skimage.transform.resize(img, FRAME_RESOLUTION)
        img = img.astype(np.float)

        gameVars = self.env.GetGameVariables()
        return [img, np.array([gameVars[v] for v in VARIABLES_4LEARNING])]

    def Step(self):
        s = self.GetState()

        doomAction, action = self.ChooseAction(s)
        gameReward, terminal = self.env.ActAction(doomAction)
        s_ = self.GetState() if not terminal else self.terminalState

        if self.env.is_episode_finished():
            r = 1.0 if gameReward > 0 else -1.0
        else:
            r = gameReward / 100

        self.history.add_transition(s, action, r, s_, terminal)

        return r

    def ChooseAction(self, s):
        validActions = self.env.ValidActions()
        targetVals = not self.training
        action = self.decisionMaker.choose_action(s, validActions, targetValues=targetVals)

        # translate action to game action
        doomAction = self.env.Action2DoomAction(action)        
        
        # 
        gameVars = self.env.GetGameVariables()
        for weapon_name, weapon_ammo, weapon_id in WEAPONS_PREFERENCES:
            min_ammo = 40 if weapon_name == 'bfg9000' else 1
            if gameVars[weapon_name] > 0 and gameVars[weapon_ammo] >= min_ammo:
                if gameVars['sel_weapon'] != weapon_id:
                    # action = ([False] * self.mapping['SELECT_WEAPON%i' % weapon_id]) + [True]
                    doomAction = self.env.AddSwitchWeaponAction(doomAction, weapon_id)
                    switchWeaponId = weapon_id
                break

        return doomAction, action


if __name__ == "__main__":
    import sys
    from absl import app
    from absl import flags

    from utils_results import PlotResults


    flags.DEFINE_string("directoryPrefix", "", "directory names to take results")
    flags.DEFINE_string("directoryNames", "", "directory names to take results")
    flags.DEFINE_string("grouping", "100", "grouping size of results.")
    flags.DEFINE_string("max2Plot", "none", "grouping size of results.")
    flags.FLAGS(sys.argv)

    directoryNames = (flags.FLAGS.directoryNames).split(",")
    for d in range(len(directoryNames)):
        directoryNames[d] = flags.FLAGS.directoryPrefix + directoryNames[d]
    
    grouping = int(flags.FLAGS.grouping)
    if flags.FLAGS.max2Plot == "none":
        max2Plot = None
    else:
        max2Plot = int(flags.FLAGS.max2Plot)

    if "results" in sys.argv:
        PlotResults(AGENT_NAME, AGENT_DIR, RUN_TYPES, runDirectoryNames=directoryNames, grouping=grouping, maxTrials2Plot=max2Plot)
    elif "multipleResults" in sys.argv:
        PlotResults(AGENT_NAME, AGENT_DIR, RUN_TYPES, runDirectoryNames=directoryNames, grouping=grouping, maxTrials2Plot=max2Plot, multipleDm=True)
    elif "multipleResultsSinglePlot" in sys.argv:
        PlotResults(AGENT_NAME, AGENT_DIR, RUN_TYPES, runDirectoryNames=directoryNames, grouping=grouping, maxTrials2Plot=max2Plot, multipleDm=True, singlePlot=True)
