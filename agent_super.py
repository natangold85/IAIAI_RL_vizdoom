#!/usr/bin/python3

from __future__ import division
from __future__ import print_function


import os 
from random import sample, randint, random
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf

from agent_attack import AttackAgent
from agent_navigation import NavigationAgent

from utils import VD_Env, WEAPONS_PREFERENCES

from algo_decisionMaker import CreateDecisionMaker

AGENT_NAME = "super_agent"

SUB_AGENT_ID_NAVIGATION = 0
SUB_AGENT_ID_ATTACK = 1

SUBAGENTS_NAMES = {}
SUBAGENTS_NAMES[SUB_AGENT_ID_NAVIGATION] = "NavigationAgent"
SUBAGENTS_NAMES[SUB_AGENT_ID_ATTACK] = "AttackAgent"

class SuperActions:
    NAVIGATION = 0
    ATTACK = 1
    SIZE = 2

class SuperState:
    ENEMY_PRESENCE = 0
    SIZE = (1,)
    VARIABLES_4LEARNING = []


FRAME_REPEAT = 12

class SuperAgent:
    def __init__(self, configDict, decisionMaker, trainList=[], playList=[], isMultiThreaded=False, dmCopy=None, playMode=False):
        
        from agentStatesAndActions import BUTTONS_ORDER, ACTION_LIST
        self.env = VD_Env(configDict["mapPath"], FRAME_REPEAT, buttonsOrder=BUTTONS_ORDER, actionList=ACTION_LIST, playMode=playMode)

        self.training = AGENT_NAME in trainList
        self.playing = True if self.training else AGENT_NAME in trainList

            
        saPlayList = "inherit" if self.playing else playList

        if decisionMaker == None:
            decisionMaker, _ = CreateDecisionMaker(AGENT_NAME, configDict, isMultiThreaded, dmCopy=dmCopy)

        self.decisionMaker = decisionMaker
        self.history = self.decisionMaker.historyMngr
        
        self.subAgents = {}
        self.subAgentAction = {}
        for key, name in SUBAGENTS_NAMES.items():
            saClass = eval(name)
            saDM = self.decisionMaker.GetSubAgentDecisionMaker(key)
            self.subAgents[key] = saClass(gameEnv=self.env, configDict=configDict, decisionMaker=saDM, isMultiThreaded=isMultiThreaded, 
                                                playList=saPlayList, trainList=trainList, dmCopy=dmCopy)

            self.decisionMaker.SetSubAgentDecisionMaker(key, self.subAgents[key].GetDecisionMaker())
            self.subAgentAction[key] = None

        self.terminalState = [np.zeros(SuperState.SIZE, float), np.zeros(len(SuperState.VARIABLES_4LEARNING), float) ] 
        
        self.idx2Check = 0

        if not self.playing:
            self.saPlaying = -1
            for key, sa in self.subAgents.items():
                if sa.playing:
                    self.saPlaying = key


    def GetDecisionMaker(self):
        return self.decisionMaker

    def GetAgentByName(self, name):
        if AGENT_NAME == name:
            return self
        
        for sa in self.subAgents.values():
            ret = sa.GetAgentByName(name)
            if ret != None:
                return ret
            
        return None

    def RunEpisode(self):      
        self.env.new_episode()

        r = 0
        step = 0

        self.curr_state = None
        self.prev_state = None
        
        self.curr_action = None
        self.curr_reward = None

        while not self.env.IsFinal():
            step += 1
            self.Step()

        self.EndRun(step)
        
                
    def PlayEpisode(self):   
        self.env.new_episode()

        r = 0
        while not self.env.IsFinal():
            self.CreateState()
            doomAction = self.ChooseAction()  
            self.env.PlayAction(doomAction)
            sleep(0.5)

        r = 1.0 if r > 0 else -1.0
        print("\tterminal reward =", r)
        self.idx2Check = (self.idx2Check + 1) % 19

    def Close(self):
        self.env.close()

    def CreateState(self):
        for sa in self.subAgents.values():
            sa.CreateState()

        state = self.env.get_state()
        gameVars = self.env.GetGameVariables()

        self.curr_state = np.array([0])

    def Step(self):
        self.CreateState()
        self.Learn()
        doomAction = self.ChooseAction()

        gameReward, terminal = self.env.ActAction(doomAction)
        self.curr_reward = gameReward / 100

        return terminal

    def Learn(self, terminal=False):
        for sa in self.subAgents.values():
            sa.Learn(terminal)

        if self.training and self.curr_action != None:
            self.history.add_transition(self.curr_state, self.curr_action, self.curr_reward, self.prev_state, terminal)

        self.prev_state = self.curr_state
        self.curr_action = None

    def ChooseAction(self):
        for key, sa in self.subAgents.items():
            self.subAgentAction[key] = sa.ChooseAction()

        if self.playing:
            validActions = list(range(SuperActions.SIZE))
            targetVals = not self.training
            self.curr_action = self.decisionMaker.choose_action(self.curr_state, validActions, targetValues=targetVals)
        else:
            self.curr_action = self.saPlaying

        # translate action to game action
        doomAction = self.subAgentAction[self.curr_action]

        
        # switch to strongest available weapon if possible (as arnold agent does)
        gameVars = self.env.GetGameVariables()
        for weapon_name, weapon_ammo, weapon_id in WEAPONS_PREFERENCES:
            min_ammo = 40 if weapon_name == 'bfg9000' else 1
            if gameVars[weapon_name] > 0 and gameVars[weapon_ammo] >= min_ammo:
                if gameVars['sel_weapon'] != weapon_id:
                    doomAction = self.env.AddSwitchWeaponAction(doomAction, weapon_id)
                break
        
        return doomAction

    def EndRun(self, step):
        self.curr_reward = 1.0 if self.curr_reward > 0 else -1.0
        acumR = self.env.get_total_reward()

        self.curr_state = self.terminalState
        self.Learn(terminal=True)

        for sa in self.subAgents.values():
            sa.EndRun(self.curr_reward, acumR, step)

        if self.training:
            self.decisionMaker.end_run(self.curr_reward, acumR, step)

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
        PlotResults(AGENT_NAME, runDirectoryNames=directoryNames, grouping=grouping, maxTrials2Plot=max2Plot)
    elif "multipleResults" in sys.argv:
        PlotResults(AGENT_NAME, runDirectoryNames=directoryNames, grouping=grouping, maxTrials2Plot=max2Plot, multipleDm=True)