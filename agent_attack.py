#!/usr/bin/python3

from __future__ import division
from __future__ import print_function


import os 
import numpy as np
import skimage.color, skimage.transform
import tensorflow as tf

from agent_base import BaseAgent
from utils import VD_Env, WEAPONS_PREFERENCES

from algo_decisionMaker import CreateDecisionMaker

AGENT_NAME = "attack"

class AttackActions:
    ACTION_LIST = [['MOVE_FORWARD', 'MOVE_BACKWARD'], ['TURN_LEFT', 'TURN_RIGHT'], ['ATTACK']]
    SIZE = np.prod([len(group) + 1 for group in ACTION_LIST])

class AttackState:
    FRAME_REPEAT = 12
    SIZE = (30, 45)
    VARIABLES_4LEARNING = ["health", "sel_ammo"]

AttackRewards = {
    'base_reward': 0.,
    "kill": 1.0,
    'suicide': -5.0,
    "death": -1.0,
    "injured": -0.1,
    "medikit": 0,
    'armor': 0,
    'weapon': 0,
    'ammo': 0,
    'use_ammo': 0,
}

class AttackAgent(BaseAgent):
    def __init__(self, gameEnv, configDict, decisionMaker, trainList=[], playList=[], isMultiThreaded=False, dmCopy=None, playMode=False):
        super(AttackAgent, self).__init__(AGENT_NAME)

        self.env = gameEnv

        self.training = AGENT_NAME in trainList 
        self.playing = AGENT_NAME in playList or "inherit" in playList

        if decisionMaker == None:
            decisionMaker, _ = CreateDecisionMaker(AGENT_NAME, configDict, isMultiThreaded, dmCopy=dmCopy)

        self.decisionMaker = decisionMaker
        self.history = self.decisionMaker.historyMngr
    
        actionList = self.env.GetActionsRec(AttackActions.ACTION_LIST)
        self.actions = []
        for i in range(len(actionList)):
            self.actions.append(self.env.GetActionIdx(actionList[i]))

        self.terminalState = [np.zeros(AttackState.SIZE, float), np.zeros(len(AttackState.VARIABLES_4LEARNING), float) ] 

        self.InitEpisode()

    def InitEpisode(self):
        self.curr_action = None
        self.curr_state = None
        self.prev_state = None
        self.curr_reward = None

        self.prevVariables = None
        self.currVariables = None

        self.accumulatedReward = 0.0


    def CreateState(self):        
        state = self.env.get_state()
        img = state.screen_buffer
        img = skimage.transform.resize(img, AttackState.SIZE)
        img = img.astype(np.float)

        gameVars = self.env.GetGameVariables()

        self.currVariables =  gameVars
        self.curr_state = [img, np.array([gameVars[v] for v in AttackState.VARIABLES_4LEARNING])]

    def ChooseAction(self):
        validActions = list(range(AttackActions.SIZE))
        targetVals = not self.training
        self.curr_action = self.decisionMaker.choose_action(self.curr_state, validActions, targetValues=targetVals)

        # translate action to game action
        doomAction = self.env.Action2DoomAction(self.actions[self.curr_action])   
        
        return doomAction


    def Learn(self, terminal=False):
        if self.curr_action != None:
            r = self.CalcReward()
            r = r if not terminal else self.accumulatedReward
            self.curr_state = self.terminalState if terminal else self.curr_state
            self.history.add_transition(self.prev_state.copy(), self.curr_action, r, self.curr_state.copy(), terminal)

        self.prev_state = self.curr_state
        self.prevVariables = self.currVariables
        self.curr_action = None


    def EndRun(self, reward, score, steps):
        if self.training:
            self.decisionMaker.end_run(self.accumulatedReward, score, steps)
        self.InitEpisode()    

    def CalcReward(self):
        r = 0
        
        if self.prevVariables == None:
            return r
        
        # kill
        newKills = self.currVariables['frag_count'] - self.prevVariables['frag_count']
        if newKills > 0:
            r += AttackRewards["kill"] * newKills
            self.accumulatedReward += 0.1 * newKills

        if self.env.is_player_dead():
            r += AttackRewards["death"]
            self.accumulatedReward += -1.0

        dHealth = self.currVariables["health"] - self.prevVariables["health"]
        if dHealth != 0:
            if dHealth > 0:
                r += AttackRewards["medikit"] 
            else:
                r += AttackRewards["injured"]

        if self.currVariables["armor"] > self.prevVariables["armor"]:
            r += AttackRewards["armor"] 

        # if self.currVariables['sel_ammo'] < self.prevVariables['sel_ammo']:
        #     r += AttackRewards["use_ammo"] 
        #     print("\t\tuse ammo")
        # elif self.env.IsAttackAction(self.actions[self.curr_action]):
        #     print("ERROR: attack action without ammo use")
        # else:
        #     print("\tnon attack action")

        if self.currVariables['frag_count'] < self.prevVariables['frag_count']:
            print("\n\nSUICIDE!! - CHECK!!\n\n")
            r += AttackRewards["suicide"] 
        
        

        return r


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