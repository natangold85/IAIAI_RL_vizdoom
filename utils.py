import random
import math
import time

import numpy as np

import itertools as it

import vizdoom as vzd
from vizdoom import Button, GameVariable

WEAPONS_PREFERENCES = [
    ('bfg9000', 'cells', 7), ('shotgun', 'shells', 3),
    ('chaingun', 'bullets', 4), ('plasmarifle', 'cells', 6),
    ('pistol', 'bullets', 2), ('rocketlauncher', 'rockets', 5)
]


GAME_VARIABLES = {
    # 'KILLCOUNT': GameVariable.KILLCOUNT,
    # 'ITEMCOUNT': GameVariable.ITEMCOUNT,
    # 'SECRETCOUNT': GameVariable.SECRETCOUNT,
    'frag_count': GameVariable.FRAGCOUNT,
    # 'DEATHCOUNT': GameVariable.DEATHCOUNT,
    'health': GameVariable.HEALTH,
    'armor': GameVariable.ARMOR,
    # 'DEAD': GameVariable.DEAD,
    # 'ON_GROUND': GameVariable.ON_GROUND,
    # 'ATTACK_READY': GameVariable.ATTACK_READY,
    # 'ALTATTACK_READY': GameVariable.ALTATTACK_READY,
    'sel_weapon': GameVariable.SELECTED_WEAPON,
    'sel_ammo': GameVariable.SELECTED_WEAPON_AMMO,
    # 'AMMO0': GameVariable.AMMO0,  # UNK
    # 'AMMO1': GameVariable.AMMO1,  # fist weapon, should always be 0
    'bullets': GameVariable.AMMO2,  # bullets
    'shells': GameVariable.AMMO3,  # shells
    # 'AMMO4': GameVariable.AMMO4,  # == AMMO2
    'rockets': GameVariable.AMMO5,  # rockets
    'cells': GameVariable.AMMO6,  # cells
    # 'AMMO7': GameVariable.AMMO7,  # == AMMO6
    # 'AMMO8': GameVariable.AMMO8,  # UNK
    # 'AMMO9': GameVariable.AMMO9,  # UNK
    # 'WEAPON0': GameVariable.WEAPON0,  # UNK
    'fist': GameVariable.WEAPON1,  # Fist, should be 1, unless removed
    'pistol': GameVariable.WEAPON2,  # Pistol
    'shotgun': GameVariable.WEAPON3,  # Shotgun
    'chaingun': GameVariable.WEAPON4,  # Chaingun
    'rocketlauncher': GameVariable.WEAPON5,  # Rocket Launcher
    'plasmarifle': GameVariable.WEAPON6,  # Plasma Rifle
    'bfg9000': GameVariable.WEAPON7,  # BFG9000
    # 'WEAPON8': GameVariable.WEAPON8,  # UNK
    # 'WEAPON9': GameVariable.WEAPON9,  # UNK
    'position_x': GameVariable.POSITION_X,
    'position_y': GameVariable.POSITION_Y,
    'position_z': GameVariable.POSITION_Z,
    # 'velocity_x': GameVariable.VELOCITY_X,
    # 'velocity_y': GameVariable.VELOCITY_Y,
    # 'velocity_z': GameVariable.VELOCITY_Z,
    'kill_count' : GameVariable.KILLCOUNT,
}

class EmptyLock:
    def acquire(self):
        return
    def release(self):
        return

# params base
class ParamsBase:
    def __init__(self, frameSize, gameVarsSize, numActions, discountFactor = 0.95, accumulateHistory = True, maxReplaySize=500000, minReplaySize=1000, numTrials2Learn=None, numTrials2Save=100, withConvolution=True, includeEmbedding=False):
        self.frameSize = frameSize
        self.gameVarsSize = gameVarsSize

        self.numActions = numActions
        self.discountFactor = discountFactor
        self.accumulateHistory = accumulateHistory
        self.maxReplaySize = maxReplaySize
        self.minReplaySize = minReplaySize
        
        self.numTrials2Save = numTrials2Save
        self.numTrials2Learn = numTrials2Learn if numTrials2Learn != None else numTrials2Save

        self.normalizeRewards = False
        self.normalizeState = False
        self.numRepeatsTerminalLearning = 0

        self.withConvolution = withConvolution
        self.includeEmbedding = includeEmbedding

class VD_Env(vzd.DoomGame):   
    def __init__(self, mapPath, frameRepeat, buttonsOrder, actionList, variables2Use=list(GAME_VARIABLES.keys()), playMode=False):
        super(VD_Env, self).__init__()

        self.frameRepeat = frameRepeat

        self.speed = False
        self.crouch = False

        self.respawn_protect = True
        self.spawn_farthest = True

        self.freelook=False
        self.name = "IAIAI"

        self.color = 0

        self.numBots = 5

        self.variables2Use = variables2Use

        self.load_config(mapPath)

        if playMode:
            self.set_window_visible(True)
            self.set_mode(vzd.Mode.ASYNC_PLAYER)
        else:     
            self.set_window_visible(False)
            self.set_mode(vzd.Mode.PLAYER)
        
        # params taken from arnold initialization of deathmatch
        self.set_screen_format(vzd.ScreenFormat.GRAY8)
        self.set_screen_resolution(vzd.ScreenResolution.RES_400X225)

        self.set_render_hud(False) # render status bar
        self.set_render_minimal_hud(False)
        self.set_render_crosshair(True)
        self.set_render_weapon(True)
        self.set_render_decals(False)
        self.set_render_particles(False)
        self.set_render_effects_sprites(False)

        # game parameters taken from arnold initialization of deathmatch
        
        args = []   
        # deathmatch mode
        # players will respawn automatically after they die
        # autoaim is disabled for all players
        # args.append('-deathmatch')
        args.append('+sv_forcerespawn 1')
        args.append('+sv_noautoaim 1')

        # respawn invincibility / distance
        # players will be invulnerable for two second after spawning
        # players will be spawned as far as possible from any other players
        args.append('+sv_respawnprotect %i' % self.respawn_protect)
        args.append('+sv_spawnfarthest %i' % self.spawn_farthest)

        # freelook / agent name / agent color
        args.append('+freelook %i' % (1 if self.freelook else 0))
        args.append('+name %s' % self.name)
        args.append('+colorset %i' % self.color)

        # enable the cheat system (so that we can still
        # send commands to the game in self-play mode)
        args.append('+sv_cheats 1')
        self.args = args
        for arg in args:
            self.add_game_args(arg)

        self.buttonsMapping = self.MapButtons(buttonsOrder)

        self.set_doom_skill(1)

        for v in GAME_VARIABLES.values():
            self.add_available_game_variable(v)

        self.init()

        #n = self.get_available_buttons_size()
        self.doom_actions, self.actions2Str = self.GetActions(actionList)

    def MapButtons(self, buttonsAvailable):
        buttonsDict = {}
        for s in buttonsAvailable:
            buttonName = "Button." + s
            buttonClass = eval(buttonName)
            buttonsDict[s] = buttonClass
            self.add_available_button(buttonClass)

        # add buttons that are not actions
        for i in range(10):
            buttonName = "SELECT_WEAPON%i" % i
            buttonClass = eval("Button." + buttonName)
            self.add_available_button(buttonClass)
            buttonsDict[buttonName] = buttonClass

        # map buttons 2 idx
        buttonList = self.get_available_buttons()
        
        self.numButtons = len(buttonList)
        return {key: buttonList.index(val) for key, val in buttonsDict.items()}

    def init(self):
        super(VD_Env, self).init()

        game_state = self.get_state()
        self._screen_buffer = game_state.screen_buffer
        self._depth_buffer = game_state.depth_buffer
        self._labels_buffer = game_state.labels_buffer
        self._labels = game_state.labels

        # actor properties
        self.prev_properties = None
        self.properties = None

        # advance a few steps to avoid bugs due
        # to initial weapon changes in ACS
        self.advance_action(3)

        # if there are bots in the game, and if this is a new game
        self.UpdateBots()

    def GetActions(self, actionList):
        action2Str = self.GetActionsRec(actionList)

        doom_actions = []
        for sub_actions in action2Str:
            doom_action = [False] * self.numButtons
            for button, idx in self.buttonsMapping.items():
                doom_action[idx] = button in sub_actions
            
            doom_action[self.buttonsMapping["SPEED"]] = self.speed
            doom_action[self.buttonsMapping["CROUCH"]] = self.crouch
            
            doom_actions.append(doom_action)

        return doom_actions, action2Str

    def GetGameVariables(self):
        """
        Check and update game variables.
        """
        # read game variables
        
        variables = {var: self.get_game_variable(GAME_VARIABLES[var]) for var in self.variables2Use}
        variables = {k: (int(v) if v.is_integer() else float(v)) for k, v in variables.items()}
        return variables


    def GetActionsRec(self, actionList, currActionComb=[], currIdx=0):
        if currIdx == len(actionList):
            a = currActionComb.copy()
            a.sort()
            return [a]
        else:
            actionsComb = []
            for a in actionList[currIdx]:
                currActionComb.append(a)
                actionsComb += self.GetActionsRec(actionList, currActionComb, currIdx + 1)
                del currActionComb[-1]

            # add none action for option
            actionsComb += self.GetActionsRec(actionList, currActionComb, currIdx + 1)

            

            return actionsComb
    
    def GetActionIdx(self, actionStr):
        return self.actions2Str.index(actionStr)

    def UpdateBots(self):
        self.send_game_command("removebots")
        for _ in range(self.numBots):
            self.send_game_command("addbot")

    def NumActions(self):
        return len(self.doom_actions)

    def ValidActions(self):
        return list(range(len(self.doom_actions)))

    def ActAction(self, action):
        r = self.make_action(action, self.frameRepeat)
        return r, self.is_episode_finished()
    
    def PlayAction(self, action):
        self.set_action(action)
        r = 0
        for _ in range(self.frameRepeat):
            self.advance_action()
            r += self.get_last_reward()

        return r, self.is_episode_finished()

    def Action2DoomAction(self, actionNum):
        return self.doom_actions[actionNum]

    def AddSwitchWeaponAction(self, action, weaponId):
        action[self.buttonsMapping['SELECT_WEAPON%i' % weaponId]] = True
        return action

    def IsFinal(self):
        return self.is_player_dead() or self.is_episode_finished()

    def IsAttackAction(self, actionNum):
        actionStr = self.actions2Str[actionNum]
        return "ATTACK" in actionStr

 
        


