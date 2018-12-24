# import state
from agent_super import SuperState
from agent_attack import AttackState
from agent_navigation import NavigationState

# import actions
from agent_super import SuperActions
from agent_attack import AttackActions
from agent_navigation import NavigationActions


BUTTONS_ORDER = ["MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT", "MOVE_LEFT", "MOVE_RIGHT", "ATTACK", "SPEED", "CROUCH"]
ACTION_LIST = [['MOVE_FORWARD', 'MOVE_BACKWARD'], ['TURN_LEFT', 'TURN_RIGHT'], ['ATTACK']]


ACTIONS = {}
STATES = {}
def StatesParams2Agent(agentName):
    params = {}
    params["frame_size"] = STATES[agentName].SIZE
    params["var_size"] = len(STATES[agentName].VARIABLES_4LEARNING)
    return params
        
def NumActions2Agent(agentName):
    return int(ACTIONS[agentName].SIZE)

STATES["super_agent"] = SuperState()
STATES["attack"] = AttackState()
STATES["navigation"] = NavigationState()


ACTIONS["super_agent"] = SuperActions()
ACTIONS["attack"] = AttackActions()
ACTIONS["navigation"] = NavigationActions()


