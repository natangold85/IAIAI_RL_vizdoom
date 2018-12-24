from algo_dqn import DQN_PARAMS
from algo_a2c import A2C_PARAMS
from algo_a3c import A3C_PARAMS

# possible types of decision maker
A2C = 'A2C'
DQN = 'dqn'
DQN2L = 'dqn_2l'
HEURISTIC = 'heuristic' 
USER_PLAY = 'play'

# params to dm
EMBEDDING = "Embedding"

# params 2 a2c
A2C_SAVE_EXP = 'Exp'

# data for run type
ALGO_TYPE = "algo_type"
DECISION_MAKER_TYPE = "dm_type"
DECISION_MAKER_NAME = "dm_name"
HISTORY = "history"
RESULTS = "results"
PARAMS = 'params'
DIRECTORY = 'directory'

NUM_TRIALS_2_SAVE = 50

def GetRunType(agentName, configDict):
    runType = {}
    agentRun = configDict[agentName]
    if agentRun == "none":
        return runType
    
    from agentStatesAndActions import StatesParams2Agent
    from agentStatesAndActions import NumActions2Agent

    params = StatesParams2Agent(agentName)

    frameSize = params["frame_size"]
    gameVarsSize = params["var_size"]

    numActions = NumActions2Agent(agentName)
    stateParams = StatesParams2Agent(agentName) 
    withConvolution = len(stateParams["frame_size"]) > 1


    runType[HISTORY] = "history"
    runType[RESULTS] = "results"
    runType[DIRECTORY] = ""
    
    if agentRun == DQN:
        runType[DECISION_MAKER_TYPE] = "DecisionMakerExperienceReplay"
        runType[ALGO_TYPE] = "DQN_WithTarget"
        runType[DIRECTORY] = agentName + "_dqn"
        runType[PARAMS] = DQN_PARAMS(frameSize=frameSize, gameVarsSize=gameVarsSize, numActions=numActions, numTrials2Save=NUM_TRIALS_2_SAVE, 
                        numTrials2CmpResults=NUM_TRIALS_2_SAVE, withConvolution=withConvolution)
        runType[DECISION_MAKER_NAME] = agentName + "_dqn"
    
    elif agentRun == DQN2L:
        runType[DECISION_MAKER_TYPE] = "DecisionMakerExperienceReplay"
        runType[ALGO_TYPE] = "DQN_WithTarget"
        runType[DIRECTORY] = agentName + "_dqn"
        runType[PARAMS] = DQN_PARAMS(frameSize=frameSize, gameVarsSize=gameVarsSize, numActions=numActions, numTrials2Save=NUM_TRIALS_2_SAVE, 
                        numTrials2CmpResults=NUM_TRIALS_2_SAVE, withConvolution=withConvolution, layersNum=2)
        runType[DECISION_MAKER_NAME] = agentName + "_dqn"

    elif agentRun == A2C:
        accumulateHistory = True if agentRun.find(A2C_SAVE_EXP) > 0 else False
        runType[DECISION_MAKER_TYPE] = "DecisionMakerExperienceReplay"
        runType[ALGO_TYPE] = "DQN_WithTarget"
        runType[DIRECTORY] = agentName + "_dqn"
        runType[PARAMS] = A2C_PARAMS(frameSize=frameSize, gameVarsSize=gameVarsSize, numActions=numActions, numTrials2Save=NUM_TRIALS_2_SAVE, 
                                    numTrials2CmpResults=NUM_TRIALS_2_SAVE, accumulateHistory=accumulateHistory, withConvolution=withConvolution)
        runType[DECISION_MAKER_NAME] = agentName + "_dqn"

    elif agentRun == HEURISTIC:
        runType[HISTORY] = ""
        runType[RESULTS] = ""
        runType[DIRECTORY] = agentName + "_heuristic"

    elif agentRun == USER_PLAY:
        runType[HISTORY] = ""
        runType[RESULTS] = ""
        runType[DECISION_MAKER_TYPE] = "UserPlay"

    return runType







