#!/usr/bin/python3

import logging
import traceback

import os
import sys
import threading

import tensorflow as tf

from absl import app
from absl import flags

from agent_super import SuperAgent

from paramsCalibration import GeneticProgrammingGeneration
from paramsCalibration import ReadGPFitness
from paramsCalibration import TrainSingleGP
from paramsCalibration import GetPopulationDict
from paramsCalibration import SetPopulationTrained

# general params
flags.DEFINE_string("act", "run", "what to  act: options =[run, check, copyNN, gp, play]") 
flags.DEFINE_string("device", "gpu", "Which device to run nn on.")
flags.DEFINE_string("runDir", "none", "directory of the decision maker (should contain config file name config.txt)")

# run params
flags.DEFINE_string("playAgent", "none", "Which agent to train.")
flags.DEFINE_string("trainAgent", "none", "Which agent to train.")
flags.DEFINE_string("testAgent", "none", "Which agent to test.")
flags.DEFINE_string("map", "simpler_basic", "Which map to run.")
flags.DEFINE_string("numRuns", "1", "num of game threads.")
flags.DEFINE_string("numEpisodes", "none", "num of episodes agent to run.")
flags.DEFINE_string("resetModel", "False", "if to reset data(dm params, history and results)")

# play params
flags.DEFINE_string("copy2Play", "none", "which dm copy to play")

# params for genetic programming
flags.DEFINE_string("gpAct", "trainPopulation", "train population or test single individual")
flags.DEFINE_string("populationSize", "20", "population size")
flags.DEFINE_string("populationInstances", "2", "num instances for each individual")
flags.DEFINE_string("numGenerations", "20", "num generations to run")
flags.DEFINE_string("populationIdx", "none", "which idx of population to test or train")
flags.DEFINE_string("populationInstanceIdx", "none", "which instance of individual to test or train")

#check params
flags.DEFINE_string("copy2Check", "none", "which dm copy to check")

MAP_PATH = "./../ViZDoom/scenarios/"
MAP_END_FNAME = ".cfg"

def runAgent(sess, agent, numEpisodes):
    with sess.as_default(), sess.graph.as_default():  
        if numEpisodes == None:
            while True:
                agent.RunEpisode()
        else:
            for _ in range(numEpisodes):
                agent.RunEpisode()
 
def startRun(configDict=None, copy2Run=None, threadName="Thread", numEpisodes=None):
    # if configDict is not argument read it from file
    if configDict == None:
        configDictOrg = eval(open("./" + flags.FLAGS.runDir + "/config.txt", "r+").read())
        configDict = configDictOrg.copy()
        configDict["directory"] = flags.FLAGS.runDir
    
    trainList = flags.FLAGS.trainAgent
    trainList = trainList.split(",")

    testList = flags.FLAGS.testAgent
    testList = testList.split(",")
        
    training = trainList != ["none"]

    if flags.FLAGS.playAgent == "none":
        playList = trainList if trainList != ["none"] else testList
    else:
        playList = flags.FLAGS.playAgent
        playList = playList.split(",")

    print("\n\n\nplay list =", playList)        
    print("train list =", trainList)
    print("test list =", testList, "\n\n\n")


    if numEpisodes == None:
        if flags.FLAGS.numEpisodes == "none":
            numEpisodes = None
        else:
            numEpisodes = int(flags.FLAGS.numEpisodes)

    if "sharedDM" in configDict.keys():
        sharedDM = configDict["sharedDM"]
    else:
        sharedDM = True


    numRuns = int(flags.FLAGS.numRuns)
    if "mapName" in configDict.keys():
        mapFullName = MAP_PATH + configDict["mapName"] + MAP_END_FNAME
    else:
        mapFullName = MAP_PATH + flags.FLAGS.map + MAP_END_FNAME
    
    configDict["mapPath"] = mapFullName
    threads = []
    agents = []
    for i in range(numRuns):
        print("\n\n\n init dm #", i, "\n\n\n")
        agent = SuperAgent(configDict, None, trainList, playList, dmCopy=i)
        agents.append(agent)

    with tf.Session() as sess:
        for i in range(numRuns):
            agent = agents[i]
            decisionMaker = agent.GetDecisionMaker()
            # create savers
            resetModel = eval(flags.FLAGS.resetModel)
            decisionMaker.InitModel(sess, resetModel)

            threadArgs = (sess, agent, numEpisodes)
            t = threading.Thread(target=runAgent, args=threadArgs, daemon=True)
            t.setName( "Game" + threadName + "_" + str(i))
            threads.append(t)
        
        for t in threads:   
            t.start()

        for t in threads:
            t.join()



def run_gp_threads(populationSize, numInstances, gpActArg):
    numThreadsRunning = 8

    threadsB4Run = []  
    for idx in range(populationSize):
        for instance in range(numInstances):
            t = threading.Thread(target=run_script, args=(gpActArg, idx, instance))
            threadsB4Run.append(t)
    
    runningThreads = []

    while len(threadsB4Run) > 0 or len(runningThreads) > 0:
        if len(runningThreads) < numThreadsRunning and len(threadsB4Run) > 0:
            t = threadsB4Run.pop()
            t.start()
            runningThreads.append(t)

        for t in runningThreads:
            if not t.isAlive():
                runningThreads.remove(t)


            
def gp_train(): 
    print("\n\n\ntrain gp\n\n\n")

    numGenerations = int(flags.FLAGS.numGenerations)
    populationSize = int(flags.FLAGS.populationSize)
    numPopulationInstances = int(flags.FLAGS.populationInstances)
    
    configDict = eval(open("./" + flags.FLAGS.runDir + "/config.txt", "r+").read())
    configDict["directory"] = flags.FLAGS.runDir

    trainAgent = flags.FLAGS.testAgent

    if trainAgent == "VDAgent":
        runType = GetRunTypeVDAgent(configDict)

    populationDict, currGeneration = GetPopulationDict(configDict["directory"])
    
    while currGeneration != numGenerations:

        # advance to next generation (in case current population has fitness or this is the first population)
        if "fitness" in populationDict or populationDict == {}:
            print("\n\n\ncreate population generation #", currGeneration, "\n\n\n")
            GeneticProgrammingGeneration(populationSize, numPopulationInstances, configDict, runType)
            populationDict, currGeneration = GetPopulationDict(configDict["directory"])

        # train generation (if not trained)
        if "trained" not in populationDict:
            print("\n\n\ntrain population generation #", currGeneration, "\n\n\n")
            run_gp_threads(populationSize, numPopulationInstances, "trainSingle")
            SetPopulationTrained(configDict["directory"])

        # test generation
        if "fitness" not in populationDict:
            print("\n\n\ntest population generation #", currGeneration, "\n\n\n")
            run_gp_threads(populationSize, numPopulationInstances, "testSingle")
            # calculate generation fitness
            ReadGPFitness(configDict, runType)
            populationDict, currGeneration = GetPopulationDict(configDict["directory"])



def getGP_Params(population, params2Calibrate, populationIdx, instanceIdx):
    individual2Run = population[populationIdx]

    dirsCopy2Run = individual2Run[0][instanceIdx]
    paramDict = {}
    for i in range(len(params2Calibrate)):
        paramDict[params2Calibrate[i]] = individual2Run[1][i]    
    
    return dirsCopy2Run, paramDict

def gp_train_single():  
    populationIdx = int(flags.FLAGS.populationIdx)
    instance2Train  = int(flags.FLAGS.populationInstanceIdx)

    configDict = eval(open("./" + flags.FLAGS.runDir + "/config.txt", "r+").read())
    configDict["directory"] = flags.FLAGS.runDir

    gpPopulation = eval(open("./" + flags.FLAGS.runDir + "/gp_population.txt", "r+").read())

    params2Calibrate = configDict["params2Calibrate"]

    dirsCopy2Run, paramDict = getGP_Params(gpPopulation["population"], params2Calibrate, populationIdx, instance2Train)
    
    print("\n\n\ntrain single", populationIdx, "instance =", instance2Train)
    print(populationIdx, ": dir copy =", dirsCopy2Run, "params dict =", paramDict, "\n\n\n")

    configDict["hyperParams"] = paramDict
    
    threading.current_thread().setName("TrainSingle_" + str(populationIdx) + "_" + str(instance2Train))
    if "hundredsTrainEpisodes" in paramDict:
        numEpisodes = paramDict["hundredsTrainEpisodes"] * 100
    else:
        numEpisodes = None

    startRun(configDict, dirsCopy2Run, threadName="train_pop#" + str(populationIdx) + "_" + str(instance2Train), numEpisodes=numEpisodes)


def gp_test_single(): 
    populationIdx = int(flags.FLAGS.populationIdx)
    instance2Train  = int(flags.FLAGS.populationInstanceIdx)


    configDictOrg = eval(open("./" + flags.FLAGS.runDir + "/config.txt", "r+").read())
    configDict = configDictOrg.copy()
    configDict["directory"] = flags.FLAGS.runDir

    gpPopulation = eval(open("./" + flags.FLAGS.runDir + "/gp_population.txt", "r+").read())
    params2Calibrate = configDict["params2Calibrate"]

    dirsCopy2Run, paramDict = getGP_Params(gpPopulation["population"], params2Calibrate, populationIdx, instance2Train) 
    
    print("test single", populationIdx, "instance =", instance2Train, "dircopyNum =", dirsCopy2Run, "\n\n\n") 
    
    configDict["hyperParams"] = paramDict
    configDict["numGeneration"] = gpPopulation["numGeneration"]

    startRun(configDict, dirsCopy2Run, threadName="test_pop#" + str(populationIdx) + "_" + str(instance2Train))

def Play():
    configDictOrg = eval(open("./" + flags.FLAGS.runDir + "/config.txt", "r+").read())
    configDict = configDictOrg.copy()
    configDict["directory"] = flags.FLAGS.runDir    

    testAgent = flags.FLAGS.testAgent

    numEpisodes = int(flags.FLAGS.numEpisodes)
    
    if "mapName" in configDict.keys():
        mapFullName = MAP_PATH + configDict["mapName"] + MAP_END_FNAME
    else:
        mapFullName = MAP_PATH + flags.FLAGS.map + MAP_END_FNAME
    configDict["mapPath"] = mapFullName

    copy2Play = None if flags.FLAGS.copy2Play == "none" else int(flags.FLAGS.copy2Play)
    agent = SuperAgent(configDict, None, trainList=[], playList=[testAgent], dmCopy=copy2Play, playMode=True)
    decisionMaker = agent.GetDecisionMaker()

    with tf.Session() as sess:
        decisionMaker.InitModel(sess, resetModel=False)
        for _ in range(numEpisodes):
            agent.PlayEpisode()

        agent.Close()

def run_script(gpActArg, populationIdx, instanceIdx):     
    cmd = ' '.join(sys.argv)
    cmd.replace(".\\", "")                                                        
    os.system('python {}'.format(cmd + " --gpAct=" + gpActArg + " --populationIdx=" + str(populationIdx) + " --populationInstanceIdx=" + str(instanceIdx)))    

def check_model():
    configDictOrg = eval(open("./" + flags.FLAGS.runDir + "/config.txt", "r+").read())
    configDict = configDictOrg.copy()
    configDict["directory"] = flags.FLAGS.runDir

    mapFullName = MAP_PATH + flags.FLAGS.map + MAP_END_FNAME
    configDict["mapPath"] = mapFullName

    trainAgent = flags.FLAGS.trainAgent
    copy2Check = None if flags.FLAGS.copy2Check == "none" else int(flags.FLAGS.copy2Check)

    agent = VDAgent(configDict, decisionMaker=None, dmCopy=copy2Check)
    decisionMaker = agent.GetDecisionMaker()
    with tf.Session() as sess:
        decisionMaker.InitModel(sess, resetModel=False)
        s = decisionMaker.DrawStateFromHist(realState=True)
        print("\nframe layer =", s[0])
        print("\n\ngame vars =", s[1])

        feedDict = {decisionMaker.decisionMaker.frameLayer: s[0]}

        conv1, conv2, convOut = sess.run([decisionMaker.decisionMaker.conv1, decisionMaker.decisionMaker.conv2, decisionMaker.decisionMaker.convOut], feed_dict=feedDict)
        print("\n\nconvOut =", convOut)
        print("\n\nshapes: \ninput frame =", s[0].shape, "conv1 =", conv1.shape, "conv2 =", conv2.shape, "convOut =", convOut.shape)
      


def main(argv):    
    if flags.FLAGS.device == "cpu":
        # run from cpu
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


    modelDir = "./" + flags.FLAGS.runDir
    histCmdFName = modelDir + "/cmdHistory.txt"
    if not os.path.isdir(modelDir):
        os.makedirs(modelDir)

    if flags.FLAGS.resetModel == "True":
        open(histCmdFName, "w+").write(str(sys.argv))
    else:
        open(histCmdFName, "a+").write("\n\n" + str(sys.argv))

    if flags.FLAGS.act == "run":
        startRun()
    elif flags.FLAGS.act == "play":
        Play()
    elif flags.FLAGS.act == "check":
        check_model()
    elif flags.FLAGS.act == "gp":
        if flags.FLAGS.gpAct == "trainPopulation":
            gp_train()
        elif flags.FLAGS.gpAct == "trainSingle":
            gp_train_single()
        elif flags.FLAGS.gpAct == "testSingle":
            gp_test_single()   

if __name__ == '__main__':
    print('Starting...')
    app.run(main)
