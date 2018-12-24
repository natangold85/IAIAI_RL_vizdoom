#!/usr/bin/python3

from __future__ import division
from __future__ import print_function


class BaseAgent:
    def __init__(self, agentName):
        self.agentName = agentName
        
        self.subAgents = {}

        self.decisionMaker = None

    def GetDecisionMaker(self):
        return self.decisionMaker

    def GetAgentByName(self, name):
        if self.agentName == name:
            return self
        
        for sa in self.subAgents.values():
            ret = sa.GetAgentByName()
            if ret != None:
                return ret

        return None


    def CreateState(self):
        pass

    def ChooseAction(self):
        pass


    def Learn(self, terminal=False, reward=0):
        pass


    def EndRun(self, reward, score, steps):
        pass  



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