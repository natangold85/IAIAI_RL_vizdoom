#!/bin/sh
python ./runVDAgent.py --runDir=A2CRun --resetModel=True --trainAgent=VDAgent --numGameThreads=8 --numEpisodes=3000
python ./runVDAgent.py --runDir=DQNRun --resetModel=True --trainAgent=VDAgent --numGameThreads=8 --numEpisodes=3000
python ./runVDAgent.py --runDir=DQNRun4Transition --resetModel=True --trainAgent=VDAgent --numGameThreads=8 --numEpisodes=3000
python ./runVDAgent.py --runDir=OrigRun --resetModel=True --trainAgent=VDAgent --numGameThreads=8 --numEpisodes=3000
