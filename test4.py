import tensorflow as tf
import itertools as it
from time import time, sleep
import numpy as np
import skimage.color, skimage.transform
from tqdm import trange

from algo_decisionMaker import DecisionMakerExperienceReplay

from utils import InitVizdoom

from algo_a2c import A2C_PARAMS
from algo_orig import ORIG_PARAMS

import vizdoom as vzd

MAP_NAME = "./../ViZDoom/scenarios/simpler_basic.cfg"

FRAME_RESOLUTION = (30, 45)


learning_rate = 0.00025
# learning_rate = 0.0001
discount_factor = 0.99
numRuns = 1
epochs = 5
episodes4Epoch = 100
replay_memory_size = 10000
batch_size = 64
test_episodes_per_epoch = 100
frame_repeat = 12
episodes_to_watch = 50

def preprocess(img):
    img = skimage.transform.resize(img, FRAME_RESOLUTION)
    img = img.astype(np.float)
    return img

def Play(env, actions, dm, numEpisodes=10, sleep4Episode=0.5):
    env.close()
    env.set_window_visible(True)
    env.set_mode(vzd.Mode.ASYNC_PLAYER)
    env.init()

    validActions = list(range(len(actions)))

    for _ in range(numEpisodes):
        env.new_episode()
        while not env.is_episode_finished():
            state = preprocess(env.get_state().screen_buffer)
            best_action_index = dm.choose_action(state, validActions)
            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            

            env.set_action(actions[best_action_index])
            r = env.get_last_reward()
            for _ in range(frame_repeat):
                env.advance_action()
                r += env.get_last_reward()
            
            print(r, actions[best_action_index])

        # Sleep between episodes
        sleep(sleep4Episode)
        score = env.get_total_reward()
        print("Total score: ", score)

    env.close()
    env.set_window_visible(False)
    env.init()



def Test(env, allDm, currResults):
    print("\nTesting...")
    for dmIdx in range(len(allDm)):
        decisionMaker = allDm[dmIdx]
        test_episode = []
        test_scores = []
        for test_episode in trange(test_episodes_per_epoch, leave=False):
            env.new_episode()
            while not env.is_episode_finished():
                state = preprocess(env.get_state().screen_buffer)
                best_action_index = decisionMaker.choose_action(state, validActions)

                env.set_action(actions[best_action_index])
                for _ in range(frame_repeat):
                    env.advance_action()
            r = env.get_total_reward()
            test_scores.append(r)

        test_scores = np.array(test_scores)
        print(dmNames[dmIdx], "Results: mean: %.1f±%.1f," % (
            test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
            "max: %.1f" % test_scores.max())

        currResults[dmIdx].append(test_scores.mean())

if __name__ == '__main__':

    env = InitVizdoom(MAP_NAME)

    terminalState = np.zeros(FRAME_RESOLUTION, float)

    n = env.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    validActions = list(range(len(actions)))

    p_org = ORIG_PARAMS(FRAME_RESOLUTION, len(actions))
    
    decisionMaker_org = DecisionMakerExperienceReplay("ORIG_MODEL", p_org, "VD_Simple", "ORIG", "results", "replayHistory", "OrigRun/VD_Simple/VDAgent_ORIG", False)
    hist_org = decisionMaker_org.historyMngr

    p_a2c = A2C_PARAMS(FRAME_RESOLUTION, len(actions))
    
    decisionMaker_a2c = DecisionMakerExperienceReplay("A2C", p_org, "VD_Simple", "A2C", "results", "replayHistory", "A2CRun/VD_Simple/VDAgent_A2C", False)
    hist_a2c = decisionMaker_a2c.historyMngr


    time_start = time()

    # dmNames = ["orig model", "a2c model"]
    # allDm = [decisionMaker_org, decisionMaker_a2c]
    # allHist = [hist_org, hist_a2c]
    # allResults = [[], []]

    dmNames = ["orig model"]
    allDm = [decisionMaker_org]
    allHist = [hist_org]
    allResults = [[]]

    with tf.Session() as sess:
        decisionMaker_org.InitModel(sess, resetModel=True)
        decisionMaker_a2c.InitModel(sess, resetModel=True)

        for run in range(numRuns):
            run_start = time()
            currResults = [[], []]
            Test(env, allDm, currResults)
            for epoch in range(epochs):
                epoch_start = time()
                print("\nRun #", run + 1," Epoch #", epoch + 1,"\n-------------")
                train_scores = []
                print("\nTraining...")

                for dmIdx in range(len(allDm)):
                    decisionMaker = allDm[dmIdx]
                    hist = allHist[dmIdx]
                    
                    epoch_single_start = time()
                    for episode in range(episodes4Epoch):
                        env.new_episode()

                        while not env.is_episode_finished():
                            s = preprocess(env.get_state().screen_buffer)
                            a = decisionMaker_org.choose_action(s, validActions)
                            r = env.make_action(actions[a], frame_repeat)
                            terminal = env.is_episode_finished()
                            s_ = preprocess(env.get_state().screen_buffer) if not terminal else terminalState
                            hist.add_transition(s, a, r, s_, terminal)

                            decisionMaker.Train()
                        score = env.get_total_reward()

                        train_scores.append(score)
                        env.new_episode()

                    print(dmNames[dmIdx], "epoch real time =", (time() - epoch_single_start) / 60.0)

                Test(env, allDm, currResults)
                print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))
                print("Run Duration time: %.2f minutes" % ((time() - run_start) / 60.0))
                print("Epoch duration: %.2f minutes" % ((time() - epoch_start) / 60.0))
                

            for dmIdx in range(len(allDm)):
                allResults[dmIdx].append(currResults[dmIdx])
                allDm[dmIdx].ResetAllData()

        print("\nPlaying...")
        Play(env, actions, allDm[0], numEpisodes=100, sleep4Episode=2)

    print("\n\nall results:")
    for dmIdx in range(len(dmNames)):
        print("\n\n", dmNames[dmIdx])
        for run in range(numRuns):
            print(allResults[dmIdx][run])
    
    from utils_plot import PlotMeanWithInterval
    import matplotlib.pyplot as plt

    epochsArr = np.arange(epochs + 1)
    for results in allResults:
        PlotMeanWithInterval(epochsArr, np.average(results, axis=0), np.std(results, axis=0))

    plt.legend(dmNames)
    plt.show()

    env.close()