from numpy import argmax
from random import random, randint
import matplotlib.pyplot as plt
from dqn import Dqn
from neural_network import NeuralNetwork
from snake_environment import SnakeEnvironment

# PARAMETRY UCZENIA
learningRate = 0.01  # współczynnik uczenia
gamma = 0.9  # parametr dyskontujący do dqn
batchSize = 128  # wielkość wkładu do sieci
epsilon = 1  # prawdopodobieństwo podjęcia losowego ruchu przez snake'a w danej epoce
epsilonMultiplier = 0.999  # zmiana epsilon bo każdej grze
epochs = 100000  # liczba epok (rozegranych gier)
maxMemory = 15000 # pojemność pamięci

# STWORZENIE ŚRODOWISKA, MODELU SIECI ORAZ DQN
env = SnakeEnvironment(waitTime = 1, segments = 4)
nn = NeuralNetwork(24, 4, learningRate)
model = nn.model
DQN = Dqn(gamma, maxMemory)

# NAUKA/TRENING
epoch = 1
scoreInEpochs = []
meanScore = 0
bestScore = [0,0]
fullMemoryEpoch = 0
win = False

while not win:
    # NOWA GRA -- reset środowiska, i początkowy input
    env.reset()
    currentState = env.newState(False)
    nextState = currentState
    gameOver = False

    while not gameOver:
        # ustalenie czy podejmowana akcja będzie losowa czy predykowane przez sieć (prawdopodobieństwo = epsilon)
        if random() <= epsilon:
            action = randint(0, 3)
        else:
            action = argmax(model.predict(currentState))

        # podjęcie akcji
        nextState, reward, gameOver, win = env.step(action)
        env.drawScreen()

        # umieszczenie ruchu w pamięci i trening sieci na pobranym batchu
        DQN.remember([currentState, action, reward, nextState], gameOver)
        inputs, targets = DQN.getBatch(model, batchSize)
        model.train_on_batch(inputs, targets)

        currentState = nextState

    # zmniejszenie prawdopodobieństwa losowości
    if epsilon > 0.05:
        epsilon *= epsilonMultiplier
    else:
        epsilon = 0.05

    # statystyki z pojedynczej gry
    if env.score > bestScore[0]:
        bestScore = [env.score,env.moves]
        print("NEW BEST SCORE:",bestScore[0],"points in", bestScore[1], "moves")
    print("Epoch:\t", epoch, "Score:\t", env.score, "Moves:\t", env.moves, "Epsilon:\t", round(epsilon, 5), "Best score:", bestScore[0])

    # co 100 epok -- statystyka ze 100 epok
    meanScore+=env.score
    if epoch % 100 == 0:
        win = True
        scoreInEpochs.append(meanScore/100)
        print("Mean in last 100 epochs:", meanScore/100, "All time best score:", bestScore, "Actual Memory Capacity:",len(DQN.memory))
        plt.plot(scoreInEpochs)
        plt.xlabel('Epoki*100')
        plt.ylabel('Średni wynik w 100 epokach')
        plt.show()
        meanScore = 0

    # zapisanie epoki, w której pamięć się zapełniła
    if len(DQN.memory) < maxMemory:
        fullMemoryEpoch = epoch+1

    epoch += 1
model.save('model.h5')
