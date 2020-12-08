from numpy import zeros, argmax
from random import random, randint
import matplotlib.pyplot as plt
from dqn import Dqn
from neural_network import NeuralNetwork
from snake_environment import SnakeEnvironment

# parametry uczenia
learningRate = 0.005 #współczynnik uczenia
gamma = 0.9 # paeametr dyskontujący do dqn --> równanie Bellmana
batchSize = 16 # wielkość wkładu do sieci
epsilon = 1 # prawdopodobieństwo podjęcia losowego ruchu przez snake'a
epsilonMultiplier = 0.995 # zmiana epsilon bo każdej grze
epochs = 2000 # liczba epok (rozegranych gier)
maxMemory = 2000

# stworzenie środowiska, modelu sieci, oraz DQN
env = SnakeEnvironment(segments = 4, waitTime = 1)
nn = NeuralNetwork(len(env.screenMap)**2+1, 4, learningRate)
model = nn.model
DQN = Dqn(gamma,maxMemory)

# nauka
epoch = 1
rewardsInEpochs = []

while epoch <= epochs:
    env.reset() # przywrócenie środowiska do początkowych ustawień
    # zapoczątkownie currentState i nextState [wartości z mapy ekrany i kierunek ruchu]
    currentState = zeros((1,len(env.screenMap)**2+1))
    for i in range(len(env.screenMap)):
        for j in range(len(env.screenMap)):
            currentState[0][len(env.screenMap)*i+j] = env.screenMap[i][j]
    currentState[0][len(env.screenMap)**2] = env.direction
    nextState = currentState
    gameOver = False
    totalReward = 0

    # pojedyncza gra/epoko
    while not gameOver:

        # ustalenie czy podejmowana akcja będzie losowa czy predykowane przez sieć (prawdopodobieństwo = epsilon)
        if random() <= epsilon:
            action = randint(0,3)
        else:
            action = argmax(model.predict(currentState))

        # podjęcie akcji i pobranie parametrów nextState, reward, gameOver oraz rysowanie ekranu gry
        nextState, reward, gameOver = env.step(action)
        env.drawScreen()

        # umieszczenie ruchu w pamięci i trening sieci na batchu pobranym z pamięci
        DQN.remember([currentState, action, reward, nextState], gameOver)
        inputs, targets = DQN.getBatch(model,batchSize)
        model.train_on_batch(inputs,targets)

        # nextState staje się currentState
        currentState = nextState
        totalReward += reward

    # print statystyki z pojedynczej epoki(gry)
    print("Epoch:\t", epoch, "Score:\t", env.score, "Moves:\t", env.moves, "Epsilon:\t", round(epsilon,5),"Total reward:\t", totalReward )

    # zmniejszenie prawdopodobieństwa losowości
    epsilon *= epsilonMultiplier
    rewardsInEpochs.append(totalReward)
    totalReward = 0
    epoch +=1

# podsumowanie całości treningu i wykres
print("BEST REWARD:", max(rewardsInEpochs))
plt.plot(rewardsInEpochs)
plt.xlabel('EPOKA')
plt.ylabel('REWARD')
plt.show()
