from snake_environment import SnakeEnvironment
from numpy import argmax
from keras.models import load_model

waitTime = 0  # czas między akcjami
nSegments = 4  # wielkość planszy
nAttempts = 1000  # ilość gier w teście
model_name = 'trained_model.h5'  # nazwa modelu do testów
maxMoves = 100  # maksymalna liczba ruchów bez punktu


# zmienne używate do testowania
scoreSum = 0
winSum = 0
best = 0
movesNoPoint = 0

env = SnakeEnvironment(segments=nSegments, waitTime=waitTime, segmentSize=40)
model = load_model(model_name)

for i in range(nAttempts):
    env.reset()
    currentState = env.newState(False)
    nextState = currentState
    gameOver = False
    while not gameOver:
        action = argmax(model.predict(currentState))
        nextState, reward, gameOver, win = env.step(action)
        if reward == -0.02:
            movesNoPoint += 1
        else:
            movesNoPoint = 0
        if win:
            winSum += 1
        env.drawScreen()
        currentState = nextState
        if movesNoPoint > maxMoves:
            gameOver = True
            movesNoPoint = 0
        if env.score > best:
            best = env.score
    scoreSum += env.score
    print("Score:\t", env.score, "Moves:\t", env.moves)

print("Mean score in", nAttempts, "attempts:", scoreSum/nAttempts, ", wins:",
      winSum, ", best score:", best)
