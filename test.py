from snake_environment import SnakeEnvironment
from numpy import argmax
from keras.models import load_model

waitTime = 0
nSegments = 10
nAttempts = 1000
model_name = 'przetrenowany.h5'
scoreSum = 0
maxMoves = 100
winSum = 0
best = 0

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
        if win:
            winSum += 1
        env.drawScreen()
        currentState = nextState
        if (env.moves > maxMoves and env.score < 2) or env.moves > 150:
            gameOver = True
        if env.score > best:
            best = env.score
    scoreSum += env.score
    print("Score:\t", env.score, "Moves:\t", env.moves)

print("Mean score in", nAttempts, "attempts:", scoreSum/nAttempts, ", wins:",
      winSum, ", best score:", best)
