# Reinforcement Learning Snake :video_game::snake::brain:

The aim of the project is to create an artificial neural network capable of playing the Snake game. A Reinforcement Learning Method was used to learn the network.

## Game rules:
- snake can only move forward, left or right. Backwards moves are blocked,
- you win the game when the snake segments are on **every** segment of the gameboard,
- you lose if snake hit the border of the gameboard or eats itself.

## Python packeges used:
 - [pygame](https://www.pygame.org/) v. 2.0.0 :video_game:,
 - [tensorflow](https://www.tensorflow.org/) v. 1.14.0 :brain:,
 - [numpy](https://numpy.org/) v. 1.19.3 :1234:,
 - [matplotlib](https://matplotlib.org/) v. 3.3.3 :chart:.

## Repository contains several files:
- main.py -- it is the main file of the project, it is used to learn neural network to play,
- neural_network.py -- contains the architecture of the neural network
- dqn.py -- contains agent's memory,
- snake_environment.py -- contains the game environment, there are implemented all the game rules,
- test.py -- it is used for testing already trained neural network model,
- snke_condaenv.yml -- this is the Anaconda environment containing appropriate versions of all required packages,
- trained_model.py -- already trained model.

## What can you do here?
- just play the game by running snake_environment.py file :snake:
- train your own model by main.py (you can use my deafult parameters or set your own) :memo:
- you can test trained model by running test.py (you can test your own model or test mine) :clipboard:

## How the snake can learn?
Reinforcement Learning is the machine learning method, in which the agent (snake) interacts with the environment (gameboard with snacks). Than, based on the informations it collects about the environment (initially by randomly executing an action), selects an action (dirrection of the next move) in the environment and is rewarded or punished depending on the success of the action. Snake is rewarded for collecting a snack and punished for loosing the game. There is also leaving penalty to prevent unsuspecting movements.


On the input of the neural network there is the vision of the snake. Snake have vision in 8 different directions: North (k1), North-East (k2), East (k3), South-East (k4), South (k5), South-West (k6), West (k7) and North-West (k8). On each direction snake 'looks' for three data: distance to the wall, distance to the potential snack and distance to the potential snake-segment. In that way, there are 24 neurons on input to neural network. On the output we have 4 neurons corresponding to 4 movements (up, down, left, right). There are 2 hidden layers, 18 neurons each. On the hidden layers there is relu activation function and softmax activation on the output layer.

![directions](./images/directions.jpg) ![nn](./images/nn_schema.jpg)

## Results
I have trained my model on 4x4 gameboard (I wanted this not to take a lot of time) to the first win during training (score 13, because intial snake has 3 segments). The rewards were:
- positive: 1.0 point
- negative: -2.0 points
- livingPenalty: -0.02 point

During testing snake was able to score an average of 4.4 points on the 4x4 board, with best score 13 points. The following chart shows how model has trained (mean score from last 100 games) and the gif shows how snake behave on 4x4 board (of course you can test how it will behave on bigger boards).

![chart](./images/chart.jpg) ![4x4](./images/4x4gf.gif)

you should get results comparable to mine if you use the default settings during training the neural network model:smile:





