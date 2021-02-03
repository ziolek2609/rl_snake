# Reinforcement Learning Snake


The aim of the project is to create an artificial neural network capable of playing a snake. I used Reinforcement Learning method to learn the network. 

## Game rules:
- snake can only move forward, left or right. Backwards moves are blocked,
- you win the game when the snake segments are on *every* segment of the gameboard,
- you lose if snake hit the border of the gameboard or eat itself.

## Python packeges used:
 - [pygame](https://www.pygame.org/) v. 2.0.0
 - [tensorflow](https://www.tensorflow.org/) v. 1.14.0
 - [numpy](https://numpy.org/) v. 1.19.3
 - [matplotlib](https://matplotlib.org/) v. 3.3.3

## Repository contains several files:
- main.py -- it is the main file of the project, it is used to learn neural network to play,
- neural_network.py -- contains architecture of the neural network
- dqn.py -- contains agent's memory ,
- snake_environment.py -- contains the game environment, there are implemented all the game rules,
- test.py -- it is used for testing already trained neural network model,
- snke_condaenv.yml -- this is the Anaconda environment which is usefull to run the hole project, using appropriate versions of all required packages
- trained_model.py -- already trained model.

## What you can do here?
- just play the game by running snake_environment.py file :snake:
- train your own model by main.py :memo:

     You can use default parameters or you can change it. I had used my deafult seatings and you can see the results below. On the chart you can see how my model has learnt and the gif shows model can play on 4x4 board (of course you can check how it will manage on bigger boards). If you use deafult parameters you should get similar results to mine

     ![chart](./images/chart.jpg) ![4x4](./images/4x4.gif)

- you can test trained model by test.py (you can test your own model or test mine) :clipboard:

## How the snake can learn?
Reinforcement Learning is the machine learning method, in which the agent (snake) interacts with the environment (gameboard with snacks), based on the informations it collects about the environment (initially by randomly executing an action), selects an action (dirrection of the next move) in the environment and is rewarded or punished depending on the success of the action. Snake is rewarded for collecting a snack and punished for loosing the game. There is also leaving penalty to to prevent unsuspecting movements.


On input of the neural network there is the vision of the snake. Snake have vision in 8 different directions: North, North-East, East, South-East, South, South-West, West and North-West. On each direction snake is 'looks' for three data: distance to the wall, distance to the potential snack and distance to the potetial snake-segment. In that way, there are 24 neurons on input to neural network. On output we have 4 neurons corresponding to 4 movements. There are 2 hidden layers, 18 neurons each. On hidden layers there is relu activation function and softmax on output layer.

![directions](./images/directions.jpg) ![nn](./images/nn_schema.jpg)






