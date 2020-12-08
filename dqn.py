from numpy import zeros, max
from numpy.random import randint


class Dqn():

    def __init__(self,gamma,maxMemory):
        self.gamma = gamma # discount (czynnik dyskontujący) -> równanie Bellman'a
        self.memory = [] # lista pamięci
        self.maxMemory = maxMemory

    # dołączenie nowego przypadku do pamięci
    def remember(self, transition, gameOver):
        self.memory.append([transition, gameOver]) # transition = [currentState, action, reward, nextState]
        if len(self.memory)>self.maxMemory:
            del self.memory[0]

    # uzyskanie batchu do sieci z pamięci
    def getBatch(self,model,batchSize):
        nInputs = self.memory[0][0][0].shape[1] # wymiar inputu do sieci (currentState)
        nOutputs = model.output_shape[-1] # liczba neuronów na wyjściu z ostatniej warstwy modelu sieci (wymiar outputu z sieci)

        # zapoczątkowanie batchu składającego się z inputu i targetu -- same zera
        inputs = zeros((min(batchSize,len(self.memory)), nInputs)) # input do sieci
        targets = zeros((min(batchSize,len(self.memory)), nOutputs)) # targety dla sieci


        randomMemory = randint(0,len(self.memory), size = min(batchSize,len(self.memory))) # losowy wybor indeksów z pamięci w ilości batchSize
        for i in range(len(randomMemory)):
            # pobranie z pamięci przypadków o wylosowanych indeksach
            currentState, action, reward, nextState = self.memory[randomMemory[i]][0]
            gameOver = self.memory[randomMemory[i]][1]

            # przypisanie pobranych z pamięci wartości do inputs i targets
            inputs[i] = currentState
            targets[i] = model.predict(currentState)[0] # predykcja modelu o możliwych reward z tego currentState

            # aktualizacja targetów
            if gameOver:
                # nagroda zostaje niezmieniona
                targets[i][action] = reward
            else:
                # aktualna nagroda zostaje wzmocniona o najwyższą z możliwych potejncjalnych nagród z przyszłego stanu
                targets[i][action] = reward + self.gamma*max(model.predict(nextState)[0])

        return inputs, targets
