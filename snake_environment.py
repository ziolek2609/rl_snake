from random import randint
from numpy import zeros
import pygame

class SnakeEnvironment():
    # rozpoczęcie z ustawieniami startowymi
    def __init__(self, waitTime = 100, segments = 10, grid = 1, segmentSize = 30, livingPenalty = -0.02, posReward = 1, negReward = -2):
        # stałe parametry środowiska
        self.SEGMENTS = segments #ilość segmentów na jednym boku ekranu gry
        self.GRID = grid # wielkość granicy między segmentami
        self.SEGMENTSIZE = segmentSize # wielkość jednego segmentu
        self.SCREENSIZE = self.SEGMENTS*self.SEGMENTSIZE + (self.SEGMENTS-1)*self.GRID # wielkość boku ekranu gry wyliczana na podstawie pozostałych parametrów
        self.LIVINGPENALTY = livingPenalty # kara za niezjadanie jabłka
        self.POSREWARD = posReward # nagoda za zjedzenie jabłka
        self.NEGREWARD = negReward # kara za przegraną (zjedzenie samego siebie, uderzenie w ścianę)
        self.WAITTIME = waitTime # czas pomiędzy akcjami
        self.SCREEN = pygame.display.set_mode((self.SCREENSIZE,self.SCREENSIZE)) # ekran gry

        self.reset()

    # przywrócenie zmiennych gry do ustawień początkowych
    def reset(self):
        self.direction = 1 # aktualny kierunek ruchu węża (0-up,1-right,2-down,3-left)
        self.moves = 0 # liczba wykonanych ruchów
        self.score = 0 # liczba zdobytych punktów
        self.snakeLoc = [] # lista lokalizacji segmentów węża
        # mapa ekranu, na której 0 -> puste pole, 0.5 -> segment snake'a, 1 -> jabłko
        self.screenMap = zeros((self.SEGMENTS,self.SEGMENTS))
        # zapoczątkowanie węża (3 segmenty) i jabłka
        for i in range(3):
            self.snakeLoc.append([2-i,1])
            self.screenMap[1,2-i] = 0.5
        self.appleLoc = self.createApple()
        self.drawScreen()

    # tworzenie jabłko w wolnej lokacji na screenMap
    def createApple(self):
        x = randint(0,self.SEGMENTS-1)
        y = randint(0,self.SEGMENTS-1)
        while self.screenMap[x][y] == 0.5 or self.screenMap[x][y] == 1:
            x = randint(0,self.SEGMENTS-1)
            y = randint(0,self.SEGMENTS-1)
        self.screenMap[x][y] = 1
        return x,y

    # rysowanie ekranu gry z elementami
    def drawScreen(self):
        self.SCREEN.fill((0,0,0))
        for i in range(self.SEGMENTS):
            for j in range(self.SEGMENTS):
                if self.screenMap[j][i] == 0.5: # segmenty węża
                    pygame.draw.rect(self.SCREEN, (0, 255, 0), (i*(self.SEGMENTSIZE+self.GRID),j*(self.SEGMENTSIZE+self.GRID), self.SEGMENTSIZE,self.SEGMENTSIZE))
                elif self.screenMap[j][i] == 1: # jabłko
                    pygame.draw.rect(self.SCREEN, (255, 0, 0), (i*(self.SEGMENTSIZE+self.GRID),j*(self.SEGMENTSIZE+self.GRID), self.SEGMENTSIZE,self.SEGMENTSIZE))
        pygame.display.flip()

    # pojedyncze poruszenie węża
    def moveSnake(self, nextLoc):
        # aktualizacja snakeLoc i appleLoc
        self.snakeLoc.insert(0,nextLoc)
        if nextLoc != (self.appleLoc[1],self.appleLoc[0]): # niezjedzenie jabłka -> usunięcie ostatniego segmentu węża
            self.snakeLoc.pop(len(self.snakeLoc)-1)
        else: # zjedzenie jabłka -> nowe jabłko(o ile not win) i score+1
            self.score +=1
            if self.score < self.SEGMENTS**2-3:
                self.appleLoc = self.createApple()

        # aktualizacja screenMap
        self.screenMap = zeros((self.SEGMENTS,self.SEGMENTS))
        self.screenMap[self.appleLoc[0],self.appleLoc[1]] = 1
        for i in self.snakeLoc:
            self.screenMap[i[1],i[0]] = 0.5

    # obliczenie newState, który jest inputem do sieci neuronowej
    def newState(self, wallCrush):
        # input to 24 liczby: 8 kierunków patrzenia(N,NE,E,SE,S,SW,W,NW)*3 dystanse[appleDistance, snakeDistance, wallDistance]
        # jeśli jabłko bądź segmet węża nie jest widoczny to domyślna wartość 0
        # jeśli wąż uderzy w ściane (wallCrush==True) -- odległość od ściany na danym kierunku to 0
        # na kierunkach NE,SE,SW,NW wartość odległości mówi o ilości segmetów pokonywanych na skos,
        # więc jeśli jabłko jest na kolejnym polu planszy patrząc na skos to wartość wynosi 1
        newState = zeros((1,24))

        # NORTH
        for i in range(0,self.snakeLoc[0][1]):
            if self.screenMap[i][self.snakeLoc[0][0]] == 1:
                newState[0][0] = self.snakeLoc[0][1]-i
            if self.screenMap[i][self.snakeLoc[0][0]] == 0.5:
                newState[0][1] = self.snakeLoc[0][1]-i
        newState[0][2] = self.snakeLoc[0][1]+1

        # NORTH-EAST
        snakeSeen = False
        for i,j in zip(range(self.snakeLoc[0][1]-1,-1,-1), range(self.snakeLoc[0][0]+1,self.SEGMENTS)):
            if self.screenMap[i][j] == 1:
                newState[0][3] = j-self.snakeLoc[0][0]
            if self.screenMap[i][j] == 0.5 and snakeSeen == False:
                newState[0][4] = j-self.snakeLoc[0][0]
                snakeSeen = True
        newState[0][5] = min(self.snakeLoc[0][1]+1,self.SEGMENTS-self.snakeLoc[0][0])

        # EAST
        snakeSeen = False
        for i in range(self.snakeLoc[0][0]+1,self.SEGMENTS):
            if self.screenMap[self.snakeLoc[0][1]][i] == 1:
                newState[0][6] = i-self.snakeLoc[0][0]
            if self.screenMap[self.snakeLoc[0][1]][i] == 0.5 and snakeSeen == False:
                newState[0][7] = i-self.snakeLoc[0][0]
                snakeSeen = True
        newState[0][8] = self.SEGMENTS-self.snakeLoc[0][0]

        # SOUTH-EAST
        snakeSeen = False
        for i,j in zip(range(self.snakeLoc[0][1]+1,self.SEGMENTS), range(self.snakeLoc[0][0]+1,self.SEGMENTS)):
            if self.screenMap[i][j] == 1:
                newState[0][9] = j-self.snakeLoc[0][0]
            if self.screenMap[i][j] == 0.5 == 0.5 and snakeSeen == False:
                newState[0][10] = j-self.snakeLoc[0][0]
                snakeSeen = True
        newState[0][11] = min(self.SEGMENTS-self.snakeLoc[0][0],self.SEGMENTS - self.snakeLoc[0][1])

        # SOUTH
        snakeSeen = False
        for i in range(self.snakeLoc[0][1]+1,self.SEGMENTS):
            if self.screenMap[i][self.snakeLoc[0][0]] == 1:
                newState[0][12] = i-self.snakeLoc[0][1]
            if self.screenMap[i][self.snakeLoc[0][0]] == 0.5 and snakeSeen == False:
                newState[0][13] = i-self.snakeLoc[0][1]
                snakeSeen = True
        newState[0][14] = self.SEGMENTS - self.snakeLoc[0][1]

        # SOUTH-WEST
        snakeSeen = False
        for i,j in zip(range(self.snakeLoc[0][1]+1,self.SEGMENTS), range(self.snakeLoc[0][0]-1,-1,-1)):
            if self.screenMap[i][j] == 1:
                newState[0][15] = i-self.snakeLoc[0][1]
            if self.screenMap[i][j] == 0.5 and snakeSeen == False:
                newState[0][16] = i-self.snakeLoc[0][1]
                snakeSeen = True
        newState[0][17] = min(self.snakeLoc[0][0]+1, self.SEGMENTS - self.snakeLoc[0][1])

        # WEST
        for i in range(0,self.snakeLoc[0][0]):
            if self.screenMap[self.snakeLoc[0][1]][i] == 1:
                newState[0][18] = self.snakeLoc[0][0] - i
            if self.screenMap[self.snakeLoc[0][1]][i] == 0.5:
                newState[0][19] = self.snakeLoc[0][0] - i
        newState[0][20] = self.snakeLoc[0][0]+1

        # NORTH-WEST
        for i,j in zip(range(self.snakeLoc[0][1]-1,-1,-1), range(self.snakeLoc[0][0]-1,-1,-1)):
            if self.screenMap[i][j] == 1:
                newState[0][21] = self.snakeLoc[0][0] - j
            if self.screenMap[i][j] == 0.5:
                newState[0][22] = self.snakeLoc[0][0] - j
        newState[0][23] = min(self.snakeLoc[0][0]+1, self.snakeLoc[0][1]+1)

        if wallCrush:
            if self.direction == 0:
                newState[0][2] = 0
            elif self.direction == 1:
                newState[0][8] = 0
            elif self.direction == 2:
                newState[0][14] = 0
            elif self.direction == 3:
                newState[0][20] = 0

        return newState

    # krok w grze i przyszłej nauce
    def step(self,action):
        self.moves +=1
        gameOver = False
        wallCrush = False
        win = False
        reward = self.LIVINGPENALTY # domyślna nagroda to livingPenalty

        # wykluczneie możliwości skrętu o 180 stopni
        if action == 1 and self.direction == 3:
            action = 3
        elif action == 2 and self.direction == 0:
            action = 0
        elif action == 3 and self.direction == 1:
            action = 1
        elif action == 0 and self.direction == 2:
            action = 2

        # krok w grze -- możliwe scenariusze -- ustalenie nagrody i stanów: gameOver, win oraz wallCrush
        if action == 0: # ruch w górę
            if self.snakeLoc[0][1]>0:
                if self.screenMap[self.snakeLoc[0][1]-1][self.snakeLoc[0][0]] == 0.5: # zjedzenie samego siebie -> PRZEGRANA
                    gameOver = True
                    reward = self.NEGREWARD
                elif self.screenMap[self.snakeLoc[0][1]-1][self.snakeLoc[0][0]] == 1: # zjedzenie jabłka -> POZYTYWNA NAGRODA
                    reward = self.POSREWARD
                self.moveSnake((self.snakeLoc[0][0],self.snakeLoc[0][1]-1)) # przeunięcie węża
            else: #wyjście poza mapę -> PRZEGRANA
                gameOver = True
                wallCrush = True
                reward = self.NEGREWARD
        elif action == 1: # ruch w prawo
            if self.snakeLoc[0][0]<self.SEGMENTS-1:
                if self.screenMap[self.snakeLoc[0][1]][self.snakeLoc[0][0]+1] == 0.5:
                    gameOver = True
                    reward = self.NEGREWARD
                elif self.screenMap[self.snakeLoc[0][1]][self.snakeLoc[0][0]+1] == 1:
                    reward = self.POSREWARD
                self.moveSnake((self.snakeLoc[0][0]+1, self.snakeLoc[0][1]))
            else:
                gameOver = True
                wallCrush = True
                reward = self.NEGREWARD
        elif action == 2: # ruch w dół
            if self.snakeLoc[0][1]<self.SEGMENTS-1:
                if self.screenMap[self.snakeLoc[0][1]+1][self.snakeLoc[0][0]] == 0.5:
                    gameOver = True
                    reward = self.NEGREWARD
                elif self.screenMap[self.snakeLoc[0][1]+1][self.snakeLoc[0][0]] == 1:
                    reward = self.POSREWARD
                self.moveSnake((self.snakeLoc[0][0], self.snakeLoc[0][1]+1))
            else:
                gameOver = True
                wallCrush = True
                reward = self.NEGREWARD
        elif action == 3: # ruch w lewo
            if self.snakeLoc[0][0]>0:
                if self.screenMap[self.snakeLoc[0][1]][self.snakeLoc[0][0]-1] == 0.5:
                    gameOver = True
                    reward = self.NEGREWARD
                elif self.screenMap[self.snakeLoc[0][1]][self.snakeLoc[0][0]-1] == 1:
                    reward = self.POSREWARD
                self.moveSnake((self.snakeLoc[0][0]-1, self.snakeLoc[0][1]))
            else:
                gameOver = True
                wallCrush = True
                reward = self.NEGREWARD
        if self.score == self.SEGMENTS**2-3: # WYGRANA!:)
            win = True
            gameOver = True
            print("YOU ARE THE WINNER!")

        # aktualizacja stanu gry(inputu do sieci) i ekranu gry
        self.direction = action
        self.drawScreen()
        pygame.time.wait(self.WAITTIME)
        newState = self.newState(wallCrush)

        return newState, reward, gameOver, win

# główna pętla do gry samodzielnej
if __name__ == '__main__':
    env = SnakeEnvironment(waitTime=300,segments=4)
    action = env.direction
    win = False
    while not win:
        # sterowanie
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_RIGHT:
                    action = 1
                elif event.key == pygame.K_DOWN:
                    action = 2
                elif event.key == pygame.K_LEFT:
                    action = 3
        _,_,gameOver,win = env.step(action) # podjęcie akcji (przy grze samodzielnej potrzeba tylko gameOver i win; reward i newState używa się w treningu)
        if gameOver: # przegrana i wyświetlenie wynikóww konsoli
            print("WYNIK: ", env.score,"\nLICZBA RUCHÓW: ", env.moves)
            # reset środowiska
            env.reset()
            action = env.direction
            gameOver = False
