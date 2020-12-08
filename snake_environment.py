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
        for i in range(3): # zapoczątkowanie 3 startowych segmentów węża
            self.snakeLoc.append([2-i,1]) # pierwsze 3 kolumny, 2 rząd
            self.screenMap[2-i,1] = 0.5
        self.appleLoc = self.createApple() # stworzenie jabłka
        self.drawScreen() # rysowanie elementów

    # tworzy jabłko w losowej lokacji poza snake'iem
    def createApple(self):
        x = randint(0,self.SEGMENTS-1)
        y = randint(0,self.SEGMENTS-1)
        while self.screenMap[x][y] == 0.5:
            x = randint(0,self.SEGMENTS-1)
            y = randint(0,self.SEGMENTS-1)
        self.screenMap[x][y] = 1
        return x,y

    # rysuje elementy na ekranie (węża i jabłko)
    def drawScreen(self):
        self.SCREEN.fill((0,0,0)) # czarne tło
        for i in range(self.SEGMENTS):
            for j in range(self.SEGMENTS):
                if self.screenMap[i][j] == 0.5: # segmenty węża
                    pygame.draw.rect(self.SCREEN, (0, 255, 0), (i*(self.SEGMENTSIZE+self.GRID),j*(self.SEGMENTSIZE+self.GRID), self.SEGMENTSIZE,self.SEGMENTSIZE))
                elif self.screenMap[i][j] == 1: # jabłko
                    pygame.draw.rect(self.SCREEN, (255, 0, 0), (i*(self.SEGMENTSIZE+self.GRID),j*(self.SEGMENTSIZE+self.GRID), self.SEGMENTSIZE,self.SEGMENTSIZE))
        pygame.display.flip()

    # pojedyncze poruszenie węża, aktualizacja listy lokacji węża, mapy, liczby ruchów i punktów
    def moveSnake(self, nextLoc):
        self.snakeLoc.insert(0,nextLoc) # wstawienie nowego segmentu przed pierwszy segment
        if nextLoc != self.appleLoc: # niezjedzenie jabłka -> usunięcie ostatniego segmentu węża
            self.snakeLoc.pop(len(self.snakeLoc)-1)
        else: # zjedzenie jabłka -> nowe jabłko i score+1
            self.appleLoc = self.createApple()
            self.score +=1
        # aktualizacja screenMap
        self.screenMap = zeros((self.SEGMENTS,self.SEGMENTS))
        for i in self.snakeLoc:
            self.screenMap[i[0],i[1]] = 0.5 # wąż na mapę
        self.screenMap[self.appleLoc[0],self.appleLoc[1]] = 1 # jabłko na mapę

    # krok w grze i przyszłej nauce
    def step(self,action):
        self.moves +=1
        gameOver = False
        reward = self.LIVINGPENALTY # domyślna kara to livingPenalty
        # wykluczneie możliwości skrętu o 180 stopni
        if action == 1 and self.direction == 3:
            action = 3
        elif action == 2 and self.direction == 0:
            action = 0
        elif action == 3 and self.direction == 1:
            action = 1
        elif action == 0 and self.direction == 2:
            action = 2

        # aktualizowanie gameOver i nagrody przy poszczególnych ruchach(0,1,2,3)
        if action == 0: # ruch w górę
            if self.snakeLoc[0][1]>0:
                if self.screenMap[self.snakeLoc[0][0]][self.snakeLoc[0][1]-1] == 0.5: # zjedzenie samego siebie -> PRZEGRANA
                    gameOver = True
                    reward = self.NEGREWARD
                elif self.screenMap[self.snakeLoc[0][0]][self.snakeLoc[0][1]-1] == 1: # zjedzenie jabłka -> POZYTYWNA NAGRODA
                    reward = self.POSREWARD
                self.moveSnake((self.snakeLoc[0][0],self.snakeLoc[0][1]-1)) # przeunięcie węża
            else: #wyjście poza mapę -> PRZEGRANA
                gameOver = True
                reward = self.NEGREWARD
        elif action == 1: # ruch w prawo
            if self.snakeLoc[0][0]<self.SEGMENTS-1:
                if self.screenMap[self.snakeLoc[0][0]+1][self.snakeLoc[0][1]] == 0.5:
                    gameOver = True
                    reward = self.NEGREWARD
                elif self.screenMap[self.snakeLoc[0][0]+1][self.snakeLoc[0][1]] == 1:
                    reward = self.POSREWARD
                self.moveSnake((self.snakeLoc[0][0]+1, self.snakeLoc[0][1]))
            else:
                gameOver = True
                reward = self.NEGREWARD
        elif action == 2: # ruch w dół
            if self.snakeLoc[0][1]<self.SEGMENTS-1:
                if self.screenMap[self.snakeLoc[0][0]][self.snakeLoc[0][1]+1] == 0.5:
                    gameOver = True
                    reward = self.NEGREWARD
                elif self.screenMap[self.snakeLoc[0][0]][self.snakeLoc[0][1]+1] == 1:
                    reward = self.POSREWARD
                self.moveSnake((self.snakeLoc[0][0], self.snakeLoc[0][1]+1))
            else:
                gameOver = True
                reward = self.NEGREWARD
        elif action == 3: # ruch w lewo
            if self.snakeLoc[0][0]>0:
                if self.screenMap[self.snakeLoc[0][0]-1][self.snakeLoc[0][1]] == 0.5:
                    gameOver = True
                    reward = self.NEGREWARD
                elif self.screenMap[self.snakeLoc[0][0]-1][self.snakeLoc[0][1]] == 1:
                    reward = self.POSREWARD
                self.moveSnake((self.snakeLoc[0][0]-1, self.snakeLoc[0][1]))
            else:
                gameOver = True
                reward = self.NEGREWARD

        self.direction = action # zmiana kierunku ruchu
        self.drawScreen() # aktualizacja ekranu
        pygame.time.wait(self.WAITTIME) # czas między przesunięciami

        # zainstniały nowy currentState jako input do sieci
        newState = zeros((1,len(self.screenMap)**2+1))
        for i in range(len(self.screenMap)):
            for j in range(len(self.screenMap)):
                newState[0][len(self.screenMap)*i+j] = self.screenMap[i][j]
        newState[0][len(self.screenMap)**2] = self.direction

        return newState, reward, gameOver

# główna pętla do gry samodzielnej
if __name__ == '__main__':
    env = SnakeEnvironment()
    gameOver = False
    action = 1 # początkowy ruch w prawo jako domyślny
    while True:
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

        _,_,gameOver = env.step(action) # podjęcie akcji (przy grze samodzielnej potrzeba tylko gameOver, reward dopiero w treningu)

        if gameOver: # przegrana
            print("WYNIK: ", env.score,"\nLICZBA RUCHÓW: ", env.moves) # print wyniku w konsoli
            env.reset() # reset do początkowych parametrów
            action = 1 # początkowy ruch w prawo jako domyślny
            gameOver = False
