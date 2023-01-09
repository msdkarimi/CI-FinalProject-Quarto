# Free for personal or classroom use; see 'LICENSE.md' for details.
# https://github.com/squillero/computational-intelligence

import numpy as np
from abc import abstractmethod
import copy
import random
from functools import cmp_to_key





class Player(object):

    def __init__(self, quarto, rounds=10000, alpha=0.1, randomFactor=0.2) -> None:
        self.quarto = quarto
        self.moveHistory = []  # place, reward
        self.pieceHistory = []  # piece, reward
        self.alpha = alpha
        self.randomFactor = randomFactor
        self.gainPlace = {}
        self.gainPiece = {}
        self.initReward(self.quarto)
        self.rounds = rounds
        self.currentRound = 0


    def initReward(self, quarto):
        for i, row in enumerate(quarto.board):
            for j, col in enumerate(row):
                self.gainPlace[(i, j)] = np.random.uniform(low=1.0, high=0.1)

        for pI,_ in enumerate(quarto.pieces):
            self.gainPiece[pI] = np.random.uniform(low=1.0, high=0.1)


    @abstractmethod
    def choose_piece(self) -> int:
        self.quarto.getAvailablePieces()
        maxG = -10e15
        next_piece_index = None
        randomN = np.random.random()
        if randomN < self.randomFactor:
            # if random number below random factor, choose random action
            next_piece_index = random.randint(0, 15)
        else:
            # if exploiting, gather all possible actions and choose one with the highest gainPlace (reward)
            for pieceIndex in self.quarto.availablePieces:
                new_piece_index = pieceIndex
                if self.gainPiece[new_piece_index] >= maxG:
                    next_piece_index = new_piece_index
                    maxG = self.gainPiece[pieceIndex]

        return next_piece_index

    @abstractmethod
    def place_piece(self) -> tuple[int, int]:
        pI = self.quarto.selected_piece_index
        self.quarto.getFreePlaces()
        maxG = -10e15
        next_move = None
        randomN = np.random.random()
        if randomN < self.randomFactor:
            # if random number below random factor, choose random action
            freePLaces = [tuple(i) for i in self.quarto.zippedFreePlaces]
            next_move = random.choice(freePLaces)
        else:
            # if exploiting, gather all possible actions and choose one with the highest gainPlace (reward)\
            # to built the list which shows fitness of each place based on sum of unfeasiblility
            th = 6
            pairXYfeasibility = []
            for placeXY in self.quarto.zippedFreePlaces:
                h,v,d =self.findForGivenPieceAndPlaceWeight(pI, placeXY)
                sum = h+v+d
                if sum > th:
                    weight = False
                else:
                    weight = True

                pairXYfeasibility.append((placeXY,weight))
            for action, weight in pairXYfeasibility:
                new_state = action
                if self.gainPlace[new_state] >= maxG or weight :
                    next_move = new_state
                    maxG = self.gainPlace[new_state]

            # for action in self.quarto.zippedFreePlaces:
            #     new_state = action
            #     if self.gainPlace[new_state] >= maxG :
            #         next_move = new_state
            #         maxG = self.gainPlace[new_state]

        return next_move

    def findForGivenPieceAndPlaceWeight(self, pi, placeXY):
        x, y = placeXY
        pieceWithCharacteristic = self.quarto.get_piece_charachteristics(pi)
        pieceWithPropertyCounter = PropertyCounter(pieceWithCharacteristic)

        dictDLTR = dict()
        dictDRTL = dict()
        dictV = dict()
        dictH = dict()

        horizentallyFeasible = True
        verticallyFeasible = True
        diagonallyFeasible = True

        horizentallyUneasibleCounter = 0
        verticallyUneasibleCounter = 0
        diagonallyUneasibleCounter = 0

        ifWinnerD, theDictD = self.quarto.check_diagonal()
        ifWinnerV, theDictV = self.quarto.check_vertical()
        ifWinnerH, theDictH = self.quarto.check_horizontal()
        if ifWinnerD == -1:
            dictDLTR = theDictD["LTR"]
            dictDRTL = theDictD["RTL"]
        else:
            dictDLTR = None
            dictDRTL = None
        if ifWinnerV == -1:
            dictV = theDictV
        else:
            dictV = None
        if ifWinnerH == -1:
            dictH = theDictH
        else:
            dictH = None

        if dictH is not None:
            for i in range(self.quarto.BOARD_SIDE):
                if i == x :
                    horizental = dictH[i]
                    self.performAddition( pieceWithPropertyCounter, horizental)
                    if horizental.h == 3 or horizental.nh == 3 or horizental.c == 3 or horizental.nc == 3 or horizental.so == 3 or horizental.nso == 3 or horizental.sq == 3 or horizental.nsq == 3:
                        # horizentallyFeasible = False
                        # break
                        horizentallyUneasibleCounter += 1
                continue

        if dictV is not None:
            for j in range(self.quarto.BOARD_SIDE):
                if j == y :
                    vertical = dictV[j]
                    self.performAddition(pieceWithPropertyCounter, vertical)
                    if vertical.h == 3 or vertical.nh == 3 or vertical.c == 3 or vertical.nc == 3 or vertical.so == 3 or vertical.nso == 3 or vertical.sq == 3 or vertical.nsq == 3:
                        # verticallyFeasible = False
                        # break
                        verticallyUneasibleCounter += 1
                continue

        if dictDLTR is not None and dictDRTL is not None:
            for i in range(self.quarto.BOARD_SIDE):
                for j in range(self.quarto.BOARD_SIDE):
                    z = j+i
                    sumXY = x+y
                    if (i == j and i == x and j == y) or ( z == 3 and sumXY == 3):
                        Diagonal1 = dictDLTR[i]
                        Diagonal2 = dictDLTR[i]

                        self.performAddition(pieceWithPropertyCounter, Diagonal1)
                        self.performAddition(pieceWithPropertyCounter, Diagonal2)

                        if Diagonal1.h == 3 or Diagonal1.nh == 3 or Diagonal1.c == 3 or Diagonal1.nc == 3 or Diagonal1.so == 3 or Diagonal1.nso == 3 or Diagonal1.sq == 3 or Diagonal1.nsq == 3 or Diagonal2.h == 3 or Diagonal2.nh == 3 or Diagonal2.c == 3 or Diagonal2.nc == 3 or Diagonal2.so == 3 or Diagonal2.nso == 3 or Diagonal2.sq == 3 or Diagonal2.nsq == 3 :
                            # diagonallyFeasible = False
                            # break
                            diagonallyUneasibleCounter += 1
        # return horizentallyFeasible, verticallyFeasible, diagonallyFeasible
        return horizentallyUneasibleCounter, verticallyUneasibleCounter, diagonallyUneasibleCounter

    def performAddition(self, pieceWithPropertyCounter, horizental):
        horizental.h += pieceWithPropertyCounter.h
        horizental.nh += pieceWithPropertyCounter.nh

        horizental.c += pieceWithPropertyCounter.c
        horizental.nc += pieceWithPropertyCounter.nc

        horizental.so += pieceWithPropertyCounter.so
        horizental.nso += pieceWithPropertyCounter.nso

        horizental.sq += pieceWithPropertyCounter.sq
        horizental.nsq += pieceWithPropertyCounter.nsq




    def updateMovesHistory(self, place):
        if self.quarto.assignReward() != 0:
            reward = 1 / self.quarto.assignReward()
        else:
            reward =  self.quarto.assignReward()
        self.moveHistory.append((place, reward))
    def updatePieceHistory(self, piece):
        reward = self.quarto.assignReward()
        self.pieceHistory.append((piece, reward))

    def get_game(self):
        return self.quarto

class PropertyCounter(object):
    def __init__(self, selectedPieceCharacteristic = None):
        if selectedPieceCharacteristic == None:
            self.h = 0
            self.nh = 0
            self.c = 0
            self.nc = 0
            self.so = 0
            self.nso = 0
            self.sq = 0
            self.nsq = 0
        else:
            if selectedPieceCharacteristic.HIGH:
                self.h = 1
                self.nh = 0
            else:
                self.nh = 1
                self.h = 0

            if selectedPieceCharacteristic.COLOURED:
                self.c = 1
                self.nc = 0
            else:
                self.nc = 1
                self.c = 0

            if selectedPieceCharacteristic.SOLID:
                self.so = 1
                self.nso = 0
            else:
                self.nso = 1
                self.so = 0

            if selectedPieceCharacteristic.SQUARE:
                self.sq = 1
                self.nsq = 0
            else:
                self.nsq = 1
                self.sq = 0

class Piece(object):

    def __init__(self, high: bool, coloured: bool, solid: bool, square: bool) -> None:
        self.HIGH = high
        self.COLOURED = coloured
        self.SOLID = solid
        self.SQUARE = square


class Quarto(object):

    MAX_PLAYERS = 2
    BOARD_SIDE = 4

    def __init__(self) -> None:
        self.players = ()
        self.reset()


    def reset(self):
        self.board = np.ones(shape=(self.BOARD_SIDE, self.BOARD_SIDE), dtype=int) * -1
        self.pieces = []
        self.pieces.append(Piece(False, False, False, False))  # 0
        self.pieces.append(Piece(False, False, False, True))  # 1
        self.pieces.append(Piece(False, False, True, False))  # 2
        self.pieces.append(Piece(False, False, True, True))  # 3
        self.pieces.append(Piece(False, True, False, False))  # 4
        self.pieces.append(Piece(False, True, False, True))  # 5
        self.pieces.append(Piece(False, True, True, False))  # 6
        self.pieces.append(Piece(False, True, True, True))  # 7
        self.pieces.append(Piece(True, False, False, False))  # 8
        self.pieces.append(Piece(True, False, False, True))  # 9
        self.pieces.append(Piece(True, False, True, False))  # 10
        self.pieces.append(Piece(True, False, True, True))  # 11
        self.pieces.append(Piece(True, True, False, False))  # 12
        self.pieces.append(Piece(True, True, False, True))  # 13
        self.pieces.append(Piece(True, True, True, False))  # 14
        self.pieces.append(Piece(True, True, True, True))  # 15
        self.current_player = 0
        self.selected_piece_index = -1
        self.getFreePlaces()
        self.learningPhase = False
        self.zippedFreePlaces = None
        self.availablePieces = None


    def set_players(self, players: tuple[Player, Player]):
        self.players = players
        # self.learnModelParameters()

    def select(self, pieceIndex: int) -> bool:
        '''
        select a piece. Returns True on success
        '''
        if pieceIndex not in self.board:
            self.selected_piece_index = pieceIndex
            return True
        return False

    def place(self, x: int, y: int) -> bool:
        '''
        Place piece in coordinates (x, y). Returns true on success
        '''
        if self.placeable(x, y):
            self.board[y, x] = self.selected_piece_index
            return True
        return False

    def placeable(self, x: int, y: int) -> bool:
        return not (y < 0 or x < 0 or x > 3 or y > 3 or self.board[y, x] >= 0)

    def print(self):
        '''
        Print the board
        '''
        for row in self.board:
            print("\n -------------------")
            print("|", end="")
            for element in row:
                print(f" {element: >2}", end=" |")
        print("\n -------------------\n")
        print(f"Selected piece: {self.selected_piece_index}\n")

    def get_piece_charachteristics(self, index: int) -> Piece:
        '''
        Gets charachteristics of a piece (index-based)
        '''
        return copy.deepcopy(self.pieces[index])

    def get_board_status(self) -> np.ndarray:
        '''
        Get the current board status (pieces are represented by index)
        '''
        return copy.deepcopy(self.board)

    def get_selected_piece(self) -> int:
        '''
        Get index of selected piece
        '''
        return copy.deepcopy(self.selected_piece_index)


    def getFreePlaces(self):
        self.freePlaces = np.where(self.board == -1)
        self.zippedFreePlaces = zip(self.freePlaces[1],self.freePlaces[0])


    def getAvailablePieces(self):
        allIndices = set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        board = self.get_board_status().ravel()
        allSelectedPieces = set(board[board > -1])
        self.availablePieces = list(allIndices - allSelectedPieces)



    def check_horizontal(self) -> int:
        horDict = dict()
        horDict[0] = None
        horDict[1] = None
        horDict[2] = None
        horDict[3] = None

        for i in range(self.BOARD_SIDE):
            initiallist = PropertyCounter()

            high_values = [
                elem for elem in self.board[i] if elem >= 0 and self.pieces[elem].HIGH
            ]
            initiallist.h = len(high_values)


            coloured_values = [
                elem for elem in self.board[i] if elem >= 0 and self.pieces[elem].COLOURED
            ]
            initiallist.c = len(coloured_values)

            solid_values = [
                elem for elem in self.board[i] if elem >= 0 and self.pieces[elem].SOLID
            ]
            initiallist.so = len(solid_values)

            square_values = [
                elem for elem in self.board[i] if elem >= 0 and self.pieces[elem].SQUARE
            ]
            initiallist.sq = len(square_values)

            low_values = [
                elem for elem in self.board[i] if elem >= 0 and not self.pieces[elem].HIGH
            ]
            initiallist.nh = len(low_values)

            noncolor_values = [
                elem for elem in self.board[i] if elem >= 0 and not self.pieces[elem].COLOURED
            ]
            initiallist.nc = len(noncolor_values)

            hollow_values = [
                elem for elem in self.board[i] if elem >= 0 and not self.pieces[elem].SOLID
            ]
            initiallist.nso = len(hollow_values)

            circle_values = [
                elem for elem in self.board[i] if elem >= 0 and not self.pieces[elem].SQUARE
            ]
            initiallist.nsq = len(circle_values)
            horDict[i] = initiallist

            if len(high_values) == self.BOARD_SIDE or len(
                    coloured_values
            ) == self.BOARD_SIDE or len(solid_values) == self.BOARD_SIDE or len(
                    square_values) == self.BOARD_SIDE or len(low_values) == self.BOARD_SIDE or len(
                        noncolor_values) == self.BOARD_SIDE or len(
                            hollow_values) == self.BOARD_SIDE or len(
                                circle_values) == self.BOARD_SIDE:
                return self.current_player, None

        return -1, horDict

    def check_vertical(self):
        verDict = dict()
        verDict[0] = None
        verDict[1] = None
        verDict[2] = None
        verDict[3] = None

        for i in range(self.BOARD_SIDE):
            # counts the total value of hight are selected
            initiallist = PropertyCounter()
            high_values = [
                elem for elem in self.board[:, i] if elem >= 0 and self.pieces[elem].HIGH
            ]
            initiallist.h = len(high_values)

            coloured_values = [
                elem for elem in self.board[:, i] if elem >= 0 and self.pieces[elem].COLOURED
            ]
            initiallist.c = len(coloured_values)

            solid_values = [
                elem for elem in self.board[:, i] if elem >= 0 and self.pieces[elem].SOLID
            ]
            initiallist.so = len(solid_values)

            square_values = [
                elem for elem in self.board[:, i] if elem >= 0 and self.pieces[elem].SQUARE
            ]
            initiallist.sq = len(square_values)

            low_values = [
                elem for elem in self.board[:, i] if elem >= 0 and not self.pieces[elem].HIGH
            ]
            initiallist.nh = len(low_values)

            noncolor_values = [
                elem for elem in self.board[:, i] if elem >= 0 and not self.pieces[elem].COLOURED
            ]
            initiallist.nc = len(noncolor_values)

            hollow_values = [
                elem for elem in self.board[:, i] if elem >= 0 and not self.pieces[elem].SOLID
            ]
            initiallist.nso = len(hollow_values)

            circle_values = [
                elem for elem in self.board[:, i] if elem >= 0 and not self.pieces[elem].SQUARE
            ]
            initiallist.nsq = len(circle_values)

            verDict[i] = initiallist

            if len(high_values) == self.BOARD_SIDE or len(
                    coloured_values
            ) == self.BOARD_SIDE or len(solid_values) == self.BOARD_SIDE or len(
                    square_values) == self.BOARD_SIDE or len(low_values) == self.BOARD_SIDE or len(
                        noncolor_values) == self.BOARD_SIDE or len(
                            hollow_values) == self.BOARD_SIDE or len(
                                circle_values) == self.BOARD_SIDE:
                return self.current_player , None
        return -1, verDict

    def check_diagonal(self):
        LTRdiagDict = dict()
        LTRdiagDict[0] = None
        LTRdiagDict[1] = None
        LTRdiagDict[2] = None
        LTRdiagDict[3] = None
        high_values = []
        coloured_values = []
        solid_values = []
        square_values = []
        low_values = []
        noncolor_values = []
        hollow_values = []
        circle_values = []


        for i in range(self.BOARD_SIDE):
            # if self.board[i, i] < 0:
            #     break
            LTRinitiallist = PropertyCounter()

            if self.pieces[self.board[i, i]].HIGH:
                if self.board[i, i] != -1:
                    high_values.append(self.board[i, i])
                    LTRinitiallist.h = len(high_values)
            else:
                if self.board[i, i] != -1:
                    low_values.append(self.board[i, i])
                    LTRinitiallist.nh = len(low_values)

            if self.pieces[self.board[i, i]].COLOURED:
                if self.board[i, i] != -1:
                    coloured_values.append(self.board[i, i])
                    LTRinitiallist.c = len(coloured_values)
            else:
                if self.board[i, i] != -1:
                    noncolor_values.append(self.board[i, i])
                    LTRinitiallist.nc = len(noncolor_values)

            if self.pieces[self.board[i, i]].SOLID:
                if self.board[i, i] != -1:
                    solid_values.append(self.board[i, i])
                    LTRinitiallist.so = len(solid_values)
            else:
                if self.board[i, i] != -1:
                    hollow_values.append(self.board[i, i])
                    LTRinitiallist.nso = len(hollow_values)


            if self.pieces[self.board[i, i]].SQUARE:
                if self.board[i, i] != -1:
                    square_values.append(self.board[i, i])
                    LTRinitiallist.sq = len(square_values)
            else:
                if self.board[i, i] != -1:
                    circle_values.append(self.board[i, i])
                    LTRinitiallist.nsq = len(circle_values)

            LTRdiagDict[i] = LTRinitiallist

        if len(high_values) == self.BOARD_SIDE or len(coloured_values) == self.BOARD_SIDE or len(
                solid_values) == self.BOARD_SIDE or len(square_values) == self.BOARD_SIDE or len(
                    low_values
                ) == self.BOARD_SIDE or len(noncolor_values) == self.BOARD_SIDE or len(
                    hollow_values) == self.BOARD_SIDE or len(circle_values) == self.BOARD_SIDE:
            return self.current_player, None

        RTLdiagDict = dict()
        RTLdiagDict[0] = None
        RTLdiagDict[1] = None
        RTLdiagDict[2] = None
        RTLdiagDict[3] = None
        high_values = []
        coloured_values = []
        solid_values = []
        square_values = []
        low_values = []
        noncolor_values = []
        hollow_values = []
        circle_values = []


        for i in range(self.BOARD_SIDE):
            # if self.board[i, self.BOARD_SIDE - 1 - i] < 0:
            #     break
            RTLinitiallist = PropertyCounter()
            if self.pieces[self.board[i, self.BOARD_SIDE - 1 - i]].HIGH:
                if self.board[i, self.BOARD_SIDE - 1 - i] != -1:
                    high_values.append(self.board[i, self.BOARD_SIDE - 1 - i])
                    RTLinitiallist.h = len(high_values)
            else:
                if self.board[i, self.BOARD_SIDE - 1 - i] != -1:
                    low_values.append(self.board[i, self.BOARD_SIDE - 1 - i])
                    RTLinitiallist.nh = len(low_values)

            if self.pieces[self.board[i, self.BOARD_SIDE - 1 - i]].COLOURED:
                if self.board[i, self.BOARD_SIDE - 1 - i] != -1:
                    coloured_values.append(
                        self.board[i, self.BOARD_SIDE - 1 - i])
                    RTLinitiallist.c = len(coloured_values)
            else:
                if self.board[i, self.BOARD_SIDE - 1 - i] != -1:
                    noncolor_values.append(
                        self.board[i, self.BOARD_SIDE - 1 - i])
                    RTLinitiallist.nc = len(noncolor_values)

            if self.pieces[self.board[i, self.BOARD_SIDE - 1 - i]].SOLID:
                if self.board[i, self.BOARD_SIDE - 1 - i] != -1:
                    solid_values.append(self.board[i, self.BOARD_SIDE - 1 - i])
                    RTLinitiallist.so = len(solid_values)
            else:
                if self.board[i, self.BOARD_SIDE - 1 - i] != -1:
                    hollow_values.append(self.board[i, self.BOARD_SIDE - 1 - i])
                    RTLinitiallist.nso = len(hollow_values)

            if self.pieces[self.board[i, self.BOARD_SIDE - 1 - i]].SQUARE:
                if self.board[i, self.BOARD_SIDE - 1 - i] != -1:
                    square_values.append(self.board[i, self.BOARD_SIDE - 1 - i])
                    RTLinitiallist.sq = len(square_values)
            else:
                if self.board[i, self.BOARD_SIDE - 1 - i] != -1:
                    circle_values.append(self.board[i, self.BOARD_SIDE - 1 - i])
                    RTLinitiallist.nsq = len(circle_values)

            RTLdiagDict[i] = RTLinitiallist

        if len(high_values) == self.BOARD_SIDE or len(coloured_values) == self.BOARD_SIDE or len(
                solid_values) == self.BOARD_SIDE or len(square_values) == self.BOARD_SIDE or len(
                    low_values
                ) == self.BOARD_SIDE or len(noncolor_values) == self.BOARD_SIDE or len(
                    hollow_values) == self.BOARD_SIDE or len(circle_values) == self.BOARD_SIDE:
            return self.current_player, None

        retunDict = {
            "LTR": LTRdiagDict,
            "RTL": RTLdiagDict
        }
        return -1, retunDict

    def check_winner(self) -> int:
        '''
        Check who is the winner
        '''
        checkV, _ = self.check_vertical()
        checkH, _ = self.check_horizontal()
        checkD, _ = self.check_diagonal()

        l = [checkH, checkV, checkD]
        for elem in l:
            if elem >= 0:
                return elem
        return -1

    def check_finished(self) -> bool:
        '''
        Check who is the loser
        '''
        for row in self.board:
            for elem in row:
                if elem == -1:
                    return False
        return True

    def assignReward(self):
        return -1 * int(not self.check_finished())

    def run(self) -> int:
        '''
        Run the game (with output for every move)
        '''
        winner = -1
        while winner < 0 and not self.check_finished():
            self.print()
            piece_ok = False
            while not piece_ok:
                selectedPiece =  self.players[self.current_player].choose_piece()
                piece_ok = self.select(selectedPiece)
            piece_ok = False
            self.current_player = (self.current_player + 1) % self.MAX_PLAYERS
            self.print()
            while not piece_ok:
                x, y = self.players[self.current_player].place_piece()
                piece_ok = self.place(x, y)
            winner = self.check_winner()
        self.print()
        return winner

    def updateSelectedPiecesAndPlaces(self):
        self.getAvailablePieces()
        self.getFreePlaces()
    def learnModelParameters(self, pieces, board,ifLearning = False):
        winner0 = 0
        winner1 = 0
        winner11 = 0
        if ifLearning:
            # winner = -1
            for epoch in range(self.players[0].rounds):
                self.board = np.ones(shape=(self.BOARD_SIDE, self.BOARD_SIDE), dtype=int) * -1
                self.availablePieces = pieces
                winner = -1

                ++self.players[0].currentRound
                while winner < 0 and not self.check_finished():
                    self.updateSelectedPiecesAndPlaces()
                    # self.print()
                    piece_ok = False
                    while not piece_ok:
                        self.updateSelectedPiecesAndPlaces()
                        selectedPiece = self.players[self.current_player].choose_piece()
                        if self.current_player == 0:
                            selectedPiece = self.changePieceToWorstOne(selectedPiece)

                        piece_ok = self.select(selectedPiece)
                        if piece_ok and not bool(self.current_player):
                            self.players[0].updatePieceHistory(self.selected_piece_index)
                    piece_ok = False
                    self.current_player = (self.current_player + 1) % self.MAX_PLAYERS
                    # self.print()
                    while not piece_ok:
                        self.updateSelectedPiecesAndPlaces()
                        place = self.players[self.current_player].place_piece()
                        x, y = place
                        piece_ok = self.place(x, y)
                        if piece_ok and not bool(self.current_player):
                            self.players[0].updateMovesHistory(place)
                    winner = self.check_winner()
                if winner == 0 :
                    winner0 +=1
                elif winner == 1:
                    winner1 +=1
                else:
                    winner11 +=1
                # self.print()
                # print(f"Winner is: {winner}")

                if winner == 0:
                    self.learn(self.players[0])

            #     do not forget to reinitialize board
            #  return lernt weights
            print(f"RL wins: {winner0} and Random wins: {winner1} and Draw is: {winner11}, ")
            return self.players[0].gainPlace, self.players[0].gainPiece
        else:
            return ifLearning

    def changePieceToWorstOne(self, selectedPiece):
        self.updateSelectedPiecesAndPlaces()
        # selectedPieceCharacteristics = self.get_piece_charachteristics(selectedPiece)
        vectorCharacteristicPiece = self.vectorizePieceCharacteristic(selectedPiece)
        vectorizeAvailablePieces = [self.vectorizePieceCharacteristic(i) for i in self.availablePieces]
        # makeItWorse = np.invert(vectorCharacteristicPiece)
        makeItWorse = vectorCharacteristicPiece
        pairPiceCHandItsXor = list()
        for pieceCharacter in vectorizeAvailablePieces:
            x = tuple(pieceCharacter)
            y = tuple(makeItWorse)
            if x == y:
                indexOfPiece = self.findCoresspondingIndex(x)
                return indexOfPiece

        for pieceCharacter in vectorizeAvailablePieces:
            myTuple = tuple(pieceCharacter)
            xorWorstAndCurrentPiece = np.logical_xor(makeItWorse,pieceCharacter )
            weight = np.sum(xorWorstAndCurrentPiece)
            pairPiceCHandItsXor.append((myTuple, weight))

        newList = sorted(pairPiceCHandItsXor, key=cmp_to_key(self.compare), reverse=True)
        # bestIsWorst, _ = newList[len(newList)-1]
        bestIsWorst, _ = newList[0]
        index = self.findCoresspondingIndex(bestIsWorst)
        return index

    def compare(self, pair1, pair2):
        _, fitness1 = pair1
        _, fitness2 = pair2

        if fitness2 > fitness1:
            return -1
        else:
            return 1

    def vectorizePieceCharacteristic(self, selectedPiece):
        pieceObject= self.get_piece_charachteristics(selectedPiece)
        return np.array([pieceObject.HIGH, pieceObject.COLOURED, pieceObject.SOLID, pieceObject.SQUARE])

    def findCoresspondingIndex(self, pieceC):

        for index in self.availablePieces:
            pC = tuple(self.vectorizePieceCharacteristic(index))
            if pC == pieceC:
                return index



    def learn(self, player):
        target = 0
        for prev, reward in reversed(player.moveHistory):
            player.gainPlace[prev] = player.gainPlace[prev] + player.alpha * (target - player.gainPlace[prev])
            target += reward

        target = 0
        for prev, reward in reversed(player.pieceHistory):
            player.gainPiece[prev] = player.gainPiece[prev] + player.alpha * (target - player.gainPiece[prev])
            target += reward

        player.moveHistory= []
        player.pieceHistory = []

        player.randomFactor -= 10e-5  # decrease random factor each episode of play