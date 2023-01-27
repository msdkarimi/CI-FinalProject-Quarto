# Free for personal or classroom use; see 'LICENSE.md' for details.
# https://github.com/squillero/computational-intelligence

import logging
import argparse
import random
import quarto
import numpy
from numpy import save
import matplotlib.pyplot as plot
import copy


class RandomPlayer(quarto.Player):
    """Random player"""

    def __init__(self, quarto: quarto.Quarto, learningPhase) -> None:
        super().__init__(quarto,learningPhase)

    def choose_piece(self) -> int:
        return random.randint(0, 15)

    def place_piece(self) -> tuple[int, int]:
        return random.randint(0, 3), random.randint(0, 3)

def main():
    # game = quarto.Quarto()
    # playerRL = quarto.Player(game, True)
    # playerR = RandomPlayer(game, True)
    # game.set_players((playerRL, playerR))
    # game.getAvailablePieces()
    # game.getFreePlaces()
    # placeReward, pieceReward = game.learnModelParameters(copy.deepcopy(game.availablePieces))
    # save("weightOfPlaceFinalExam.npy", placeReward)
    # save("weightOfPieceFinalExam.npy", pieceReward)

    pieceReward = numpy.load('weightOfPieceFinal.npy', allow_pickle=True)
    pieceReward = dict(enumerate(pieceReward.flatten(), 1))
    pieceReward = pieceReward[1]
    placeReward = numpy.load('weightOfPlaceFinal.npy', allow_pickle=True)
    placeReward = dict(enumerate(placeReward.flatten(), 1))
    placeReward = placeReward[1]

    print(pieceReward)
    print(placeReward)

    # -----------------------------------------------------/\ Battle Field /\-----------------------------------------------
    rounds = 1000
    runIndex = 10
    RlProportion = []
    Randomproportion = []
    valueX = []

    for j in range(runIndex):
        RL = 0
        rand = 0
        draw = 0
        for i in range(rounds):
            battleField = quarto.Quarto()
            agentRL = quarto.Player(battleField, False)
            agentRandom = RandomPlayer(battleField, False)
            agentRL.gainPiece = pieceReward
            agentRL.gainPlace = placeReward
            battleField.set_players((agentRL, agentRandom))
            winner = battleField.run()
            if winner == 0:
                RL += 1
            elif winner == 1:
                rand += 1
            else:
                draw += 1
        valueX.append(j)
        Proportion = RL / rounds
        RlProportion.append(Proportion)
        Randomproportion.append(1 - Proportion)
        print(f"RL rate ={RL / rounds} and randon rate = {rand / rounds}")

    plot.semilogy(valueX, RlProportion, "b")
    plot.axhline(y=0.5, color='r', linestyle='--')
    plot.xlim([-1.0, runIndex])
    plot.ylim([0, 1])
    plot.title("Precentage Of RL Agent Wins")
    plot.legend(["RL agent", "Mean"])
    plot.xlabel("Runs")
    plot.ylabel(f"{rounds}-Rounds")

    plot.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='count', default=0, help='increase log verbosity')
    parser.add_argument('-d',
                        '--debug',
                        action='store_const',
                        dest='verbose',
                        const=2,
                        help='log debug messages (same as -vv)')
    args = parser.parse_args()

    if args.verbose == 0:
        logging.getLogger().setLevel(level=logging.WARNING)
    elif args.verbose == 1:
        logging.getLogger().setLevel(level=logging.INFO)
    elif args.verbose == 2:
        logging.getLogger().setLevel(level=logging.DEBUG)

    main()