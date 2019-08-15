/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package demo;

import core.NeuralNetwork;
import core.NeuralNetworkException;
import core.activation.ActivationFunction;
import core.activation.ActivationFunctionType;
import core.layer.LayerType;
import core.loss.LossFunctionType;
import core.normalization.NormalizationType;
import core.optimization.OptimizationType;
import core.reinforcement.Agent;
import core.reinforcement.AgentException;
import core.reinforcement.DeepAgent;
import core.reinforcement.Environment;
import utils.*;

import java.io.IOException;
import java.util.*;

/**
 * Class that implements tic tac toe game using deep reinforcement learning.
 *
 */
public class TicTacToe implements Environment {

    /**
     * Game states.
     *
     */
    private enum GameStatus {
        ONGOING,
        NOUGHT,
        CROSS,
        DRAW
    }

    /**
     * State value of player nought.
     *
     */
    private static final double NOUGHT = -1;

    /**
     * State value of empty game board position.
     *
     */
    private static final double EMPTY = 0;

    /**
     * State value of player cross.
     *
     */
    private static final double CROSS = 1;

    /**
     * Class that defines game board.
     *
     */
    private class GameBoard {

        /**
         * Current state of game (game board).
         *
         */
        private Matrix state;

        /**
         * Size of game board (size x size)
         *
         */
        private int size;

        /**
         * Initializes game board.
         *
         * @param size size as size x size.
         */
        GameBoard (int size) {
            this.size = size;
            state = new DMatrix(size * size, 1);
            for (int row = 0; row < size; row++) {
                for (int col = 0; col < size; col++) {
                    state.setValue(getPos(row, col), 0, EMPTY);
                }
            }
        }

        /**
         * Return current state of game. (game board)
         *
         * @return state of game.
         */
        public Matrix getState() {
            return state;
        }

        /**
         * Returns row and column as specific action value.
         *
         * @param row row of game board.
         * @param col column of game board.
         * @return action value calculated.
         */
        private int getPos(int row, int col) {
            return row + col * size;
        }

        /**
         * Checks if taken move (action) is valid.
         *
         * @param action action taken.
         * @return true if move is valid otherwise false.
         */
        public boolean isValidMove(int action) {
            return state.getValue(action, 0) == EMPTY;
        }

        /**
         * Checks if taken move (action) is valid.
         *
         * @param row row of game board.
         * @param col column of game board.
         * @return true if move is valid otherwise false.
         */
        public boolean isValidMove(int row, int col) {
            return isValidMove(getPos(row, col));
        }

        /**
         * Returns list of available moves (actions).
         *
         * @return list of available moves (actions)
         */
        public ArrayList<Integer> getAvailableMoves() {
            ArrayList<Integer> availableMoves = new ArrayList<>();
            for (int row = 0; row < size; row++) {
                for (int col = 0; col < size; col++) {
                    if (isValidMove(row, col)) availableMoves.add(getPos(row, col));
                }
            }
            return availableMoves;
        }

        private boolean noAvailableMoves() {
            return gameBoard.getAvailableMoves().isEmpty();
        }

        /**
         * Makes move per action taken.
         *
         * @param row row of game board.
         * @param col column of game board.
         * @param value state value of player.
         */
        public void makeMove(int row, int col, double value) {
            state.setValue(getPos(row, col), 0, value);
        }

        /**
         * Makes move per action taken.
         *
         * @param action action taken.
         * @param value state value of player.
         */
        public void makeMove(int action, double value) {
            state.setValue(action, 0, value);
        }

        /**
         * Checks if one of players has won the game.
         *
         * @return return player id if player has won otherwise returns ongoing game state.
         */
        private GameStatus checkWinner() {
            int[] diagStat = new int[2];
            int[] adiagStat = new int[2];
            for (int row = 0; row < size; row++) {
                int[] rowStat = new int[2];
                int[] colStat = new int[2];
                for (int col = 0; col < size; col++) {
                    int pos = getPos(row, col);
                    if (state.getValue(pos, 0) == EMPTY) rowStat[0] = rowStat[1] = 0;
                    else {
                        if (state.getValue(pos, 0) == NOUGHT) rowStat[0]++;
                        if (state.getValue(pos, 0) == CROSS) rowStat[1]++;
                    }

                    pos = getPos(col, row);
                    if (state.getValue(pos, 0) == EMPTY) colStat[0] = colStat[1] = 0;
                    else {
                        if (state.getValue(pos, 0) == NOUGHT) colStat[0]++;
                        if (state.getValue(pos, 0) == CROSS) colStat[1]++;
                    }
                }

                if (rowStat[0] == size || colStat[0] == size) return GameStatus.NOUGHT;
                if (rowStat[1] == size || colStat[1] == size) return GameStatus.CROSS;

                int pos = getPos(row, row);
                if (state.getValue(pos, 0) == EMPTY) diagStat[0] = diagStat[1] = 0;
                else {
                    if (state.getValue(pos, 0) == NOUGHT) diagStat[0]++;
                    if (state.getValue(pos, 0) == CROSS) diagStat[1]++;
                }

                pos = getPos(row, size - 1 - row);
                if (state.getValue(pos, 0) == EMPTY) adiagStat[0] = adiagStat[1] = 0;
                else {
                    if (state.getValue(pos, 0) == NOUGHT) adiagStat[0]++;
                    if (state.getValue(pos, 0) == CROSS) adiagStat[1]++;
                }

            }

            if (diagStat[0] == size || adiagStat[0] == size) return GameStatus.NOUGHT;
            if (diagStat[1] == size || adiagStat[1] == size) return GameStatus.CROSS;

            return GameStatus.ONGOING;
        }

        /**
         * Updates state of game after action is taken.
         *
         * @return state of game.
         */
        public GameStatus updateGameStatus() {
            gameStatus = gameBoard.checkWinner();
            if (gameStatus == GameStatus.ONGOING && gameBoard.noAvailableMoves()) gameStatus = GameStatus.DRAW;
            return gameStatus;
        }

    }

    /**
     * Reward structure i.e. rewards returned to agent during game per action taken.
     *
     */
    public class RewardStructure {
        double WIN = 5; // 5
        double DRAW = 3; // 3
        double LOST = 0; // 0
        double MOVE = 0; // 0
        double ILLEGAL_MOVE = 0;
    }

    /**
     * Reward structure.
     *
     */
    private RewardStructure rewardStructure = new RewardStructure();

    /**
     * Deep agent for player nought.
     *
     */
    private DeepAgent nought;

    /**
     * Deep agent for player cross.
     *
     */
    private DeepAgent cross;

    /**
     * Size of game board (size x size).
     *
     */
    private int boardSize = 3;

    /**
     * Game board.
     *
     */
    private GameBoard gameBoard;

    /**
     * Current player (actor) of game.
     *
     */
    private GameStatus currentPlayer;

    /**
     * Status of game.
     *
     */
    private GameStatus gameStatus;

    /**
     * Count of draw games.
     *
     */
    private int drawCount = 0;

    /**
     * Count of games that nought won.
     *
     */
    private int playerNoughtWonCount = 0;

    /**
     * Count of games that cross won.
     *
     */
    private int playerCrossWonCount = 0;

    /**
     * Number of games to be played.
     *
     */
    private int numberOfGames = 5000000;

    /**
     * Sum of all game moves taken.
     *
     */
    private int allMoves = 0;

    /**
     * Sum of illegal moves taken.
     *
     */
    private int illegalMoves = 0;

    /**
     * Random number generator.
     *
     */
    private Random random = new Random();

    /**
     * Main function for tic tac toe.
     *
     * @param args not used.
     */
    public static void main(String[] args) {
        TicTacToe ticTacToe;
        try {
            ticTacToe = new TicTacToe();
            ticTacToe.playGames();
        }
        catch (Exception exception) {
            exception.printStackTrace();
            System.exit(-1);
        }
    }

    /**
     * Constructor for tic tac toe.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException throws exception if setting of dynamic parameter fails.
     * @throws IOException throws exception if coping of neural network instance fails.
     * @throws ClassNotFoundException throws exception if coping of neural network instance fails.
     */
    public TicTacToe() throws NeuralNetworkException, DynamicParamException, IOException, ClassNotFoundException {
        nought = createAgent(GameStatus.NOUGHT);
        cross = createAgent(GameStatus.CROSS);
    }

    private void playGames() throws MatrixException, NeuralNetworkException, IOException, ClassNotFoundException {
        int drawCountTemp = 0;
        int playerNoughtWonCountTemp = 0;
        int playerCrossWonCountTemp = 0;
        for (int game = 0; game < numberOfGames; game++) {
            playGame();
            if (gameStatus == GameStatus.NOUGHT) {
                playerNoughtWonCountTemp++;
                playerNoughtWonCount++;
            }
            if (gameStatus == GameStatus.CROSS) {
                playerCrossWonCountTemp++;
                playerCrossWonCount++;
            }
            if (gameStatus == GameStatus.DRAW) {
                drawCountTemp++;
                drawCount++;
            }
            if (game % 50 == 0 && game > 0) {
                System.out.println("Game #" + game + " Epsilon: " + nought.getEpsilon() + " Nought won: " + playerNoughtWonCountTemp + " Cross won: " + playerCrossWonCountTemp + " Draw: " + drawCountTemp + " Illegal moves: " + illegalMoves + " (Total moves: " + allMoves + ")");
                drawCountTemp = 0;
                playerNoughtWonCountTemp = 0;
                playerCrossWonCountTemp = 0;
                allMoves = 0;
                illegalMoves = 0;
            }
        }
        System.out.println("Nought won games: " + playerNoughtWonCount);
        System.out.println("Cross won games: " + playerCrossWonCount);
        System.out.println("Draw games: " + drawCount);
        nought.stop();
        cross.stop();
    }

    /**
     * Returns current state of environment for the agent.
     *
     * @return state of environment
     */
    public Matrix getState() {
        return gameBoard.getState().copy();
    }

    /**
     * True if state is terminal. This is usually true if episode is completed.
     *
     * @return true if state is terminal.
     */
    public boolean isTerminalState() {
        return gameStatus != GameStatus.ONGOING;
    }

    /**
     * Returns available actions in current state of environment
     *
     * @return available actions in current state of environment.
     */
    public ArrayList<Integer> getAvailableActions() {
        return gameBoard.getAvailableMoves();
    }

    /**
     * Checks if action is valid.
     *
     * @param agent agent that is taking action.
     * @param action action to be taken.
     * @return true if action can be taken successfully.
     */
    public boolean isValidAction(Agent agent, int action) {
        return gameBoard.isValidMove(action);
    }

    /**
     * Takes random action.
     *
     * @param agent agent that is taking action.
     * @return action taken
     */
    public int requestAction(Agent agent) {
        ArrayList<Integer> availableActions = getAvailableActions();
        return availableActions.get(random.nextInt(availableActions.size()));
    }

    /**
     * Takes specific action and updates game status after successful action.
     *
     * @param agent agent that is taking action.
     * @param action action to be taken.
     */
    public void commitAction(Agent agent, int action) {
        gameBoard.makeMove(action, currentPlayer == GameStatus.NOUGHT ? NOUGHT : CROSS);
        gameStatus = gameBoard.updateGameStatus();
    }

    /**
     * Requests immediate reward from environment after taking action.
     *
     * @param agent agent that is asking for reward.
     * @param validAction true if taken action was available one.
     * @return immediate reward.
     */
    public double requestReward(Agent agent, boolean validAction) {
        if (!validAction) return rewardStructure.ILLEGAL_MOVE;
        else {
            if (gameStatus == GameStatus.ONGOING) return rewardStructure.MOVE;
            else {
                if (gameStatus == GameStatus.DRAW) return rewardStructure.DRAW;
                else {
                    if (gameStatus == currentPlayer) return rewardStructure.WIN;
                    return rewardStructure.LOST;
                }
            }
        }
    }

    /**
     * Returns active agent (player)
     *
     * @return active agent
     */
    private Agent getAgent() {
        return (currentPlayer == GameStatus.NOUGHT) ? nought : cross;
    }

    /**
     * Plays single episode of game.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws IOException throws exception if cloning of neural network fails.
     * @throws ClassNotFoundException throws exception if cloning of neural network fails.
     */
    private void playGame() throws MatrixException, NeuralNetworkException, IOException, ClassNotFoundException {
        currentPlayer = random.nextInt(2) == 0 ? GameStatus.NOUGHT : GameStatus.CROSS;
        gameBoard = new GameBoard(boardSize);
        gameStatus = GameStatus.ONGOING;
        ArrayList<Matrix> gameStates = new ArrayList<>();
        gameStates.add(gameBoard.getState().copy());
        nought.newEpisode();
        cross.newEpisode();
        do {
            getAgent().nextEpisodeStep();
            try {
                if (!getAgent().act(false)) {
                    illegalMoves++;
                    getAgent().act(true);
                }
            }
            catch (AgentException agentException) {
                System.out.println(agentException.toString());
                System.exit(-1);
            }

            allMoves++;
            gameStates.add(gameBoard.getState().copy());

            if (gameStatus == GameStatus.ONGOING) getAgent().updateValue();

            currentPlayer = currentPlayer == GameStatus.CROSS ? GameStatus.NOUGHT : GameStatus.CROSS;

        } while (gameStatus == GameStatus.ONGOING);

        nought.endEpisode();
        cross.endEpisode();

        if (allMoves == 450 && illegalMoves == 0 && gameStatus == GameStatus.DRAW) printGame(gameStates, gameStatus);

    }

    /**
     * Prints game.
     *
     * @param gameStates list of game states during game episode.
     * @param winner winner of game.
     */
    private void printGame(ArrayList<Matrix> gameStates, GameStatus winner) {
        System.out.println("New Game");
        for (Matrix gameState : gameStates) printBoard(gameState);
        if (winner != null) System.out.println("WINNER: " + winner);
        else System.out.println("DRAW");
    }

    /**
     * Prints board for a single game state.
     *
     * @param gameState single game state.
     */
    private void printBoard(Matrix gameState) {
        for (int row = 0; row < 2 * boardSize + 1; row++) System.out.print("-");
        System.out.println("");
        for (int row = 0; row < boardSize; row++) {
            for (int col = 0; col < boardSize; col++) {
                if (col == 0) System.out.print("|");
                double posStatus = gameState.getValue(row + col * boardSize, 0);
                if (posStatus == NOUGHT) System.out.print("O");
                if (posStatus == EMPTY) System.out.print(" ");
                if (posStatus == CROSS) System.out.print("X");
                System.out.print("|");
            }
            System.out.println("");
        }
        for (int row = 0; row < 2 * boardSize + 1; row++) System.out.print("-");
        System.out.println("");
    }

    /**
     * Creates agent (player).
     *
     * @return agent
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException throws exception if setting of dynamic parameter fails.
     * @throws IOException throws exception if coping of neural network instance fails.
     * @throws ClassNotFoundException throws exception if coping of neural network instance fails.
     */
    private DeepAgent createAgent(GameStatus player) throws NeuralNetworkException, DynamicParamException, IOException, ClassNotFoundException {
        NeuralNetwork QNN = buildNeuralNetwork(player == GameStatus.NOUGHT ? "Nought" : "Cross", boardSize * boardSize, boardSize * boardSize);
        DeepAgent agent = new DeepAgent(this, QNN);
        agent.start();
        return agent;
    }

    /**
     * Build neural network for tic tac toe player (agent).
     *
     * @param inputSize input size of neural network (number of states)
     * @param outputSize output size of neural network (number of actions and their values).
     * @return built neural network
     * @throws DynamicParamException throws exception if setting of dynamic parameters fails.
     * @throws NeuralNetworkException throws exception if building of neural network fails.
     */
    private static NeuralNetwork buildNeuralNetwork(String neuralNetworkName, int inputSize, int outputSize) throws DynamicParamException, NeuralNetworkException {
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.addInputLayer("width = " + inputSize);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(ActivationFunctionType.RELU), "width = 40");
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(ActivationFunctionType.RELU), "width = 40");
        neuralNetwork.addOutputLayer(LayerType.FEEDFORWARD, new ActivationFunction(ActivationFunctionType.RELU), "width = " + outputSize);
        neuralNetwork.build();
        neuralNetwork.setOptimizer(OptimizationType.ADAM);
        neuralNetwork.addNormalizer(2, NormalizationType.WEIGHT_NORMALIZATION);
        neuralNetwork.setLossFunction(LossFunctionType.HUBER);
        neuralNetwork.setNeuralNetworkName(neuralNetworkName);
        neuralNetwork.setTrainingSampling(100, false, true);
        neuralNetwork.setTrainingIterations(50);
        neuralNetwork.verboseTraining(100);
        return neuralNetwork;
    }

}
