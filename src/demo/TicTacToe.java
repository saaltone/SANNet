/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package demo;

import core.network.NeuralNetwork;
import core.network.NeuralNetworkConfiguration;
import core.network.NeuralNetworkException;
import core.activation.ActivationFunction;
import core.layer.LayerType;
import core.optimization.OptimizationType;
import core.reinforcement.agent.*;
import core.reinforcement.policy.executablepolicy.*;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Implements Tic-Tac-Toe game using deep reinforcement learning.<br>
 *
 */
public class TicTacToe implements Environment, AgentFunctionEstimator, ActionListener, MouseListener {

    /**
     * Player role.
     *
     */
    private enum PlayerRole {
        NOUGHT,
        CROSS,
    }

    /**
     * Implements player
     *
     */
    private static class Player implements Comparable<Player> {

        /**
         * Role of player.
         *
         */
        private PlayerRole playerRole;

        /**
         * Agent of player.
         *
         */
        private final Agent agent;

        /**
         * Player index.
         *
         */
        private final int index;

        /**
         * If true player is human.
         *
         */
        private boolean isHuman = false;

        /**
         * Total count of won games.
         *
         */
        private int wonCount = 0;

        /**
         * Total count of lost games.
         *
         */
        private int lostCount = 0;

        /**
         * Total count of draw games.
         *
         */
        private int drawCount = 0;

        /**
         * Cyclical count of won games.
         *
         */
        private int wonCountCyclic = 0;

        /**
         * Cyclical count of lost games.
         *
         */
        private int lostCountCyclic = 0;

        /**
         * Cyclical count of draw games.
         *
         */
        private int drawCountCyclic = 0;

        /**
         * Constructor for player.
         *
         * @param agent agent of player.
         * @param index player index
         */
        Player(Agent agent, int index) {
            this.agent = agent;
            this.index = index;
        }

        /**
         * Sets role of player.
         *
         * @param playerRole role of player.
         */
        public void setPlayerRole(PlayerRole playerRole) {
            this.playerRole = playerRole;
        }

        /**
         * Returns role of player.
         *
         * @return role of player.
         */
        public PlayerRole getPlayerRole() {
            return playerRole;
        }

        /**
         * Returns agent of player.
         *
         * @return agent of player.
         */
        public Agent getAgent() {
            return agent;
        }

        /**
         * Returns player index.
         *
         * @return player index.
         */
        public int getIndex() {
            return index;
        }

        /**
         * Sets player as human or as agent.
         *
         * @param isHuman if true player is human otherwise agent.
         */
        public void setAsHuman(boolean isHuman) {
            this.isHuman = isHuman;
        }

        /**
         * Returns flag is player is human or agent.
         *
         * @return if true player is human otherwise agent.
         */
        public boolean isHuman() {
            return isHuman;
        }

        /**
         * Marks game as won.
         *
         */
        public void won() {
            wonCount++;
            wonCountCyclic++;
        }

        /**
         * Marks game as lost.
         *
         */
        public void lost() {
            lostCount++;
            lostCountCyclic++;
        }

        /**
         * Marks game as draw.
         *
         */
        public void draw() {
            drawCount++;
            drawCountCyclic++;
        }

        /**
         * Resets cyclical counts.
         *
         */
        public void resetCyclicalCounts() {
            wonCountCyclic = 0;
            lostCountCyclic = 0;
            drawCountCyclic = 0;
        }

        /**
         * Returns total won count.
         *
         * @return total won count.
         */
        public int getWonCount() {
            return wonCount;
        }

        /**
         * Returns cyclical won count.
         *
         * @return cyclical won count.
         */
        public int getWonCountCyclic() {
            return wonCountCyclic;
        }

        /**
         * Returns total lost count.
         *
         * @return total lost count.
         */
        public int getLostCount() {
            return lostCount;
        }

        /**
         * Returns cyclical lost count.
         *
         * @return cyclical lost count.
         */
        public int getLostCountCyclic() {
            return lostCountCyclic;
        }

        /**
         * Returns total draw count.
         *
         * @return total draw count.
         */
        public int getDrawCount() {
            return drawCount;
        }

        /**
         * Returns cyclical draw count.
         *
         * @return cyclical draw count.
         */
        public int getDrawCountCyclic() {
            return drawCountCyclic;
        }

        /**
         * Compares two players based on
         *
         * @param otherPlayer other player
         * @return result of comparison
         */
        public int compareTo(Player otherPlayer) {
            return Double.compare(agent.getCumulativeReward(false), otherPlayer.getAgent().getCumulativeReward(false));
        }
    }

    /**
     * Game states.
     *
     */
    private enum GameStatus {
        ONGOING,
        NOUGHT_WON,
        CROSS_WON,
        DRAW
    }

    /**
     * Implements game board.
     *
     */
    private static class GameBoard {

        /**
         * Game states.
         *
         */
        private enum BoardSlot {
            NOUGHT,
            CROSS,
            EMPTY
        }

        /**
         * First player.
         *
         */
        private final PlayerRole firstPlayer;

        /**
         * Game board.
         *
         */
        final BoardSlot[][] gameBoard;

        /**
         * Current state.
         *
         */
        private Matrix state;

        /**
         * Available moves from current state.
         *
         */
        private HashSet<Integer> availableMoves;

        /**
         * Initializes game board.
         *
         */
        GameBoard(PlayerRole firstPlayer) {
            this.firstPlayer = firstPlayer;
            gameBoard = new BoardSlot[boardSize][boardSize];
            for (BoardSlot[] boardSlots : gameBoard) {
                Arrays.fill(boardSlots, BoardSlot.EMPTY);
            }
        }

        /**
         * Returns game board.
         *
         * @return game board.
         */
        public BoardSlot[][] getGameBoard() {
            return gameBoard;
        }

        /**
         * Updates state.
         *
         */
        public void updateState() {
            state = new DMatrix(getInputSize(), 1);
            availableMoves = new HashSet<>();
            for (int row = 0; row < gameBoard.length; row++) {
                for (int col = 0; col < gameBoard[row].length; col++) {
                    double positionValue = 0;
                    switch (gameBoard[row][col]) {
                        case NOUGHT -> positionValue = firstPlayer == PlayerRole.NOUGHT ? -1 : 1;
                        case CROSS -> positionValue = firstPlayer == PlayerRole.NOUGHT ? 1 : -1;
                        case EMPTY -> {
                            positionValue = 0;
                            availableMoves.add(getPos(row, col));
                        }
                    }
                    state.setValue(getPos(row, col), 0, positionValue);
                }
            }
        }

        /**
         * Returns current state of game (game board).
         *
         * @return state of game.
         */
        public Matrix getState() {
            return state;
        }

        /**
         * Return available moves from current state.
         *
         * @return available moves.
         */
        public HashSet<Integer> getAvailableMoves() {
            return availableMoves;
        }

        /**
         * Returns row and column as specific action value.
         *
         * @param row row of game board.
         * @param col column of game board.
         * @return action value calculated.
         */
        public int getPos(int row, int col) {
            return row + col * boardSize;
        }

        /**
         * Returns row and col in size 2 array based on given action.
         *
         * @param action action as input.
         * @return row and col in size 2 array
         */
        public int[] getPos(int action) {
            int[] pos = new int[2];
            pos[0] = action % boardSize;
            pos[1] = action / boardSize;
            return pos;
        }

        /**
         * Checks if taken move (action) is valid.
         *
         * @param row row of game board.
         * @param col column of game board.
         * @return true if move is valid otherwise false.
         */
        public boolean isValidMove(int row, int col) {
            if (row < 0 || row > boardSize - 1 || col < 0 || col > boardSize - 1) return false;
            else return gameBoard[row][col] == BoardSlot.EMPTY;
        }

        /**
         * Check if there are available moves.
         *
         * @return returns true if there are no available moves otherwise returns true.
         */
        private boolean noAvailableMoves() {
            return getAvailableMoves().isEmpty();
        }

        /**
         * Executes move per action defined.
         *
         * @param player current player.
         * @param row row of game board.
         * @param col column of game board.
         */
        public void makeMove(TicTacToe.Player player, int row, int col) {
            gameBoard[row][col] = player.playerRole == PlayerRole.NOUGHT ? BoardSlot.NOUGHT : BoardSlot.CROSS;
        }

        /**
         * Executes move per action defined.
         *
         * @param player current player.
         * @param action action defined.
         */
        public void makeMove(TicTacToe.Player player, int action) {
            int[] pos = getPos(action);
            makeMove(player, pos[0], pos[1]);
        }

        /**
         * Checks if player has won the game.
         *
         * @param player current player.
         * @return return player if player has won otherwise returns false.
         */
        private boolean checkWinner(TicTacToe.Player player) {
            int diagonalStatistics = 0;
            int adiagonalStatistics = 0;
            BoardSlot gameSlotType = player.playerRole == PlayerRole.NOUGHT ? BoardSlot.NOUGHT : BoardSlot.CROSS;
            for (int row = 0; row < boardSize; row++) {
                int rowStat = 0;
                int colStat = 0;
                for (int col = 0; col < boardSize; col++) {
                    if (gameBoard[row][col] == gameSlotType) rowStat++; else rowStat = 0;
                    if (gameBoard[col][row] == gameSlotType) colStat++; else colStat = 0;
                }
                if (rowStat == boardSize || colStat == boardSize) return true;

                if (gameBoard[row][row] == gameSlotType) diagonalStatistics++; else diagonalStatistics = 0;
                if (gameBoard[row][boardSize - 1 - row] == gameSlotType) adiagonalStatistics++; else adiagonalStatistics = 0;
            }

            return diagonalStatistics == boardSize || adiagonalStatistics == boardSize;
        }

        /**
         * Updates state of game after action is taken.
         *
         * @param player current player.
         * @return state of game.
         */
        public GameStatus updateGameStatus(TicTacToe.Player player) {
            if (checkWinner(player)) return player.playerRole == PlayerRole.NOUGHT ? GameStatus.NOUGHT_WON : GameStatus.CROSS_WON;
            else {
                if (noAvailableMoves()) return GameStatus.DRAW;
                else return GameStatus.ONGOING;
            }
        }

    }

    /**
     * Implements JPanel into which Tic-Tac-Toe game is drawn.<br>
     *
     */
    private class TicTacToePanel extends JPanel {

        /**
         * Game board for drawing maze.
         *
         */
        private GameBoard.BoardSlot[][] gameBoard;

        /**
         * Status of game.
         *
         */
        private GameStatus gameStatus;

        /**
         * Current human player.
         *
         */
        private PlayerRole currentHumanPlayerRole;

        /**
         * Sets current human player
         *
         * @param currentHumanPlayerRole current human player.
         */
        public void setHumanPlayer(PlayerRole currentHumanPlayerRole) {
            this.currentHumanPlayerRole = currentHumanPlayerRole;
        }

        /**
         * Sets gameBoard to be drawn.
         *
         * @param gameBoard game board to be drawn.
         * @param gameStatus status of game.
         */
        public void setGameBoard(GameBoard.BoardSlot[][] gameBoard, GameStatus gameStatus) {
            this.gameBoard = gameBoard;
            this.gameStatus = gameStatus;
        }

        /**
         * Paints gameBoard to JPanel.
         *
         * @param g graphics.
         */
        public void paintComponent(Graphics g) {
            panelLock.lock();
            if (gameBoard == null) return;
            super.paintComponent(g);
            boolean darkColor = true;
            for (int row = 0; row < boardSize; row++) {
                for (int col = 0; col < boardSize; col++) {
                    if (darkColor) g.setColor(Color.GRAY);
                    else g.setColor(Color.LIGHT_GRAY);
                    g.fillRect(row * tileSize, col * tileSize, tileSize, tileSize);
                    darkColor = !darkColor;
                    g.setColor(Color.BLACK);
                }
            }
            for (int row = 0; row < boardSize; row++) {
                for (int col = 0; col < boardSize; col++) {
                    g.setFont(new Font("ARIAL", Font.PLAIN, 32));
                    if (gameBoard != null) {
                        if (gameBoard[row][col] == GameBoard.BoardSlot.NOUGHT) g.drawString("O", col * tileSize + 40, row * tileSize + 60);
                        if (gameBoard[row][col] == GameBoard.BoardSlot.CROSS) g.drawString("X", col * tileSize + 40, row * tileSize + 60);
                    }
                }
            }
            if (currentHumanPlayerRole != null && gameStatus != GameStatus.ONGOING) {
                g.setFont(new Font("ARIAL", Font.PLAIN, 32));
                g.setColor(Color.RED);
                if (gameStatus == GameStatus.DRAW) g.drawString("DRAW", 100, 165);
                if (gameStatus == GameStatus.NOUGHT_WON) g.drawString("NOUGHT WON", 50, 165);
                if (gameStatus == GameStatus.CROSS_WON) g.drawString("CROSS WON", 50, 165);
            }
            panelLock.unlock();
        }

    }

    /**
     * Size of game board (size x size)
     *
     */
    private static final int boardSize = 3;

    /**
     * Game count.
     *
     */
    private int gameCount = 0;

    /**
     * List of players participating into game.
     *
     */
    private final ArrayList<Player> playerList = new ArrayList<>();

    /**
     * List of current players in game.
     *
     */
    private ArrayList<Player> currentPlayerList = new ArrayList<>();

    /**
     * Current active player.
     *
     */
    private int currentPlayer = 0;

    /**
     * Game board.
     *
     */
    private GameBoard gameBoard;

    /**
     * Current environment state.
     *
     */
    private EnvironmentState environmentState;

    /**
     * Current human player role if active.
     *
     */
    private PlayerRole humanPlayerRole = null;

    /**
     * Row into where human player wants to make a move into.
     *
     */
    private int humanRow = -1;

    /**
     * Column into where human player wants to make a move into.
     *
     */
    private int humanCol = -1;

    /**
     * Status of game.
     *
     */
    private GameStatus gameStatus;

    /**
     * Random number generator.
     *
     */
    private final Random random = new Random();

    /**
     * Tile size in pixels (tileSize x tileSize)
     *
     */
    private final int tileSize = 100;

    /**
     * JFrame for Tic Tac Toe.
     *
     */
    private JFrame jFrame;

    /**
     * Root panel holding both Tic-Tac-Toe and radio button panels.
     *
     */
    private final JPanel jRootPanel = new JPanel();

    /**
     * Tic-Tac-Toe panel that is used to show Tic-Tac-Toe game grid.
     *
     */
    private final TicTacToePanel ticTacToePanel = new TicTacToePanel();

    /**
     * Panel that holds radio buttons.
     *
     */
    private final JPanel jRadioButtonPanel = new JPanel();

    /**
     * Radio button that is used to choose nought as human game role.
     *
     */
    private final JRadioButton noughtRadioButton = new JRadioButton("Nought");

    /**
     * Radio button that is used to choose auto mode where agents are playing against each other (default choice).
     *
     */
    private final JRadioButton autoRadioButton = new JRadioButton("Auto", true);

    /**
     * Radio button that is used to choose cross as human game role.
     *
     */
    private final JRadioButton crossRadioButton = new JRadioButton("Cross");

    /**
     * Button group for game mode selection.
     *
     */
    private final ButtonGroup gameModeButtonGroup = new ButtonGroup();

    /**
     * Lock that is used to synchronize GUI (radio button events) and Tic-Tac-Toe threads with each other.
     *
     */
    private final Lock lock = new ReentrantLock();

    /**
     * Lock that is used to synchronize GUI panel and Tic-Tac-Toe threads with each other.
     *
     */
    private final Lock panelLock = new ReentrantLock();

    /**
     * Condition for human action lock.
     *
     */
    private final Condition humanAction = lock.newCondition();

    /**
     * Main function for Tic-Tac-Toe.
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
     * Constructor for Tic-Tac-Toe.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException throws exception if setting of dynamic parameter fails.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public TicTacToe() throws NeuralNetworkException, MatrixException, DynamicParamException, IOException, ClassNotFoundException, AgentException {
        int numberfOfAgents = 4;
        boolean singleFunctionEstimator = false;
        boolean sharedPolicyFunctionEstimator = false;
        boolean sharedValueFunctionEstimator = false;
        boolean sharedMemory = false;
        int policyType = 4;
        ExecutablePolicyType executablePolicyType = null;
        String policyTypeParams = "";
        switch (policyType) {
            case 0 -> executablePolicyType = ExecutablePolicyType.GREEDY;
            case 1 -> {
                executablePolicyType = ExecutablePolicyType.EPSILON_GREEDY;
                policyTypeParams += "epsilonInitial = 0.5, epsilonMin = 0.09";
            }
            case 2 -> {
                executablePolicyType = ExecutablePolicyType.NOISY_NEXT_BEST;
                policyTypeParams += "initialExplorationNoise = 1, minExplorationNoise = 0.2";
            }
            case 3 -> {
                executablePolicyType = ExecutablePolicyType.SAMPLED;
                policyTypeParams += "thresholdInitial = 1, thresholdMin = 0.3";
            }
            case 4 -> executablePolicyType = ExecutablePolicyType.ENTROPY_GREEDY;
            case 5 -> executablePolicyType = ExecutablePolicyType.ENTROPY_NOISY_NEXT_BEST;
        }

        AgentFactory.AgentAlgorithmType agentAlgorithmType = AgentFactory.AgentAlgorithmType.DDQN;
        boolean onlineMemory = switch (agentAlgorithmType) {
            case DDQN, DDPG, SACDiscrete -> false;
            default -> true;
        };
        boolean applyDueling = switch (agentAlgorithmType) {
            case DQN -> true;
            default -> false;
        };
        String algorithmParams = switch (agentAlgorithmType) {
            case QN -> "agentUpdateCycle = 1, learningRate = 0.001";
            case DDQN -> "applyImportanceSamplingWeights = true, applyUniformSampling = false, capacity = 20000, batchSize = 128, targetFunctionUpdateCycle = 0, targetFunctionTau = 0.01, agentUpdateCycle = 10";
            case SACDiscrete -> "applyImportanceSamplingWeights = false, applyUniformSampling = true, capacity = 20000, targetFunctionUpdateCycle = 0, targetFunctionTau = 0.01";
            case MCTS -> "gamma = 1, updateValuePerEpisode = true";
            default -> "";
        };

        String params = "";
        if (policyTypeParams.isEmpty() && !algorithmParams.isEmpty()) params = algorithmParams;
        if (!policyTypeParams.isEmpty() && algorithmParams.isEmpty()) params = policyTypeParams;
        if (!policyTypeParams.isEmpty() && !algorithmParams.isEmpty()) params = policyTypeParams + ", " + algorithmParams;

        Agent agent = null;
        for (int agentCount = 0; agentCount < numberfOfAgents; agentCount++) {
            if (agent == null) agent = AgentFactory.createAgent(this, agentAlgorithmType, this, getInputSize(), getOutputSize(), onlineMemory, singleFunctionEstimator, applyDueling, executablePolicyType, params);
            else agent = AgentFactory.createAgent(agent, sharedPolicyFunctionEstimator, sharedValueFunctionEstimator, sharedMemory);
            agent.start();
            playerList.add(new Player(agent, agentCount));
        }
    }

    /**
     * Builds neural network for Tic-Tac-Toe player (agent).
     *
     * @param inputSize input size of neural network (number of states).
     * @param outputSize output size of neural network (number of actions and their values).
     * @param policyGradient if true neural network is policy gradient network.
     * @param applyDueling if true applied dueling layer to non policy gradient network otherwise not.
     * @return built neural network
     * @throws DynamicParamException throws exception if setting of dynamic parameters fails.
     * @throws NeuralNetworkException throws exception if building of neural network fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public NeuralNetwork buildNeuralNetwork(int inputSize, int outputSize, boolean policyGradient, boolean applyDueling) throws DynamicParamException, NeuralNetworkException, MatrixException {
        NeuralNetworkConfiguration neuralNetworkConfiguration = new NeuralNetworkConfiguration();
        neuralNetworkConfiguration.addInputLayer("width = " + inputSize);
        neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), "width = 100");
        neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), "width = 100");
        neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, !policyGradient ? new ActivationFunction(UnaryFunctionType.RELU) : new ActivationFunction(UnaryFunctionType.SOFTMAX), "width = " + outputSize);
        if (!policyGradient && applyDueling) neuralNetworkConfiguration.addHiddenLayer(LayerType.DUELING, "width = " + outputSize);
        neuralNetworkConfiguration.addOutputLayer(!policyGradient ? BinaryFunctionType.MEAN_SQUARED_ERROR : BinaryFunctionType.DIRECT_GRADIENT);
        neuralNetworkConfiguration.connectLayersSerially();

        NeuralNetwork neuralNetwork = new NeuralNetwork(neuralNetworkConfiguration);

        neuralNetwork.setOptimizer(OptimizationType.ADAM);
        if (!policyGradient) neuralNetwork.verboseTraining(10);
        neuralNetwork.setNeuralNetworkName("TicTacToe");
        return neuralNetwork;
    }

    /**
     * Builds neural network for Tic-Tac-Toe player (agent).
     *
     * @param outputSize output size of neural network (number of actions and their values).
     * @return built neural network
     * @throws DynamicParamException throws exception if setting of dynamic parameters fails.
     * @throws NeuralNetworkException throws exception if building of neural network fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public NeuralNetwork buildNeuralNetwork(int inputSize, int outputSize) throws DynamicParamException, NeuralNetworkException, MatrixException {
        NeuralNetworkConfiguration neuralNetworkConfiguration = new NeuralNetworkConfiguration();
        int inputLayerIndex = neuralNetworkConfiguration.addInputLayer("width = " + inputSize);
        int hiddenLayerIndex1 = neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), "width = 100");
        int hiddenLayerIndex2 = neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), "width = 100");
        int hiddenLayerIndex3 = neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.SOFTMAX), "width = " + outputSize);
        int outputLayerIndex1 = neuralNetworkConfiguration.addOutputLayer(BinaryFunctionType.DIRECT_GRADIENT);
        neuralNetworkConfiguration.connectLayersSerially();

        int hiddenLayerIndex4 = neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.SINACT), "width = 1");
        int outputLayerIndex2 = neuralNetworkConfiguration.addOutputLayer(BinaryFunctionType.MEAN_SQUARED_ERROR);
        neuralNetworkConfiguration.connectLayers(hiddenLayerIndex2, hiddenLayerIndex4);
        neuralNetworkConfiguration.connectLayers(hiddenLayerIndex4, outputLayerIndex2);

        NeuralNetwork neuralNetwork = new NeuralNetwork(neuralNetworkConfiguration);

        neuralNetwork.verboseTraining(10);
        neuralNetwork.setOptimizer(OptimizationType.RADAM);
        neuralNetwork.setNeuralNetworkName("TicTacToe");
        return neuralNetwork;
    }

    /**
     * Returns input size of the board.
     *
     * @return input size.
     */
    public static int getInputSize() {
        return boardSize * boardSize;
    }

    /**
     * Returns output size of the board.
     *
     * @return output size.
     */
    public static int getOutputSize() {
        return boardSize * boardSize;
    }

    /**
     * Reward structure i.e. rewards returned to agent during game per action taken.
     *
     */
    public static class RewardStructure {
        final double WIN = 1;
        final double DRAW = 0.5;
        final double LOST = 0;
        final double MOVE = 0;
    }

    /**
     * Reward structure.
     *
     */
    private final RewardStructure rewardStructure = new RewardStructure();

    /**
     * Initializes window for maze.
     *
     */
    private void initWindow() {
        JFrame.setDefaultLookAndFeelDecorated(true);
        jFrame = new JFrame("Tic Tac Toe");
        jFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        jFrame.setBackground(Color.WHITE);
        jFrame.setSize(boardSize * tileSize, boardSize * tileSize + 60);

        jRootPanel.setLayout(new BorderLayout());
        jRootPanel.setSize(boardSize * tileSize, boardSize * tileSize + 60);
        jFrame.add(jRootPanel);

        noughtRadioButton.addActionListener(this);
        autoRadioButton.addActionListener(this);
        crossRadioButton.addActionListener(this);
        gameModeButtonGroup.add(noughtRadioButton);
        gameModeButtonGroup.add(autoRadioButton);
        gameModeButtonGroup.add(crossRadioButton);
        jRadioButtonPanel.add(noughtRadioButton);
        jRadioButtonPanel.add(autoRadioButton);
        jRadioButtonPanel.add(crossRadioButton);
        jRadioButtonPanel.setSize(new Dimension(boardSize * tileSize, 30));
        jRootPanel.add(jRadioButtonPanel, BorderLayout.PAGE_START);

        ticTacToePanel.setSize(boardSize * tileSize, boardSize * tileSize);
        ticTacToePanel.addMouseListener(this);
        jRootPanel.add(ticTacToePanel, BorderLayout.CENTER);

        jFrame.setVisible(true);
    }

    /**
     * Handles action taken in radio button panel to choose game mode.
     *
     * @param e action event originated from radio button panel.
     */
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == noughtRadioButton) humanPlayerRole = PlayerRole.NOUGHT;
        if (e.getSource() == autoRadioButton) humanPlayerRole = null;
        if (e.getSource() == crossRadioButton) humanPlayerRole = PlayerRole.CROSS;
    }

    /**
     * Not used.
     *
     * @param e mouse event.
     */
    public void mousePressed (MouseEvent e) {
        lock.lock();
        humanRow = e.getY() / tileSize;
        humanCol = e.getX() / tileSize;
        humanAction.signal();
        lock.unlock();
    }

    /**
     * Not used.
     *
     * @param e mouse event.
     */
    public void mouseExited (MouseEvent e) {}

    /**
     * Not used.
     *
     * @param e mouse event.
     */
    public void mouseReleased (MouseEvent e) {}

    /**
     * Not used.
     *
     * @param e mouse event.
     */
    public void mouseEntered (MouseEvent e) {}

    /**
     * Handles mouse click action taking place in Tic-Tac-Toe panel.
     *
     * @param e mouse event.
     */
    public void mouseClicked (MouseEvent e) {
    }

    /**
     * Plays given number of games.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private void playGames() throws MatrixException, NeuralNetworkException, DynamicParamException, AgentException, IOException, ClassNotFoundException {
        initWindow();
        jFrame.revalidate();

        String stringFormat = "%.3f";
        int numberOfGames = 500000000;

        for (int game = 0; game < numberOfGames; game++) {
            playGame();
            if (game % 100 == 0 && game > 0) {
                System.out.print("Game #" + game);
                for (Player player : currentPlayerList) {
                    System.out.print(" Player #" + player.getIndex() + " (won: " + player.getWonCountCyclic() + ", lost: " + player.getLostCountCyclic() + ", draw: " + player.getDrawCountCyclic() + ")");
                }
                System.out.println();
                System.out.print("Learned / non-learned cumulative reward:");
                for (Player player : playerList) {
                    System.out.print(" Player #" + player.getIndex() + " (" + player.getAgent().getCumulativeReward(true) + " / " + player.getAgent().getCumulativeReward(false) + ")");
                }
                System.out.println();
                System.out.print("Learned / non-learned moving average reward:");
                Collections.sort(playerList);
                for (Player player : playerList) {
                    System.out.print(" Player #" + player.getIndex() + " (" + String.format(stringFormat, player.getAgent().getMovingAverageReward(true)) + " / " + String.format(stringFormat, player.getAgent().getMovingAverageReward(false)) + ")");
                }
                System.out.println();
                if (game % 1000 == 0) {
                    for (Player player : playerList) {
                        player.getAgent().resetRewardMetrics();
                        player.resetCyclicalCounts();
                    }
                }
            }
        }
        for (int playerIndex = 0; playerIndex < currentPlayerList.size(); playerIndex++) {
            Player player = currentPlayerList.get(playerIndex);
            System.out.println("Player#" + playerIndex + " won games: " + player.getWonCount());
            System.out.println("Player#" + playerIndex + " lost games: " + player.getLostCount());
            System.out.println("Player#" + playerIndex + "1 draw games: " + player.getDrawCount());
            player.getAgent().stop();
        }
    }

    /**
     * Plays single episode of game.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    private void playGame() throws MatrixException, NeuralNetworkException, DynamicParamException, AgentException, IOException, ClassNotFoundException {
        PlayerRole currentHumanPlayerRole = humanPlayerRole;

        currentPlayerList = new ArrayList<>();

        if (currentHumanPlayerRole == null) {
            Collections.shuffle(playerList);

            currentPlayerList.add(playerList.get(0));
            currentPlayerList.get(0).setPlayerRole(PlayerRole.NOUGHT);
            currentPlayerList.add(playerList.get(playerList.size() - 1));
            currentPlayerList.get(1).setPlayerRole(PlayerRole.CROSS);

            currentPlayerList.get(0).setAsHuman(false);
            currentPlayerList.get(1).setAsHuman(false);

            boolean enableLearning = random.nextDouble() < 0.8;
            for (Player player : currentPlayerList) {
                if (enableLearning) player.getAgent().enableLearning();
                else player.getAgent().disableLearning();
            }
        }
        else {
            Collections.sort(playerList);

            switch (currentHumanPlayerRole) {
                case NOUGHT -> {
                    currentPlayerList.add(playerList.get(0));
                    currentPlayerList.get(0).setPlayerRole(PlayerRole.NOUGHT);
                    currentPlayerList.get(0).setAsHuman(true);
                    currentPlayerList.add(playerList.get(playerList.size() - 1));
                    currentPlayerList.get(1).setPlayerRole(PlayerRole.CROSS);
                    currentPlayerList.get(1).setAsHuman(false);
                }
                case CROSS -> {
                    currentPlayerList.add(playerList.get(playerList.size() - 1));
                    currentPlayerList.get(0).setPlayerRole(PlayerRole.NOUGHT);
                    currentPlayerList.get(0).setAsHuman(false);
                    currentPlayerList.add(playerList.get(0));
                    currentPlayerList.get(1).setPlayerRole(PlayerRole.CROSS);
                    currentPlayerList.get(1).setAsHuman(true);
                }
            }

            for (Player player : currentPlayerList) player.getAgent().disableLearning();
        }

        Collections.shuffle(currentPlayerList);

        gameBoard = new GameBoard(currentPlayerList.get(0).getPlayerRole());
        gameStatus = GameStatus.ONGOING;

        panelLock.lock();
        ticTacToePanel.setHumanPlayer(currentHumanPlayerRole);
        ticTacToePanel.setGameBoard(gameBoard.getGameBoard(), gameStatus);
        panelLock.unlock();

        currentPlayer = 0;

        for (Player player : currentPlayerList) player.getAgent().newEpisode();

        gameCount++;
        int timeStep = 0;
        while (gameStatus == GameStatus.ONGOING) {
            gameBoard.updateState();
            environmentState = new EnvironmentState(gameCount, ++timeStep, gameBoard.getState(), gameBoard.getAvailableMoves());
            panelLock.lock();
            ticTacToePanel.setGameBoard(gameBoard.getGameBoard(), gameStatus);
            panelLock.unlock();
            jFrame.revalidate();
            ticTacToePanel.paintImmediately(0, 0, boardSize * tileSize, boardSize * tileSize + 60);
            currentPlayerList.get(currentPlayer).getAgent().newStep();
            if (currentPlayerList.get(currentPlayer).isHuman()) {
                lock.lock();
                humanRow = -1;
                humanCol = -1;
                try {
                    while (!gameBoard.isValidMove(humanRow, humanCol)) humanAction.await();
                }
                catch (InterruptedException exception) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException(exception);
                }
                lock.unlock();
                currentPlayerList.get(currentPlayer).getAgent().act(gameBoard.getPos(humanRow, humanCol));
            }
            else currentPlayerList.get(currentPlayer).getAgent().act(currentHumanPlayerRole != null);

            if (gameStatus == GameStatus.ONGOING) currentPlayer = currentPlayer == 0 ? 1 : 0;
        }
        for (Player player : currentPlayerList) player.getAgent().endEpisode();

        panelLock.lock();
        ticTacToePanel.setGameBoard(gameBoard.getGameBoard(), gameStatus);
        panelLock.unlock();

        jFrame.revalidate();
        ticTacToePanel.paintImmediately(0, 0, boardSize * tileSize, boardSize * tileSize + 60);

        if (currentHumanPlayerRole != null) {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException exception) {
                Thread.currentThread().interrupt();
                throw new RuntimeException(exception);
            }
        }

    }

    /**
     * Returns true if environment is episodic otherwise false.
     *
     * @return true if environment is episodic otherwise false.
     */
    public boolean isEpisodic() {
        return true;
    }

    /**
     * Returns current state of environment.
     *
     * @return state of environment.
     */
    public EnvironmentState getState() {
        return environmentState;
    }

    /**
     * Takes specific action.
     *
     * @param agent agent that is taking action.
     * @param action action to be taken.
     */
    public void commitAction(Agent agent, int action) {
        gameBoard.makeMove(currentPlayerList.get(currentPlayer), action);
        gameBoard.updateState();
        gameStatus = gameBoard.updateGameStatus(currentPlayerList.get(currentPlayer));

        switch (gameStatus) {
            case ONGOING:
                currentPlayerList.get(currentPlayer).getAgent().respond(rewardStructure.MOVE);
                break;
            case NOUGHT_WON:
                for (Player player : currentPlayerList) {
                    if (!player.isHuman()) {
                        if (player.getPlayerRole() == PlayerRole.NOUGHT) {
                            player.won();
                            player.getAgent().respond(rewardStructure.WIN);
                        }
                        if (player.getPlayerRole() == PlayerRole.CROSS) {
                            player.lost();
                            player.getAgent().respond(rewardStructure.LOST);
                        }
                    }
                }
                break;
            case CROSS_WON:
                for (Player player : currentPlayerList) {
                    if (!player.isHuman()) {
                        if (player.getPlayerRole() == PlayerRole.CROSS) {
                            player.won();
                            player.getAgent().respond(rewardStructure.WIN);
                        }
                        if (player.getPlayerRole() == PlayerRole.NOUGHT) {
                            player.lost();
                            player.getAgent().respond(rewardStructure.LOST);
                        }
                    }
                }
                break;
            case DRAW:
                for (Player player : currentPlayerList) {
                    if (!player.isHuman()) {
                        player.draw();
                        player.getAgent().respond(rewardStructure.DRAW);
                    }
                }
                break;
            default:
                break;
        }
    }

}
