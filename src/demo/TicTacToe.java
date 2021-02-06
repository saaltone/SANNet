/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package demo;

import core.NeuralNetwork;
import core.NeuralNetworkException;
import core.activation.ActivationFunction;
import core.layer.LayerType;
import core.normalization.NormalizationType;
import core.optimization.OptimizationType;
import core.reinforcement.*;
import core.reinforcement.algorithm.*;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.function.NNFunctionEstimator;
import core.reinforcement.function.TabularFunctionEstimator;
import core.reinforcement.memory.Memory;
import core.reinforcement.memory.OnlineMemory;
import core.reinforcement.memory.PriorityMemory;
import core.reinforcement.policy.*;
import core.reinforcement.policy.executablepolicy.*;
import core.reinforcement.value.*;
import utils.*;
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
 * Class that implements tic tac toe game using deep reinforcement learning.
 *
 */
public class TicTacToe implements Environment, ActionListener, MouseListener {

    /**
     * Player.
     *
     */
    private enum Player {
        NOUGHT,
        CROSS,
    }

    /**
     * Game states.
     *
     */
    private enum GameSlot {
        NOUGHT,
        CROSS,
        EMPTY
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
     * Class that defines game board.
     *
     */
    private class GameBoard {

        /**
         * Game board.
         *
         */
        final GameSlot[][] gameBoard;

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
        GameBoard() {
            gameBoard = new GameSlot[boardSize][boardSize];
            for (GameSlot[] gameSlots : gameBoard) {
                Arrays.fill(gameSlots, GameSlot.EMPTY);
            }
        }

        /**
         * Returns game board.
         *
         * @return game board.
         */
        public GameSlot[][] getGameBoard() {
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
                    switch(gameBoard[row][col]) {
                        case NOUGHT:
                            positionValue = -1;
                            break;
                        case CROSS:
                            positionValue = 1;
                            break;
                        case EMPTY:
                            positionValue = 0;
                            availableMoves.add(getPos(row, col));
                            break;
                    }
                    if (canonicalGameBoard) positionValue = positionValue * (currentPlayer == Player.NOUGHT ? 1 : -1);
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
            else return gameBoard[row][col] == GameSlot.EMPTY;
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
        public void makeMove(Player player, int row, int col) {
            gameBoard[row][col] = player == Player.NOUGHT ? GameSlot.NOUGHT : GameSlot.CROSS;
        }

        /**
         * Executes move per action defined.
         *
         * @param player current player.
         * @param action action defined.
         */
        public void makeMove(Player player, int action) {
            int[] pos = getPos(action);
            makeMove(player, pos[0], pos[1]);
        }

        /**
         * Checks if player has won the game.
         *
         * @param player current player.
         * @return return player if player has won otherwise returns null.
         */
        private Player checkWinner(Player player) {
            int diagonalStatistics = 0;
            int adiagonalStatistics = 0;
            GameSlot targetSlot = player == Player.NOUGHT ? GameSlot.NOUGHT : GameSlot.CROSS;
            for (int row = 0; row < boardSize; row++) {
                int rowStat = 0;
                int colStat = 0;
                for (int col = 0; col < boardSize; col++) {
                    if (gameBoard[row][col] == targetSlot) rowStat++; else rowStat = 0;
                    if (gameBoard[col][row] == targetSlot) colStat++; else colStat = 0;
                }
                if (rowStat == boardSize || colStat == boardSize) return player;

                if (gameBoard[row][row] == targetSlot) diagonalStatistics++; else diagonalStatistics = 0;
                if (gameBoard[row][boardSize - 1 - row] == targetSlot) adiagonalStatistics++; else adiagonalStatistics = 0;
            }

            if (diagonalStatistics == boardSize || adiagonalStatistics == boardSize) return player;

            return null;
        }

        /**
         * Updates state of game after action is taken.
         *
         * @return state of game.
         */
        public GameStatus updateGameStatus(Player player) {
            Player winner = checkWinner(player);
            if (winner == Player.NOUGHT) return GameStatus.NOUGHT_WON;
            if (winner == Player.CROSS) return GameStatus.CROSS_WON;
            if (winner == null && noAvailableMoves()) return GameStatus.DRAW;
            return gameStatus;
        }

    }

    /**
     * Implements JPanel into which tic tac toe game is drawn.<br>
     *
     */
    private class TicTacToePanel extends JPanel {

        /**
         * Game board for drawing maze.
         *
         */
        private GameSlot[][] gameBoard;

        /**
         * Status of game.
         *
         */
        private GameStatus gameStatus;

        /**
         * Current human player.
         *
         */
        private Player currentHumanPlayer;

        /**
         * Sets current human player
         *
         * @param currentHumanPlayer current human player.
         */
        public void setHumanPlayer(Player currentHumanPlayer) {
            this.currentHumanPlayer = currentHumanPlayer;
        }

        /**
         * Sets gameBoard to be drawn.
         *
         * @param gameBoard game board to be drawn.
         * @param gameStatus status of game.
         */
        public void setGameBoard(GameSlot[][] gameBoard, GameStatus gameStatus) {
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
                        if (gameBoard[row][col] == GameSlot.NOUGHT) g.drawString("O", col * tileSize + 40, row * tileSize + 60);
                        if (gameBoard[row][col] == GameSlot.CROSS) g.drawString("X", col * tileSize + 40, row * tileSize + 60);
                    }
                }
            }
            if (currentHumanPlayer != null && gameStatus != GameStatus.ONGOING) {
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
     * If true game board state is in canonical form i.e. game board of opposite layer is defined as reverse.
     *
     */
    private static final boolean canonicalGameBoard = true;

    /**
     * Episode ID
     *
     */
    private int episodeID = 0;

    /**
     * Deep agent for player nought.
     *
     */
    private final Agent nought;

    /**
     * Deep agent for player cross.
     *
     */
    private final Agent cross;

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
     * Current player (actor) of game.
     *
     */
    private Player currentPlayer = null;

    /**
     * Current human player if selected.
     *
     */
    private Player humanPlayer = null;

    /**
     * Human player for current game.
     *
     */
    private Player currentHumanPlayer;

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
     * Root panel holding both tic tac toe and radio button panels.
     *
     */
    private final JPanel jRootPanel = new JPanel();

    /**
     * Tic tac toe panel that is used to show tic tac toe game grid.
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
     * Lock that is used to synchronize GUI (radio button events) and tic tac toe threads with each other.
     *
     */
    private final Lock lock = new ReentrantLock();

    /**
     * Lock that is used to synchronize GUI panel and tic tac toe threads with each other.
     *
     */
    private final Lock panelLock = new ReentrantLock();

    /**
     * Condition for human action lock.
     *
     */
    private final Condition humanAction = lock.newCondition();

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
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if setting of dynamic parameter fails.
     * @throws IOException throws exception if copying of neural network instance fails.
     * @throws ClassNotFoundException throws exception if copying of neural network instance fails.
     */
    public TicTacToe() throws NeuralNetworkException, MatrixException, DynamicParamException, IOException, ClassNotFoundException {
        boolean nnPolicyEstimator = false;
        boolean nnValueEstimator = false;
        boolean policyGradient = false;
        boolean stateValue = false;
        Memory estimatorMemory;
        estimatorMemory = true ? new OnlineMemory() : new PriorityMemory();
        FunctionEstimator policyEstimator;
        FunctionEstimator valueEstimator;
        if (false) {
            // Uses single neural network estimator for both policy and value functions (works for policy gradients).
            NeuralNetwork stateActionValueNN = buildNeuralNetwork(getInputSize(), getOutputSize());
            policyEstimator = new NNFunctionEstimator(estimatorMemory, stateActionValueNN, getOutputSize(), true);
            valueEstimator = new NNFunctionEstimator(estimatorMemory, stateActionValueNN, 1, true);
        }
        else {
            // Uses separate estimators for value and policy functions.
            policyEstimator = nnPolicyEstimator ? new NNFunctionEstimator(estimatorMemory, buildNeuralNetwork(getInputSize(), getOutputSize(), policyGradient, false), getOutputSize()) : new TabularFunctionEstimator(estimatorMemory, getOutputSize());
            valueEstimator = nnValueEstimator ? new NNFunctionEstimator(estimatorMemory, buildNeuralNetwork(getInputSize(), getOutputSize(), false, stateValue), (stateValue ? 1 : getOutputSize())) : new TabularFunctionEstimator(estimatorMemory, getOutputSize());
        }
//        policyEstimator = nnPolicyEstimator ? new NNFunctionEstimator(estimatorMemory, buildNeuralNetwork(getInputSize(), getOutputSize(), policyGradient, false), getOutputSize()) : new TabularFunctionEstimator(estimatorMemory, getOutputSize());
//        valueEstimator = nnValueEstimator ? new NNFunctionEstimator(estimatorMemory, buildNeuralNetwork(getInputSize(), getOutputSize(), false, stateValue), (stateValue ? 1 : getOutputSize())) : new TabularFunctionEstimator(estimatorMemory, getOutputSize());
//        valueEstimator = nnValueEstimator ? new NNFunctionEstimator(estimatorMemory, buildNeuralNetwork(getInputSize(), getOutputSize()), getOutputSize(), "numberOfIterations = 10") : new TabularFunctionEstimator(estimatorMemory, getOutputSize());
        UpdateableMCTSPolicy updateableMCTSPolicy = new UpdateableMCTSPolicy(policyEstimator, "resetCycle = 0, cPUCT = 2.75, alpha = 0.6, epsilon = 0.8, tau = 1.1");
        nought = createAgent(valueEstimator, policyEstimator, updateableMCTSPolicy, estimatorMemory, policyGradient);
        cross = createAgent(valueEstimator, policyEstimator, updateableMCTSPolicy, estimatorMemory, policyGradient);
    }

    /**
     * Creates agent (player).
     *
     * @return agent
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException throws exception if setting of dynamic parameter fails.
     */
    private Agent createAgent(FunctionEstimator valueEstimator, FunctionEstimator policyEstimator, Policy policy, Memory estimatorMemory, boolean policyGradient) throws MatrixException, NeuralNetworkException, DynamicParamException, IOException, ClassNotFoundException {
        int policyType = 1;
        boolean basicPolicy = false;
        ExecutablePolicy executablePolicy = null;
        switch (policyType) {
            case 0:
                executablePolicy = new GreedyPolicy();
                break;
            case 1:
                executablePolicy = new EpsilonGreedyPolicy("epsilonInitial = 0.2, epsilonMin = 0.2");
                break;
            case 2:
                executablePolicy = new NoisyNextBestPolicy();
                break;
        }
        Agent agent;
        if (!policyGradient) {
            agent = new DQNLearning(this, new ActionablePolicy(executablePolicy, valueEstimator), new QValueFunctionEstimator(valueEstimator, "lambda = 1"));
//            agent = new DDQNLearning(this, new ActionablePolicy(executablePolicy, valueEstimator), new QTargetValueFunctionEstimator(valueEstimator, "lambda = 1"));
//            agent = new Sarsa(this, new ActionablePolicy(executablePolicy, valueEstimator), new ActionValueFunctionEstimator(valueEstimator));
        }
        else {
//            actionablePolicy = basicPolicy ? new UpdateableBasicPolicy(executablePolicy, policyEstimator) : new UpdateableProximalPolicy(executablePolicy, policyEstimator);
//            agent = new PolicyGradient(this, actionablePolicy, new PlainValueFunction(new DirectFunctionEstimator(estimatorMemory, size * size)));
//            agent = new ActorCritic(this, actionablePolicy, new StateValueFunctionEstimator(valueEstimator, "lambda = 1"));
            agent = new MCTSLearning(this, (UpdateableMCTSPolicy) policy, new StateValueFunctionEstimator(valueEstimator, "gamma = 1"), "updateValuePerEpisode = true");
        }
        agent.start();
        return agent;
    }

    /**
     * Builds neural network for tic tac toe player (agent).
     *
     * @param outputSize output size of neural network (number of actions and their values).
     * @return built neural network
     * @throws DynamicParamException throws exception if setting of dynamic parameters fails.
     * @throws NeuralNetworkException throws exception if building of neural network fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    private static NeuralNetwork buildNeuralNetwork(int inputSize, int outputSize, boolean policyGradient, boolean stateValue) throws DynamicParamException, NeuralNetworkException, MatrixException {
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.addInputLayer("width = " + inputSize);
//        neuralNetwork.addHiddenLayer(LayerType.GRU, new ActivationFunction(UnaryFunctionType.RELU), "width = 100");
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.ELU), "width = 100");
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.ELU), "width = 100");
        if (!policyGradient) {
            neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.ELU), "width = " +  + (stateValue ? 1 : outputSize));
            neuralNetwork.addOutputLayer(BinaryFunctionType.MEAN_SQUARED_ERROR);
            neuralNetwork.verboseTraining(100);
        }
        else {
            neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), "width = " + outputSize);
            neuralNetwork.addOutputLayer(BinaryFunctionType.DIRECT_GRADIENT);
        }
        neuralNetwork.build();
        neuralNetwork.setOptimizer(OptimizationType.RADAM);
        neuralNetwork.addNormalizer(3, NormalizationType.WEIGHT_NORMALIZATION);
        neuralNetwork.setNeuralNetworkName("TicTacToe");
        return neuralNetwork;
    }

    /**
     * Builds neural network for tic tac toe player (agent).
     *
     * @param outputSize output size of neural network (number of actions and their values).
     * @return built neural network
     * @throws DynamicParamException throws exception if setting of dynamic parameters fails.
     * @throws NeuralNetworkException throws exception if building of neural network fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    private static NeuralNetwork buildNeuralNetwork(int inputSize, int outputSize) throws DynamicParamException, NeuralNetworkException, MatrixException {
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.addInputLayer("width = " + inputSize);
//        neuralNetwork.addHiddenLayer(LayerType.GRU, new ActivationFunction(UnaryFunctionType.RELU), "width = 100");
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.ELU), "width = 100");
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.ELU), "width = 100");
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), "width = " + (outputSize + 1));
        neuralNetwork.addOutputLayer(BinaryFunctionType.POLICY_VALUE);
        neuralNetwork.verboseTraining(10);
        neuralNetwork.build();
        neuralNetwork.setOptimizer(OptimizationType.ADAM);
        neuralNetwork.addNormalizer(3, NormalizationType.WEIGHT_NORMALIZATION);
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
        if (e.getSource() == noughtRadioButton) humanPlayer = Player.NOUGHT;
        if (e.getSource() == autoRadioButton) humanPlayer = null;
        if (e.getSource() == crossRadioButton) humanPlayer = Player.CROSS;
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
     * Handles mouse click action taking place in tic tac toe panel.
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
    private void playGames() throws MatrixException, NeuralNetworkException, DynamicParamException, AgentException {
        initWindow();
        jFrame.revalidate();

        int drawCountTemp = 0;
        int playerNoughtWonCountTemp = 0;
        int playerCrossWonCountTemp = 0;
        int numberOfGames = 500000000;

        for (int game = 0; game < numberOfGames; game++) {
            playGame();
            if (gameStatus == GameStatus.NOUGHT_WON) {
                playerNoughtWonCountTemp++;
                playerNoughtWonCount++;
            }
            if (gameStatus == GameStatus.CROSS_WON) {
                playerCrossWonCountTemp++;
                playerCrossWonCount++;
            }
            if (gameStatus == GameStatus.DRAW) {
                drawCountTemp++;
                drawCount++;
            }
            if (game % 50 == 0 && game > 0) {
                System.out.println("Game #" + game + " Nought won: " + playerNoughtWonCountTemp + " Cross won: " + playerCrossWonCountTemp + " Draw: " + drawCountTemp);
                drawCountTemp = 0;
                playerNoughtWonCountTemp = 0;
                playerCrossWonCountTemp = 0;
            }
        }
        System.out.println("Nought won games: " + playerNoughtWonCount);
        System.out.println("Cross won games: " + playerCrossWonCount);
        System.out.println("Draw games: " + drawCount);
        nought.stop();
        cross.stop();
    }

    /**
     * Plays single episode of game.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private void playGame() throws MatrixException, NeuralNetworkException, DynamicParamException, AgentException {
        currentPlayer = random.nextInt(2) == 0 ? Player.NOUGHT : Player.CROSS;
        currentHumanPlayer = humanPlayer;
        gameBoard = new GameBoard();
        gameStatus = GameStatus.ONGOING;

        panelLock.lock();
        ticTacToePanel.setHumanPlayer(currentHumanPlayer);
        ticTacToePanel.setGameBoard(gameBoard.getGameBoard(), gameStatus);
        panelLock.unlock();

        if (currentHumanPlayer == null) {
            nought.enableLearning();
            cross.enableLearning();
        }
        else {
            nought.disableLearning();
            cross.disableLearning();
        }

        nought.newEpisode();
        cross.newEpisode();
        episodeID++;
        int timeStep = 0;
        while (gameStatus == GameStatus.ONGOING) {
            gameBoard.updateState();
            environmentState = new EnvironmentState(episodeID, ++timeStep, gameBoard.getState(), gameBoard.getAvailableMoves());
            panelLock.lock();
            ticTacToePanel.setGameBoard(gameBoard.getGameBoard(), gameStatus);
            panelLock.unlock();
            jFrame.revalidate();
            ticTacToePanel.paintImmediately(0, 0, boardSize * tileSize, boardSize * tileSize + 60);
            getAgent().newStep();
            if (currentHumanPlayer == currentPlayer) {
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
                getAgent().act(gameBoard.getPos(humanRow, humanCol));
            }
            else getAgent().act(currentHumanPlayer != null);
            currentPlayer = currentPlayer == Player.CROSS ? Player.NOUGHT : Player.CROSS;
        }
        nought.endEpisode();
        cross.endEpisode();

        panelLock.lock();
        ticTacToePanel.setGameBoard(gameBoard.getGameBoard(), gameStatus);
        panelLock.unlock();

        jFrame.revalidate();
        ticTacToePanel.paintImmediately(0, 0, boardSize * tileSize, boardSize * tileSize + 60);

        if (currentHumanPlayer != null) {
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
        gameBoard.makeMove(currentPlayer, action);
        gameBoard.updateState();
        gameStatus = gameBoard.updateGameStatus(currentPlayer);
        setReward(agent);
    }

    /**
     * Sets immediate reward for agent.
     *
     * @param agent current agent.
     */
    private void setReward(Agent agent) {
        if (gameStatus == GameStatus.ONGOING) agent.respond(rewardStructure.MOVE);
        else {
            if (currentHumanPlayer != Player.NOUGHT) nought.respond(gameStatus == GameStatus.DRAW ? rewardStructure.DRAW : gameStatus == GameStatus.NOUGHT_WON ? rewardStructure.WIN : rewardStructure.LOST);
            if (currentHumanPlayer != Player.CROSS) cross.respond(gameStatus == GameStatus.DRAW ? rewardStructure.DRAW : gameStatus == GameStatus.CROSS_WON ? rewardStructure.WIN : rewardStructure.LOST);
        }
    }

    /**
     * Returns active agent (player)
     *
     * @return active agent
     */
    private Agent getAgent() {
        return (currentPlayer == Player.NOUGHT) ? nought : cross;
    }

    /**
     * Returns opponent agent (player)
     *
     * @return opponent agent.
     */
    private Agent getOpponentAgent() {
        return (currentPlayer == Player.NOUGHT) ? cross : nought;
    }

}
