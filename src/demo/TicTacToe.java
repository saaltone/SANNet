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
import core.optimization.OptimizationType;
import core.reinforcement.Agent;
import core.reinforcement.AgentException;
import core.reinforcement.DeepAgent;
import core.reinforcement.Environment;
import utils.*;

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

        final GameSlot[][] gameBoard;

        /**
         * Current state of game (game board).
         *
         */
        private final Matrix state;

        /**
         * Size of game board (size x size)
         *
         */
        private final int size;

        /**
         * Initializes game board.
         *
         * @param size size as size x size.
         */
        GameBoard (int size) {
            this.size = size;

            state = new DMatrix(2 * size * size, 1);

            gameBoard = new GameSlot[size][size];
            for (int row = 0; row < size; row++) {
                for (int col = 0; col < size; col++) {
                    gameBoard[row][col] = GameSlot.EMPTY;
                }
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
         * Return position as row and col for given action.
         *
         * @param action given action.
         * @return position as row and col
         */
        private int[] getPos(int action) {
            int[] pos = new int[2];
            pos[0] = action % size;
            pos[1] = action / size;
            return pos;
        }

        /**
         * Checks if taken move (action) is valid.
         *
         * @param action action taken.
         * @return true if move is valid otherwise false.
         */
        public boolean isValidMove(int action) {
            int[] pos = getPos(action);
            return isValidMove(pos[0], pos[1]);
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
            return getAvailableMoves().isEmpty();
        }

        /**
         * Makes move per action taken.
         *
         * @param row row of game board.
         * @param col column of game board.
         */
        public void makeMove(Player player, int row, int col) {
            gameBoard[row][col] = player == Player.NOUGHT ? GameSlot.NOUGHT : GameSlot.CROSS;
            state.setValue((player == Player.NOUGHT ? 0 : size * size) + getPos(row, col), 0, 1);
        }

        /**
         * Makes move per action taken.
         *
         * @param action action taken.
         */
        public void makeMove(Player player, int action) {
            int[] pos = getPos(action);
            gameBoard[pos[0]][pos[1]] = player == Player.NOUGHT ? GameSlot.NOUGHT : GameSlot.CROSS;
            state.setValue((player == Player.NOUGHT ? 0 : size * size) + action, 0, 1);
        }

        /**
         * Checks if one of players has won the game.
         *
         * @return return player id if player has won otherwise returns ongoing game state.
         */
        private Player checkWinner(Player player) {
            int diagStat = 0;
            int adiagStat = 0;
            GameSlot targetSlot = player == Player.NOUGHT ? GameSlot.NOUGHT : GameSlot.CROSS;
            for (int row = 0; row < size; row++) {
                int rowStat = 0;
                int colStat = 0;
                for (int col = 0; col < size; col++) {
                    if (gameBoard[row][col] == targetSlot) rowStat++; else rowStat = 0;
                    if (gameBoard[col][row] == targetSlot) colStat++; else colStat = 0;
                }
                if (rowStat == size || colStat == size) return player;

                if (gameBoard[row][row] == targetSlot) diagStat++; else diagStat = 0;
                if (gameBoard[row][size - 1 - row] == targetSlot) adiagStat++; else adiagStat = 0;
            }

            if (diagStat == size || adiagStat == size) return player;

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
         * Array structure to draw maze.
         *
         */
        private GameSlot[][] gameBoard;

        /**
         * Game status for drawing board.
         *
         */
        private GameStatus gameStatus;

        /**
         * If true game is in auto mode.
         *
         */
        private boolean autoMode;

        /**
         * Lock that is used to synchronize GUI and tic tac toe threads with each other.
         *
         */
        private final Lock guiLock = new ReentrantLock();

        /**
         * Sets gameBoard to be drawn.
         *
         * @param gameBoard gameBoard to be drawn.
         */
        public void setGameBoard(GameSlot[][] gameBoard, GameStatus gameStatus, boolean autoMode) {
            guiLock.lock();
            this.gameBoard = gameBoard;
            this.gameStatus = gameStatus;
            this.autoMode = autoMode;
            guiLock.unlock();
        }

        /**
         * Paints gameBoard to JPanel.
         *
         * @param g graphics.
         */
        public void paintComponent(Graphics g) {
            super.paintComponent(g);
            guiLock.lock();
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
            if (gameStatus != null && !autoMode) {
                if (gameStatus == GameStatus.NOUGHT_WON || gameStatus == GameStatus.DRAW || gameStatus == GameStatus.CROSS_WON) {
                    g.setColor(Color.CYAN);
                    g.setFont(new Font("ARIAL", Font.PLAIN, 30));
                    if (gameStatus == GameStatus.NOUGHT_WON) g.drawString("O WON", 97, 160);
                    if (gameStatus == GameStatus.DRAW) g.drawString("DRAW", 103, 160);
                    if (gameStatus == GameStatus.CROSS_WON) g.drawString("X WON", 97, 160);
                }
            }
            guiLock.unlock();
        }

    }

    /**
     * Reward structure i.e. rewards returned to agent during game per action taken.
     *
     */
    public static class RewardStructure {
        final double WIN = 20; // 5
        final double DRAW = 10; // 3
        final double LOST = 0; // 0
        final double MOVE = 0; // 0
        final double ILLEGAL_MOVE = 0;
    }

    /**
     * Reward structure.
     *
     */
    private final RewardStructure rewardStructure = new RewardStructure();

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
    private final int boardSize = 3;

    /**
     * Game board.
     *
     */
    private GameBoard gameBoard;

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
     * Row into where human player want to make a move into.
     *
     */
    private int humanRow = -1;

    /**
     * Column into where human player want to make a move into.
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
     * Number of games to be played.
     *
     */
    private final int numberOfGames = 500000000;

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
     * Condition for lock.
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
     * @throws DynamicParamException throws exception if setting of dynamic parameter fails.
     * @throws IOException throws exception if coping of neural network instance fails.
     * @throws ClassNotFoundException throws exception if coping of neural network instance fails.
     */
    public TicTacToe() throws NeuralNetworkException, DynamicParamException, IOException, ClassNotFoundException {
        nought = createAgent(Player.NOUGHT);
        cross = createAgent(Player.CROSS);
    }

    /**
     * Initializes window for maze.
     *
     */
    public void initWindow() {
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
    public void mousePressed (MouseEvent e) {}

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
        lock.lock();
        humanRow = e.getY() / tileSize;
        humanCol = e.getX() / tileSize;
        humanAction.signal();
        lock.unlock();
    }

    /**
     * Plays given number of games.
     *
     * @throws AgentException throws exception if agent operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws IOException throws exception if cloning of neural network fails.
     * @throws ClassNotFoundException throws exception if cloning of neural network fails.
     */
    private void playGames() throws AgentException, MatrixException, NeuralNetworkException, IOException, ClassNotFoundException {
        initWindow();
        jFrame.revalidate();
        int drawCountTemp = 0;
        int playerNoughtWonCountTemp = 0;
        int playerCrossWonCountTemp = 0;
        for (int game = 0; game < numberOfGames; game++) {
            playGame();
            getAgent().setEpsilon((double)illegalMoves / 50 * 0.9 + 0.1);
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
     * Plays single episode of game.
     *
     * @throws AgentException throws exception if agent operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws IOException throws exception if cloning of neural network fails.
     * @throws ClassNotFoundException throws exception if cloning of neural network fails.
     */
    private void playGame() throws AgentException, MatrixException, NeuralNetworkException, IOException, ClassNotFoundException {
        currentPlayer = random.nextInt(2) == 0 ? Player.NOUGHT : Player.CROSS;
        Player currentHumanPlayer = humanPlayer;
        gameBoard = new GameBoard(boardSize);
        gameStatus = GameStatus.ONGOING;
        ArrayList<Matrix> gameStates = new ArrayList<>();
        gameStates.add(gameBoard.getState().copy());
        boolean isTraining = true;
        if (currentHumanPlayer != null) getAgent().setEpsilon(-1);
        do {
            ticTacToePanel.setGameBoard(gameBoard.getGameBoard(), null, currentHumanPlayer == null);
            jFrame.revalidate();
            ticTacToePanel.paintImmediately(0, 0, boardSize * tileSize, boardSize * tileSize + 60);
            if (currentHumanPlayer == currentPlayer) {
                lock.lock();
                humanRow = -1;
                humanCol = -1;
                try {
                    while (!gameBoard.isValidMove(humanRow, humanCol)) humanAction.await();
                }
                catch (InterruptedException exception) {}
                gameBoard.makeMove(currentPlayer, humanRow, humanCol);
                gameStatus = gameBoard.updateGameStatus(currentPlayer);
                lock.unlock();
            }
            else {
                getAgent().newStep(isTraining);
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
            }

            currentPlayer = currentPlayer == Player.CROSS ? Player.NOUGHT : Player.CROSS;

        } while (gameStatus == GameStatus.ONGOING);

        nought.commitStep(true);
        cross.commitStep(true);

        ticTacToePanel.setGameBoard(gameBoard.getGameBoard(), gameStatus, currentHumanPlayer == null);
        jFrame.revalidate();
        ticTacToePanel.paintImmediately(0, 0, boardSize * tileSize, boardSize * tileSize + 60);

        if (currentHumanPlayer != null) {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {}
        }

        if (allMoves == 450 && illegalMoves == 0 && gameStatus == GameStatus.DRAW) printGame(gameStates, gameStatus);

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
        gameBoard.makeMove(currentPlayer, action);
        gameStatus = gameBoard.updateGameStatus(currentPlayer);
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
                    GameStatus targetPlayer = currentPlayer == Player.NOUGHT ? GameStatus.NOUGHT_WON : GameStatus.CROSS_WON;
                    if (gameStatus == targetPlayer) return rewardStructure.WIN;
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
        return (currentPlayer == Player.NOUGHT) ? nought : cross;
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
        System.out.println();
        for (int row = 0; row < boardSize; row++) {
            for (int col = 0; col < boardSize; col++) {
                if (col == 0) System.out.print("|");
                boolean nought = gameState.getValue(row + col * boardSize, 0) == 1;
                boolean cross = gameState.getValue(boardSize * boardSize + row + col * boardSize, 0) == 1;
                if (nought) System.out.print("O");
                if (!nought && !cross) System.out.print(" ");
                if (cross) System.out.print("X");
                System.out.print("|");
            }
            System.out.println();
        }
        for (int row = 0; row < 2 * boardSize + 1; row++) System.out.print("-");
        System.out.println();
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
    private DeepAgent createAgent(Player player) throws NeuralNetworkException, DynamicParamException, IOException, ClassNotFoundException {
        NeuralNetwork QNN = buildNeuralNetwork(player == Player.NOUGHT ? "Nought" : "Cross", 2 * boardSize * boardSize, boardSize * boardSize);
        DeepAgent agent = new DeepAgent(this, QNN, "trainCycle = " + (10 * 9) + ", updateTNNCycle = " + (10 * 9) + ", epsilonDecayByEpisode = false, epsilonDecayRate = 1, epsilonInitial = 1, epsilonMin = 0, learningRate = 0.4, gamma = 0.85, alpha = 1");
        agent.start();
        return agent;
    }

    /**
     * Builds neural network for tic tac toe player (agent).
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
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(ActivationFunctionType.RELU), "width = 27");
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(ActivationFunctionType.RELU), "width = 27");
        neuralNetwork.addOutputLayer(LayerType.FEEDFORWARD, new ActivationFunction(ActivationFunctionType.RELU), "width = " + outputSize);
        neuralNetwork.build();
        neuralNetwork.setOptimizer(OptimizationType.ADAM);
        neuralNetwork.setLossFunction(LossFunctionType.HUBER);
        neuralNetwork.setNeuralNetworkName(neuralNetworkName);
        neuralNetwork.setTrainingSampling(100, false, true);
        neuralNetwork.setTrainingIterations(50);
        neuralNetwork.verboseTraining(100);
        return neuralNetwork;
    }

}
