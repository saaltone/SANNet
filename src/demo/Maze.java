/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package demo;

import core.NeuralNetwork;
import core.NeuralNetworkException;
import core.activation.ActivationFunction;
import core.layer.LayerType;
import core.normalization.NormalizationType;
import core.optimization.OptimizationType;
import core.reinforcement.Agent;
import core.reinforcement.DeepAgent;
import core.reinforcement.Environment;
import utils.*;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.geom.Ellipse2D;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Class that defines maze in where agent takes journeys.<br>
 *
 * Reference: https://rosettacode.org/wiki/Maze_generation<br>
 *
 */
public class Maze implements Environment, ActionListener {

    /**
     * Class that defines neighbor i.e. connection between two cells.
     *
     */
    private class Neighbor {

        /**
         * First cell to be connected.
         *
         */
        private final Cell cell1;

        /**
         * Second cell to be connected.
         *
         */
        private final Cell cell2;

        /**
         * If true cells are connected.
         *
         */
        boolean connected;

        /**
         * Constructor for neighbor.
         *
         * @param cell1 first cell to be connected.
         * @param cell2 second cell to be connected.
         */
        Neighbor(Cell cell1, Cell cell2) {
            this.cell1 = cell1;
            this.cell2 = cell2;
            connected = false;
        }

        /**
         * Returns neighboring cell for given cell reference.
         *
         * @param cell cell reference.
         * @return neighbor of given cell reference.
         */
        public Cell getTo(Cell cell) {
            if (cell == cell1) return cell2;
            if (cell == cell2) return cell1;
            return null;
        }

        /**
         * Sets cells as connected.
         *
         */
        public void setConnected() {
            connected = true;
        }

        /**
         * Checks if cells can be connected i.e. other cell is not visited or connected yet.
         *
         * @param from cell from where connection is created.
         * @return if true cells can be connected.
         */
        public boolean canConnectTo(Cell from) {
            return !connected && !getTo(from).visited;
        }

        /**
         * Creates connection to neighbor from given cell.
         *
         * @param from cell from where connection is created.
         * @return return reference to connected cell and if connection cannot be created returns null.
         */
        public Cell passTo(Cell from) {
            if (canConnectTo(from)) {
                setConnected();
                getTo(from).setVisited();
                return getTo(from);
            }
            else return null;
        }

    }

    /**
     * Class that defines cell of maze.
     *
     */
    private class Cell {

        /**
         * X position of cell.
         *
         */
        final int x;

        /**
         * Y position of cell.
         *
         */
        final int y;

        /**
         * If true cell has been visited. Used in maze construction phase.
         *
         */
        boolean visited;

        /**
         * Counts how many times has been visited by agent.
         *
         */
        int visitCount;

        /**
         * Previous time step of cell.
         *
         */
        long previousCellTimeStep = 0;

        /**
         * Current time step of cell.
         *
         */
        long cellTimeStep = 0;

        /**
         * Neighbors of cell.
         *
         */
        final Neighbor[] neighbors;

        /**
         * Constructor for cell.
         *
         * @param x x position of cell.
         * @param y y position of cell.
         */
        public Cell(int x, int y) {
            this.x = x;
            this.y = y;
            visited = false;
            visitCount = 0;
            neighbors = new Neighbor[4];
        }

        /**
         * Connects cell to neighboring cell.
         *
         * @param to cell that this cell is connected to.
         * @param direction direction (up, down, left, right) where neighboring cell is located.
         */
        public void connectTo(Cell to, int direction) {
            Neighbor neighbor = new Neighbor(this, to);
            neighbors[direction] = neighbor;
            to.neighbors[getOppositeDirection(direction)] = neighbor;
        }

        /**
         * Sets cell as visited.
         *
         */
        public void setVisited() {
            visited = true;
        }

        /**
         * Returns list of unvisited neighbors.
         *
         * @return list of unvisited neighbors.
         */
        public ArrayList<Neighbor> getUnvisitedNeighbors() {
            ArrayList<Neighbor> unvisitedNeighbors = new ArrayList<>();
            for (int index = 0; index < 4; index++) {
                Neighbor neighbor = neighbors[index];
                if (neighbor != null) if (neighbor.canConnectTo(this)) unvisitedNeighbors.add(neighbor);
            }
            Collections.shuffle(unvisitedNeighbors);
            return unvisitedNeighbors;
        }

        /**
         * Visits all neighbors not visited so far.
         *
         */
        public void visitNeighbors() {
            for (Neighbor neighbor : getUnvisitedNeighbors()) {
                Cell cell = neighbor.passTo(this);
                if (cell != null) cell.visitNeighbors();
            }
        }

        /**
         * Return opposite direction i.e if direction is left then right is returned.
         *
         * @param direction direction to be taken as opposite.
         * @return opposite direction.
         */
        public int getOppositeDirection(int direction) {
            switch (direction) {
                case 0: return 1;
                case 1: return 0;
                case 2: return 3;
                case 3: return 2;
                default: return -1;
            }
        }

        /**
         * Checks if cell can be reached.
         *
         * @return if true cell can be reached.
         */
        public boolean isOpen() {
            for (int index = 0; index < 4; index++) {
                Neighbor neighbor = neighbors[index];
                if (neighbor != null) if (neighbor.connected) return true;
            }
            return false;
        }

        /**
         * Checks if cell is connected with given neighbor (direction).
         *
         * @param neighbor given neighbor (direction).
         * @return if true cell is connected to given direction.
         */
        public boolean isConnected(int neighbor) {
            return neighbors[neighbor] != null && neighbors[neighbor].connected;
        }

        /**
         * Checks if cell is dead end i.e. is has only one open direction.
         *
         * @return if true cell is dead end.
         */
        public boolean isDeadend() {
            int connectCount = 0;
            for (int index = 0; index < 4; index++) {
                if (isConnected(index)) connectCount++;
            }
            return connectCount == 1;
        }

        /**
         * Increments cell visit count.
         *
         */
        public void incrementCount() {
            visitCount++;
            if (cellTimeStep - previousCellTimeStep > 100) visitCount = 1;
        }

        /**
         * Gets cell visit count.
         *
         * @return cell visit count.
         */
        public int getVisitCount() {
            return visitCount;
        }

        /**
         * Updates cell time step and stores previous cell time step.
         *
         * @param timeStep current time step.
         */
        public void updateCellTimeStep(long timeStep) {
            previousCellTimeStep = cellTimeStep;
            cellTimeStep = timeStep;
        }

    }

    /**
     * Class that defines maze agent.
     *
     */
    static class MazeAgent {

        /**
         * X position of maze agent.
         *
         */
        final int x;

        /**
         * Y position of maze agent.
         *
         */
        final int y;

        /**
         * Action taken by maze agent.
         *
         */
        final int action;

        /**
         * Constructor for maze agent.
         *
         * @param x x position of maze agent.
         * @param y y position of maze agent.
         * @param action action taken by maze agent.
         */
        MazeAgent(int x, int y, int action) {
            this.x = x;
            this.y = y;
            this.action = action;
        }
    }

    /**
     * Implements JPanel into which maze is drawn.<br>
     *
     */
    private class MazePanel extends JPanel {

        /**
         * Array structure to draw maze.
         *
         */
        private Cell[][] maze;

        /**
         * Agent's maze history as current and past positions.
         *
         */
        private final ConcurrentLinkedQueue<MazeAgent> mazeAgentHistory = new ConcurrentLinkedQueue<>();

        /**
         * Sets maze to be drawn.
         *
         * @param maze maze to be drawn.
         */
        public void setMaze(Cell[][] maze) {
            this.maze = maze;
        }

        /**
         * Updates agent's history to be drawn into maze.
         *
         * @param mazeAgentHistory
         */
        public void updateAgentHistory(LinkedList<MazeAgent> mazeAgentHistory) {
            if (mazeAgentHistory == null) return;
            this.mazeAgentHistory.clear();
            this.mazeAgentHistory.addAll(mazeAgentHistory);
        }

        /**
         * Paints journey to JPanel.
         *
         * @param g graphics.
         */
        public void paintComponent(Graphics g) {
            super.paintComponent(g);
            if (maze == null) return;
            g.setColor(Color.BLACK);
            for (int x = 0; x < maze.length; x++) {
                for (int y = 0; y < maze[x].length; y++) {
                    if (maze[x][y].neighbors[0] != null) if (!maze[x][y].neighbors[0].connected) g.drawLine((x + 0) * 10, (y + 0) * 10, (x + 0) * 10, (y + 1) * 10);
                    if (maze[x][y].neighbors[1] != null) if (!maze[x][y].neighbors[1].connected) g.drawLine((x + 1) * 10, (y + 0) * 10, (x + 1) * 10, (y + 1) * 10);
                    if (maze[x][y].neighbors[2] != null) if (!maze[x][y].neighbors[2].connected) g.drawLine((x + 0) * 10, (y + 0) * 10, (x + 1) * 10, (y + 0) * 10);
                    if (maze[x][y].neighbors[3] != null) if (!maze[x][y].neighbors[3].connected) g.drawLine((x + 0) * 10, (y + 1) * 10, (x + 1) * 10, (y + 1) * 10);
                }
            }
            if (mazeAgentHistory != null) {
                Iterator iterator = mazeAgentHistory.iterator();
                Graphics2D g2d = (Graphics2D)g;
                while (iterator.hasNext()) {
                    MazeAgent mazeAgent = (MazeAgent)iterator.next();
                    if (!iterator.hasNext()) g.setColor(Color.RED);
                    else g.setColor(Color.BLUE);
                    Ellipse2D.Double circle = new Ellipse2D.Double(mazeAgent.x * 10 + 1, mazeAgent.y * 10 + 1, 7, 7);
                    g2d.fill(circle);
                }
            }
        }

    }

    /**
     * Size of maze.
     *
     */
    private int size = 60;

    /**
     * Current time step.
     *
     */
    private long timeStep = 0;

    /**
     * Array structure that contains maze.
     *
     */
    private Cell[][] maze;

    /**
     * Random function.
     *
     */
    private final Random random = new Random();

    /**
     * Reference to maze JPanel class.
     *
     */
    private final MazePanel mazePanel = new MazePanel();

    /**
     * JFrame for maze.
     *
     */
    private JFrame jFrame;

    /**
     * Current position of agent in maze.
     *
     */
    private MazeAgent mazeAgentCurrent;

    /**
     * Linked list to store agent's history.
     *
     */
    private final LinkedList<MazeAgent> mazeAgentHistory = new LinkedList<>();

    /**
     * Size of agent's history. Remembers given number of previous moves and uses then as input for agent state.
     *
     */
    private final int agentHistorySize = 12;

    /**
     * Reference to deep agent.
     *
     */
    private DeepAgent agent;

    /**
     * State of environment (agent).
     *
     */
    private Matrix state;

    /**
     * If true requests agent to be reset to a starting position.
     *
     */
    private boolean resetRequested = false;

    /**
     * If true requests to rebuild maze.
     *
     */
    private boolean rebuildRequested = false;

    /**
     * Reset button for maze. This action reset agent to the middle of maze.
     *
     */
    private JButton jResetButton;

    /**
     * Rebuild button for maze. This action regenerates maze and set agent to middle of it.
     *
     */
    private JButton jRebuildButton;

    /**
     * Constructor for maze.
     *
     */
    public Maze() {
        newMaze();
    }

    /**
     * Constructor for maze with specific size.
     *
     * @param size size of maze.
     */
    public Maze(int size) {
        this.size = size;
        newMaze();
    }

    /**
     * Initializes window for maze.
     *
     */
    public void initWindow() {
        JFrame.setDefaultLookAndFeelDecorated(true);
        jFrame = new JFrame("Maze");
        jFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        jFrame.setBackground(Color.white);
        jFrame.setSize(size * 10, size * 10 + 60);
        jResetButton = new JButton("Reset");
        jResetButton.setSize(100, 40);
        jResetButton.setLocation(0, 0);
        jResetButton.addActionListener(this);
        jFrame.add(jResetButton);
        jRebuildButton = new JButton("Rebuild");
        jRebuildButton.setSize(100, 40);
        jRebuildButton.setLocation(size * 10 - 110, 0);
        jRebuildButton.addActionListener(this);
        jFrame.add(jRebuildButton);
        jFrame.add(mazePanel);
        jFrame.setLayout(null);
        mazePanel.setSize(size * 10, size * 10);
        mazePanel.setLocation(0, 40);
        jFrame.setVisible(true);
    }

    /**
     * Creates new maze.
     *
     */
    public void newMaze() {
        maze = new Cell[size][size];
        for (int x = 0; x < maze.length; x++) {
            for (int y = 0; y < maze[x].length; y++) {
                maze[x][y] = new Cell(x, y);
            }
        }
        for (int x = 0; x < maze.length - 1; x++) {
            for (int y = 0; y < maze[x].length - 1; y++) {
                maze[x][y].connectTo(maze[x + 1][y], 1);
                maze[x][y].connectTo(maze[x][y + 1], 3);
            }
        }
    }

    /**
     * Creates maze and paints it into window.
     *
     */
    public void createMaze() {
        Cell cell = maze[0][0];
        cell.visitNeighbors();
        mazePanel.setMaze(maze);
        jFrame.revalidate();
        mazePanel.paintImmediately(0, 0, size * 10, size * 10 + 40);
    }

    /**
     * Initialized agent for the maze and maze itself.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if setting of dynamic parameter fails.
     * @throws IOException throws exception if coping of neural network instance fails.
     * @throws ClassNotFoundException throws exception if coping of neural network instance fails.
     */
    public void initializeMazeAgent() throws NeuralNetworkException, MatrixException, DynamicParamException, IOException, ClassNotFoundException {
        agent = createAgent(4 * agentHistorySize + 4, 4);
        initMaze();
    }

    /**
     * Initializes maze with given size. Positions agent to the middle of maze.
     *
     */
    public void initMaze() {
        boolean initialized = false;
        for (Cell[] cells : maze) {
            for (Cell cell : cells) {
                cell.visitCount = 0;
            }
        }
        int x = size / 2;
        int y = size / 2;
        mazeAgentHistory.clear();
        while (!initialized) {
            if (maze[x][y].isOpen()) {
                MazeAgent mazeAgent = new MazeAgent(x, y, -1);
                mazeAgentCurrent = mazeAgent;
                maze[x][y].updateCellTimeStep(++timeStep);
                maze[x][y].incrementCount();
                mazeAgentHistory.addLast(mazeAgent);
                initialized = true;
            }
            x = random.nextInt(size);
            y = random.nextInt(size);
        }
        updateState();
    }

    /**
     * Listener for button actions.
     *
     * @param e event originated from buttons.
     */
    public void actionPerformed(ActionEvent e) {
        if (e.getSource() == jResetButton) resetRequested = true;
        if (e.getSource() == jRebuildButton) rebuildRequested = true;
    }

    /**
     * Plays game until user quits game (closes window).
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws IOException throws exception if coping of neural network instance fails.
     * @throws ClassNotFoundException throws exception if coping of neural network instance fails.
     */
    public void playAgent() throws MatrixException, NeuralNetworkException, IOException, ClassNotFoundException {
        while (true) {
            agent.newStep();
            if (!agent.act(false, false)) agent.act(false, true);
            agent.updateValue();
            agent.commitStep();

            mazePanel.updateAgentHistory(mazeAgentHistory);
            jFrame.revalidate();
            mazePanel.paintImmediately(0, 0, size * 10, size * 10 + 40);

            if (resetRequested) {
                initMaze();
                resetRequested = false;
            }

            if (rebuildRequested) {
                newMaze();
                createMaze();
                initMaze();
                rebuildRequested = false;
            }

            if (mazeAgentCurrent.x == 0 || mazeAgentCurrent.y == 0 || mazeAgentCurrent.x == size - 1 || mazeAgentCurrent.y == size - 1) initMaze();
        }
    }

    /**
     * Returns current state of environment for the agent.
     *
     * @return state of environment
     */
    public Matrix getState() {
        return state.copy();
    }

    /**
     * Updates state based on state of environment.
     *
     */
    private void updateState() {
        state = new DMatrix(4 * agentHistorySize + 4, 1);
        Iterator iterator = mazeAgentHistory.iterator();
        int index = 0;
        while (iterator.hasNext()) {
            MazeAgent mazeAgent = (MazeAgent)iterator.next();
            if (mazeAgent.action != -1) {
                state.setValue(index + mazeAgent.action, 0, -1);
            }
            index += 4;
        }
        Cell mazeCell = maze[mazeAgentCurrent.x][mazeAgentCurrent.y];
        state.setValue(4 * agentHistorySize, 0, mazeCell.isConnected(0) ? -1 : 0);
        state.setValue(4 * agentHistorySize + 1, 0, mazeCell.isConnected(1) ? -1 : 0);
        state.setValue(4 * agentHistorySize + 2, 0, mazeCell.isConnected(2) ? -1 : 0);
        state.setValue(4 * agentHistorySize + 3, 0, mazeCell.isConnected(3) ? -1 : 0);
    }

    /**
     * True if state is terminal. This is usually true if episode is completed.
     *
     * @return always false.
     */
    public boolean isTerminalState() {
        return false;
    }

    /**
     * Returns available actions in current state of environment
     *
     * @return available actions in current state of environment.
     */
    public ArrayList<Integer> getAvailableActions() {
        ArrayList<Integer> availableActions = new ArrayList<>();
        Cell cell = maze[mazeAgentCurrent.x][mazeAgentCurrent.y];
        for (int index = 0; index < cell.neighbors.length; index++) {
            Neighbor neighbor = cell.neighbors[index];
            if (neighbor != null) if (neighbor.connected) availableActions.add(index);
        }
        return availableActions;
    }

    /**
     * Checks if action is valid.
     *
     * @param agent  agent that is taking action.
     * @param action action to be taken.
     * @return true if action can be taken successfully.
     */
    public boolean isValidAction(Agent agent, int action) {
        return getAvailableActions().contains(action);
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
     * @param agent  agent that is taking action.
     * @param action action to be taken.
     */
    public void commitAction(Agent agent, int action) {
        int x = mazeAgentCurrent.x;
        int y = mazeAgentCurrent.y;
        switch (action) {
            case 0: x--; break;
            case 1: x++; break;
            case 2: y--; break;
            case 3: y++; break;
        }
        mazeAgentCurrent = new MazeAgent(x, y, action);
        if (mazeAgentHistory.size() == agentHistorySize) mazeAgentHistory.pollFirst();
        mazeAgentHistory.addLast(mazeAgentCurrent);
        maze[x][y].updateCellTimeStep(++timeStep);
        maze[x][y].incrementCount();
        updateState();
    }

    /**
     * Requests immediate reward from environment after taking action.
     *
     * @param agent       agent that is asking for reward.
     * @param validAction true if taken action was available one.
     * @return immediate reward.
     */
    public double requestReward(Agent agent, boolean validAction) {
        if (!validAction) return 0;
        else {
            if (maze[mazeAgentCurrent.x][mazeAgentCurrent.y].isDeadend()) return 0;
            else return 0.1 + 3 / Math.pow(maze[mazeAgentCurrent.x][mazeAgentCurrent.y].getVisitCount(), 3);
        }
    }

    /**
     * Main function for maze. Initializes maze, deep agent travelling in maze and starts game
     *
     * @param args not used.
     */
    public static void main(String[] args) {
        Maze maze;
        try {
            maze = new Maze();
            maze.initWindow();
            maze.createMaze();
            maze.initializeMazeAgent();
            maze.playAgent();
        }
        catch (Exception exception) {
            exception.printStackTrace();
            System.exit(-1);
        }
    }

    /**
     * Creates agent (player) for maze.
     *
     * @return agent
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException throws exception if setting of dynamic parameter fails.
     * @throws IOException throws exception if coping of neural network instance fails.
     * @throws ClassNotFoundException throws exception if coping of neural network instance fails.
     */
    private DeepAgent createAgent(int inputAmount, int outputAmount) throws NeuralNetworkException, DynamicParamException, IOException, ClassNotFoundException {
        NeuralNetwork QNN = buildNeuralNetwork(inputAmount, outputAmount);
        DeepAgent agent = new DeepAgent(this, QNN, "trainCycle = 10, updateTNNCycle = 10, alpha = 0.9, gamma = 0.85, replayBufferSize = 20000, epsilonDecayByEpisode = false, epsilonDecayRate = 0.999, epsilonMin = 0.0");
        agent.start();
        return agent;
    }

    /**
     * Build neural network for maze (agent).
     *
     * @param inputSize input size of neural network (number of states)
     * @param outputSize output size of neural network (number of actions and their values).
     * @return built neural network
     * @throws DynamicParamException throws exception if setting of dynamic parameters fails.
     * @throws NeuralNetworkException throws exception if building of neural network fails.
     */
    private static NeuralNetwork buildNeuralNetwork(int inputSize, int outputSize) throws DynamicParamException, NeuralNetworkException {
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.addInputLayer("width = " + inputSize);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UniFunctionType.ELU), "width = " + 30);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UniFunctionType.RELU), "width = " + 30);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UniFunctionType.GELU), "width = " + 30);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UniFunctionType.ELU), "width = " + 30);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UniFunctionType.RELU), "width = " + 30);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UniFunctionType.GELU), "width = " + 30);
        neuralNetwork.addOutputLayer(LayerType.FEEDFORWARD, new ActivationFunction(UniFunctionType.SOFTMAX), "width = " + outputSize);
        neuralNetwork.build();
        neuralNetwork.setOptimizer(OptimizationType.ADAM);
        neuralNetwork.addNormalizer(6, NormalizationType.WEIGHT_NORMALIZATION);
        neuralNetwork.setLossFunction(BiFunctionType.HUBER);
        neuralNetwork.setTrainingSampling(100, false, true);
        neuralNetwork.setTrainingIterations(10);
        neuralNetwork.verboseTraining(100);
        return neuralNetwork;
    }

}
