/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package demo;

import core.network.NeuralNetwork;
import core.network.NeuralNetworkException;
import core.activation.ActivationFunction;
import core.layer.LayerType;
import core.optimization.OptimizationType;
import core.reinforcement.agent.*;
import core.reinforcement.policy.executablepolicy.*;
import utils.*;
import utils.matrix.*;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.geom.Ellipse2D;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Class that defines maze where agent takes journeys trying to find way out of maze.<br>
 *
 * Reference: https://rosettacode.org/wiki/Maze_generation<br>
 *
 */
public class Maze implements AgentFunctionEstimator, Environment, ActionListener {

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
            return !connected && !Objects.requireNonNull(getTo(from)).visited;
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
                Objects.requireNonNull(getTo(from)).setVisited();
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
         * Returns opposite direction i.e if direction is left then right is returned.
         *
         * @param direction direction to be taken as opposite.
         * @return opposite direction.
         */
        public int getOppositeDirection(int direction) {
            return switch (direction) {
                case 0 -> 1;
                case 1 -> 0;
                case 2 -> 3;
                case 3 -> 2;
                default -> -1;
            };
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
         * Returns cell visit count.
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
     * @param x x position of maze agent.
     * @param y y position of maze agent.
     * @param action action taken by maze agent.
     */
    record MazeAgent(int x, int y, int action) {
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
         * @param mazeAgentHistory maze agent history.
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
            g.setColor(Color.LIGHT_GRAY);
            for (int x = 0; x < maze.length; x++) {
                for (int y = 0; y < maze[x].length; y++) {
                    if (maze[x][y].neighbors[0] != null) if (!maze[x][y].neighbors[0].connected) g.drawLine((x) * 10, (y) * 10, (x) * 10, (y + 1) * 10);
                    if (maze[x][y].neighbors[1] != null) if (!maze[x][y].neighbors[1].connected) g.drawLine((x + 1) * 10, (y) * 10, (x + 1) * 10, (y + 1) * 10);
                    if (maze[x][y].neighbors[2] != null) if (!maze[x][y].neighbors[2].connected) g.drawLine((x) * 10, (y) * 10, (x + 1) * 10, (y) * 10);
                    if (maze[x][y].neighbors[3] != null) if (!maze[x][y].neighbors[3].connected) g.drawLine((x) * 10, (y + 1) * 10, (x + 1) * 10, (y + 1) * 10);
                }
            }
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

    /**
     * Size of maze.
     *
     */
    private int size = 60;

    /**
     * Episode ID
     *
     */
    private int episodeID = 0;

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
     * State size.
     *
     */
    private final int stateSize = 4 * (agentHistorySize + 1);

    /**
     * Reference to deep agent.
     *
     */
    private Agent agent;

    /**
     * State of environment (agent).
     *
     */
    private EnvironmentState environmentState;

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
        mazePanel.setBackground(Color.black);
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
     * Initializes agent for the maze and maze itself.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException throws exception if setting of dynamic parameter fails.
     * @throws IOException throws exception if copying of neural network instance fails.
     * @throws ClassNotFoundException throws exception if copying of neural network instance fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public void initializeMazeAgent() throws NeuralNetworkException, MatrixException, DynamicParamException, IOException, ClassNotFoundException, AgentException {
        agent = createAgent();
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
     * Plays maze game until user quits game (closes window).
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update of estimator fails.
     */
    public void playAgent() throws MatrixException, NeuralNetworkException, DynamicParamException, AgentException {
        while (true) {
            agent.newStep();
            agent.act();

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
     * Returns true if environment is episodic otherwise false.
     *
     * @return true if environment is episodic otherwise false.
     */
    public boolean isEpisodic() {
        return false;
    }

    /**
     * Returns current state of environment for the agent.
     *
     * @return state of environment
     */
    public EnvironmentState getState() {
        return environmentState;
    }

    /**
     * Updates state based on state of environment.
     *
     */
    private void updateState() {
        Matrix state = new DMatrix(stateSize, 1);
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

        HashSet<Integer> availableActions = new HashSet<>();
        Cell cell = maze[mazeAgentCurrent.x][mazeAgentCurrent.y];
        for (int action = 0; action < cell.neighbors.length; action++) {
            Neighbor neighbor = cell.neighbors[action];
            if (neighbor != null) if (neighbor.connected) availableActions.add(action);
        }
        environmentState = new EnvironmentState(++episodeID, 1, state, availableActions);
    }


    /**
     * Takes specific action.
     *
     * @param agent  agent that is taking action.
     * @param action action to be taken.
     */
    public void commitAction(Agent agent, int action) {
        int x = mazeAgentCurrent.x;
        int y = mazeAgentCurrent.y;
        switch (action) {
            case 0 -> x--;
            case 1 -> x++;
            case 2 -> y--;
            case 3 -> y++;
        }
        mazeAgentCurrent = new MazeAgent(x, y, action);
        if (mazeAgentHistory.size() == agentHistorySize) mazeAgentHistory.pollFirst();
        mazeAgentHistory.addLast(mazeAgentCurrent);
        maze[x][y].updateCellTimeStep(++timeStep);
        maze[x][y].incrementCount();
        updateState();
        setReward(agent);
    }

    /**
     * Sets immediate reward from environment after taking action.
     *
     * @param agent agent that is asking for reward.
     */
    private void setReward(Agent agent) {
        if (maze[mazeAgentCurrent.x][mazeAgentCurrent.y].isDeadend()) agent.respond(0);
        else {
            double distanceToStart = 1 - 1 / Math.max(1, Math.sqrt(Math.pow((double)size / 2 - mazeAgentCurrent.x, 2) + Math.pow((double)size / 2 - mazeAgentCurrent.y, 2)));
            double positionPenalty = Math.pow(1 / (double)maze[mazeAgentCurrent.x][mazeAgentCurrent.y].getVisitCount(), 10);
            agent.respond(Math.max(0, distanceToStart * positionPenalty));
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
     * @throws IOException throws exception if copying of neural network instance fails.
     * @throws ClassNotFoundException throws exception if copying of neural network instance fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    private Agent createAgent() throws MatrixException, NeuralNetworkException, DynamicParamException, IOException, ClassNotFoundException, AgentException {
        boolean onlineMemory = false;
        boolean singleFunctionEstimator = false;
        int policyType = 1;
        ExecutablePolicyType executablePolicyType = null;
        String policyTypeParams = "";
        switch (policyType) {
            case 0 -> executablePolicyType = ExecutablePolicyType.GREEDY;
            case 1 -> {
                executablePolicyType = ExecutablePolicyType.EPSILON_GREEDY;
                policyTypeParams = "epsilonInitial = 0.25, epsilonMin = 0.25";
            }
            case 2 -> {
                executablePolicyType = ExecutablePolicyType.NOISY_NEXT_BEST;
                policyTypeParams = "initialExplorationNoise = 0.25, minExplorationNoise = 0.25";
            }
            case 3 -> {
                executablePolicyType = ExecutablePolicyType.SAMPLED;
                policyTypeParams = "thresholdInitial = 0.2, thresholdMin = 0.2";
            }
        }
        AgentFactory.AgentAlgorithmType agentAlgorithmType = AgentFactory.AgentAlgorithmType.PPO;
        String algorithmParams = switch (agentAlgorithmType) {
            case SACDiscrete -> "applyImportanceSamplingWeights = false, applyUniformSampling = true, capacity = 20000, targetFunctionUpdateCycle = 0, targetFunctionTau = 0.01, agentUpdateCycle = 1";
            case MCTS -> "gamma = 1, updateValuePerEpisode = true";
            default -> "";
        };

        String params = "";
        if (policyTypeParams.isEmpty() && !algorithmParams.isEmpty()) params = algorithmParams;
        if (!policyTypeParams.isEmpty() && algorithmParams.isEmpty()) params = policyTypeParams;
        if (!policyTypeParams.isEmpty() && !algorithmParams.isEmpty()) params = policyTypeParams + ", " + algorithmParams;

        Agent agent = AgentFactory.createAgent(this, agentAlgorithmType, this, stateSize, 4, onlineMemory, singleFunctionEstimator, executablePolicyType, params);
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
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public NeuralNetwork buildNeuralNetwork(int inputSize, int outputSize, boolean policyGradient, boolean stateValue) throws DynamicParamException, NeuralNetworkException, MatrixException {
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.addInputLayer("width = " + inputSize);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.ELU), "width = " + 30);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), "width = " + 30);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.GELU), "width = " + 30);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.ELU), "width = " + 30);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), "width = " + 30);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.GELU), "width = " + 30);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), "width = " + (outputSize + (!policyGradient ? (stateValue ? 1 : 0) : 0)));
        neuralNetwork.addOutputLayer(!policyGradient ? BinaryFunctionType.MEAN_SQUARED_ERROR : BinaryFunctionType.DIRECT_GRADIENT);
        neuralNetwork.build();
        neuralNetwork.setOptimizer(OptimizationType.RADAM);
        if (!policyGradient) neuralNetwork.verboseTraining(10);
        return neuralNetwork;
    }

    /**
     * Build neural network for maze (agent).
     *
     * @param inputSize input size of neural network (number of states)
     * @param outputSize output size of neural network (number of actions and their values).
     * @return built neural network
     * @throws DynamicParamException throws exception if setting of dynamic parameters fails.
     * @throws NeuralNetworkException throws exception if building of neural network fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public NeuralNetwork buildNeuralNetwork(int inputSize, int outputSize) throws DynamicParamException, NeuralNetworkException, MatrixException {
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.addInputLayer("width = " + inputSize);
        neuralNetwork.addInputLayer("width = " + inputSize);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.ELU), "width = " + 30);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), "width = " + 30);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.GELU), "width = " + 30);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.ELU), "width = " + 30);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), "width = " + 30);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.GELU), "width = " + 30);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), "width = " + (1 + outputSize) + ", splitOutputAtPosition = 1");
        neuralNetwork.addOutputLayer(new BinaryFunctionType[] {BinaryFunctionType.MEAN_SQUARED_ERROR,BinaryFunctionType.DIRECT_GRADIENT});
        neuralNetwork.build();
        neuralNetwork.setOptimizer(OptimizationType.RADAM);
        neuralNetwork.verboseTraining(10);
        return neuralNetwork;
    }

}
