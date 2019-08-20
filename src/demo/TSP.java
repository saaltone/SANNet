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
import utils.DMatrix;
import utils.DynamicParamException;
import utils.Matrix;
import utils.MatrixException;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

/**
 * Class that implements deep reinforcement learning solution solving travelling salesman problem.
 *
 */
public class TSP implements Environment {

    /**
     * Class that defines city with coordinates x and y.
     *
     */
    private class City {

        /**
         * Coordinate x of city.
         *
         */
        double x;

        /**
         * Coordinate y of city.
         *
         */
        double y;

        /**
         * Constructor for city.
         *
         * @param x coordinate x of city.
         * @param y coordinate y of city.
         */
        City(double x, double y) {
            this.x = x;
            this.y = y;
        }

    }

    /**
     * Random number generator.
     *
     */
    private final Random random = new Random();

    /**
     * Hashmap for storing cities.
     *
     */
    private HashMap<Integer, City> cities = new HashMap<>();

    /**
     * Ordered list of visited cities.
     *
     */
    private ArrayList<Integer> visitedCities;

    /**
     * Ordered list of visited cities for previous tour.
     *
     */
    private ArrayList<Integer> visitedCitiesPrevious = null;

    /**
     * Ordered list of visited cities for shortest tour.
     *
     */
    private ArrayList<Integer> visitedCitiesMin = null;

    /**
     * Ordered list of visited cities for longest tour.
     *
     */
    private ArrayList<Integer> visitedCitiesMax = null;

    /**
     * Start city from where travelling salesman start journey from.
     *
     */
    private final int startCity = 0;

    /**
     * Total distance that travelling salesman has travelled.
     *
     */
    private double totalDistance = 0;

    /**
     * Length of shortest route found.
     *
     */
    private double minDistance = Double.MAX_VALUE;

    /**
     * Length of longest route found.
     *
     */
    private double maxDistance = Double.MIN_VALUE;

    /**
     * State of travelling salesman problem. State contains coordinates of cities.<br>
     * If city has been marked with negative coordinates it means city has been already visited during journey.<br>
     *
     */
    private Matrix state;

    /**
     * Agent that solves travelling salesman problem. Agent acts as travelling salesman.
     *
     */
    private DeepAgent agent;

    /**
     * Constructor for travelling salesman problem.
     *
     * @param cityAmount number of cities to be visited.
     * @throws NeuralNetworkException throws exception if building of neural network fails.
     * @throws DynamicParamException throws exception if setting of dynamic parameters fails.
     * @throws IOException throws exception if coping of neural network instance fails.
     * @throws ClassNotFoundException throws exception if coping of neural network instance fails.
     */
    public TSP(int cityAmount) throws NeuralNetworkException, DynamicParamException, IOException, ClassNotFoundException {
        for (int city = 0; city < cityAmount; city++) cities.put(city, new City(10 * random.nextDouble(), 10 * random.nextDouble()));
        agent = createAgent(2 * cityAmount, cityAmount);
    }

    /**
     * Updates state. State is updates based on coordinates of cities and cities visited (negated coordinates).
     *
     */
    private void updateState() {
        state = new DMatrix(2 * cities.size(), 1);
        for (Integer index : cities.keySet()) {
            City city = cities.get(index);
            double visited = visitedCities.contains(index) ? -1 : 1;
            state.setValue(2 * index, 0, city.x / 10 * visited);
            state.setValue(2 * index + 1, 0, city.y / 10 * visited);
        }
    }

    /**
     * Clears route for new journey.
     *
     */
    private void resetRoute() {
        visitedCitiesPrevious = visitedCities;
        visitedCities = new ArrayList<>();
        visitedCities.add(startCity);
        totalDistance = 0;
        updateState();
    }

    /**
     * Return total distance travelling salesman has travelled.
     *
     * @return total distance travelling salesman has travelled.
     */
    public double getTotalDistance() {
        return totalDistance;
    }

    /**
     * Returns shortest route found by deep agent (travelling salesman) as indices of cities.
     *
     * @return shortest route as indices of cities.
     */
    public ArrayList<Integer> getShortestRoute() {
        return visitedCitiesMin;
    }

    /**
     * Returns longest route found by deep agent (travelling salesman) as indices of cities.
     *
     * @return longest route as indices of cities.
     */
    public ArrayList<Integer> getLongestRoute() {
        return visitedCitiesMax;
    }

    /**
     * Returns latest (shortest) route found by deep agent (travelling salesman) as indices of cities.
     *
     * @return latest (shortest) route as indices of cities.
     */
    public ArrayList<Integer> getLatestRoute() {
        return visitedCities;
    }

    /**
     * Gets list of cities added into hashmap by indices starting from zero.
     *
     * @return list of cities.
     */
    public HashMap<Integer, City> getCities() {
        return cities;
    }

    /**
     * Main function for travelling sales man.
     *
     * @param args not used.
     */
    public static void main(String[] args) {
        TSP tsp;
        try {
            tsp = new TSP(50);
            tsp.initWindow();
            for (int tour = 0; tour < 100000; tour++) {
                int illegalMoves = tsp.route(tour % 10 == 0);
                System.out.println("Tour #" + (tour + 1) + " Total: " + tsp.getTotalDistance() + " Min: " + tsp.getMinDistance() + " Max: " + tsp.getMaxDistance() + " Illegal moves: " + illegalMoves + " (Epsilon: " + tsp.getEpsilon() + ")");

            }
            tsp.stop();
        } catch (Exception exception) {
            exception.printStackTrace();
            System.exit(-1);
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
     * True if state is terminal. This is usually true if episode is completed.
     *
     * @return true if state is terminal.
     */
    public boolean isTerminalState() {
        return visitedCities.size() == cities.size() + 1;
    }

    /**
     * Returns available actions in current state of environment
     *
     * @return available actions in current state of environment.
     */
    public ArrayList<Integer> getAvailableActions() {
        ArrayList<Integer> availableActions = new ArrayList<>();
        if (visitedCities.size() == cities.size()) availableActions.add(startCity);
        else {
            for (Integer city : cities.keySet()) {
                if (!visitedCities.contains(city)) availableActions.add(city);
            }
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
        visitedCities.add(action);
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
            int fromCity = visitedCities.get(visitedCities.size() - 2);
            int toCity = visitedCities.get(visitedCities.size() - 1);
            double distance = getDistance(cities.get(fromCity), cities.get(toCity));
            totalDistance += distance;
            if (isTerminalState()) {
                if (totalDistance < minDistance) {
                    minDistance = totalDistance;
                    visitedCitiesMin = visitedCities;
                }
                if (totalDistance > maxDistance) {
                    maxDistance = totalDistance;
                    visitedCitiesMax = visitedCities;
                }
            }
            return 0.5 * (maxDistance / cities.size() - distance);
        }
    }

    /**
     * Returns euclidean distance between two cities.
     *
     * @param city1 first city for calculation.
     * @param city2 second city for calculation.
     * @return euclidean distance between two cities.
     */
    private double getDistance(City city1, City city2) {
        return Math.sqrt(Math.pow(city1.x - city2.x, 2) + Math.pow(city1.y - city2.y, 2));
    }

    /**
     * Gets minimum journey travelling salesman has taken.
     *
     * @return minimum journey travelling salesman has taken.
     */
    public double getMinDistance() {
        return minDistance;
    }

    /**
     * Gets maximum journey travelling salesman has taken.
     *
     * @return maximum journey travelling salesman has taken.
     */
    public double getMaxDistance() {
        return maxDistance;
    }

    /**
     * Implements JPanel that draws journey of travelling salesman.<br>
     * Routes between cities that are unchanged are drawn by black color and changed routes by red color.<br>
     *
     */
    class TSPPanel extends JPanel {

        /**
         * List of cities to be drawn with given order.
         *
         */
        private final ArrayList<Integer> drawCities = new ArrayList<>();

        /**
         * List of previous cities.
         *
         */
        private final ArrayList<Integer> previousDrawCities = new ArrayList<>();

        /**
         * If there is list of previous cities.
         *
         */
        private boolean previousExists = false;

        /**
         * Add cities in order to be drawn.
         *
         * @param newDrawCities list of cities to be drawn.
         */
        public void addCities(ArrayList<Integer> newDrawCities, ArrayList<Integer> newPreviousDrawCities) {
            drawCities.clear();
            drawCities.addAll(newDrawCities);
            if (newPreviousDrawCities != null) {
                previousDrawCities.clear();
                previousDrawCities.addAll(newPreviousDrawCities);
                previousExists = true;
            }
            else previousExists = false;
        }

        /**
         * Paints journey to JPanel.
         *
         * @param g graphics.
         */
        public void paintComponent(Graphics g) {
            super.paintComponent(g);
            if (drawCities.isEmpty()) return;
            City lastCity = null;
            int lastCityIndex = -1;
            int previousLastCityIndex = -2;
            for (int index = 0; index < drawCities.size() - 1; index++) {
                City city1 = cities.get(drawCities.get(index));
                City city2 = cities.get(drawCities.get(index + 1));
                lastCity = city2;
                lastCityIndex = drawCities.get(index + 1);
                if (previousExists) {
                    if (drawCities.get(index).equals(previousDrawCities.get(index)) && drawCities.get(index + 1).equals(previousDrawCities.get(index + 1))) g.setColor(Color.BLACK);
                    else g.setColor(Color.RED);
                    previousLastCityIndex = previousDrawCities.get(index + 1);
                }
                else g.setColor(Color.BLACK);
                g.drawLine(50 + (int)(city1.x * 40), 50 + (int)(city1.y * 40), 50 + (int)(city2.x * 40), 50 + (int)(city2.y * 40));
            }
            if (lastCity != null) {
                if (previousExists) {
                    if (lastCityIndex == previousLastCityIndex) g.setColor(Color.BLACK);
                    else g.setColor(Color.RED);
                }
                else g.setColor(Color.BLACK);
                City initialCity = cities.get(startCity);
                g.drawLine(50 + (int)(lastCity.x * 40), 50 + (int)(lastCity.y * 40), 50 + (int)(initialCity.x * 40), 50 + (int)(initialCity.y * 40));
            }
        }

    }

    /**
     * Reference to travelling salesman problem JPanel class.
     *
     */
    private TSPPanel tspPanel = new TSPPanel();

    /**
     * JFrame for travelling salesman problem.
     *
     */
    private JFrame jFrame;

    /**
     * Initialized window for travelling salesman problem.
     *
     */
    public void initWindow() {
        JFrame.setDefaultLookAndFeelDecorated(true);
        jFrame = new JFrame("Travelling Salesman Problem");
        jFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        jFrame.setBackground(Color.white);
        jFrame.setSize(500, 500);
        jFrame.add(tspPanel);
        jFrame.setVisible(true);
    }

    /**
     * Returns active agent (player)
     *
     * @return active agent
     */
    private Agent getAgent() {
        return agent;
    }

    /**
     * Return current value of epsilon.
     *
     * @return current value of epsilon.
     */
    public double getEpsilon() {
        return agent.getEpsilon();
    }

    /**
     * Single journey of travelling salesman taken by deep agent.
     *
     * @param redraw if true current journey is drawn to window.
     * @return number of illegal moves taken by deep agent.
     * @throws AgentException throws exception if agent operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if building of neural network fails.
     * @throws IOException throws exception if coping of neural network instance fails.
     * @throws ClassNotFoundException throws exception if coping of neural network instance fails.
     */
    private int route(boolean redraw) throws AgentException, MatrixException, NeuralNetworkException, IOException, ClassNotFoundException {
        int illegalMoves = 0;
        resetRoute();
        while (!(visitedCities.size() == cities.size() + 1)) {
            getAgent().newStep();
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
            getAgent().updateValue();
        }

        getAgent().commitStep();

        if (redraw) {
            jFrame.remove(tspPanel);
            tspPanel = new TSPPanel();
            jFrame.add(tspPanel);
            tspPanel.addCities(visitedCities, visitedCitiesPrevious);
            jFrame.revalidate();
            tspPanel.paintImmediately(0, 0, 400, 400);
        }

        return illegalMoves;
    }

    /**
     * Stops deep agent.
     *
     */
    public void stop() {
        agent.stop();
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
    private DeepAgent createAgent(int inputAmount, int outputAmount) throws NeuralNetworkException, DynamicParamException, IOException, ClassNotFoundException {
        NeuralNetwork QNN = buildNeuralNetwork(inputAmount, outputAmount);
        DeepAgent agent = new DeepAgent(this, QNN, "trainCycle = " + (10 * outputAmount) + ", updateTNNCycle = " + (30 * outputAmount));
        agent.start();
        return agent;
    }

    /**
     * Build neural network for travelling salesman (agent).
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
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(ActivationFunctionType.SELU), "width = " + inputSize);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(ActivationFunctionType.SELU), "width = " + outputSize);
        neuralNetwork.addOutputLayer(LayerType.FEEDFORWARD, new ActivationFunction(ActivationFunctionType.SELU), "width = " + outputSize);
        neuralNetwork.build();
        neuralNetwork.setOptimizer(OptimizationType.AMSGRAD);
        neuralNetwork.addNormalizer(2, NormalizationType.WEIGHT_NORMALIZATION);
        neuralNetwork.setLossFunction(LossFunctionType.HUBER);
        neuralNetwork.setTrainingSampling(100, false, true);
        neuralNetwork.setTrainingIterations(50);
        neuralNetwork.verboseTraining(100);
        return neuralNetwork;
    }

}
