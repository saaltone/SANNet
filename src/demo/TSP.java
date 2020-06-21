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
import core.reinforcement.policy.*;
import core.reinforcement.value.*;
import utils.*;
import utils.matrix.*;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
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
    private static class City {

        /**
         * Coordinate x of city.
         *
         */
        final double x;

        /**
         * Coordinate y of city.
         *
         */
        final double y;

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
     * Number of cities for travelling salesman.
     *
     */
    private static final int numberOfCities = 5;

    /**
     * Hashmap for storing cities.
     *
     */
    private final HashMap<Integer, City> cities = new HashMap<>();

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
    private final Agent agent;

    /**
     * Constructor for travelling salesman problem.
     *
     * @param cityAmount number of cities to be visited.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if setting of dynamic parameters fails.
     * @throws IOException throws exception if copying of neural network instance fails.
     * @throws ClassNotFoundException throws exception if copying of neural network instance fails.
     */
    public TSP(int cityAmount) throws NeuralNetworkException, MatrixException, DynamicParamException, IOException, ClassNotFoundException {
        Random random = new Random();
        for (int city = 0; city < cityAmount; city++) cities.put(city, new City(10 * random.nextDouble(), 10 * random.nextDouble()));
        agent = createAgent(4 * cityAmount, cityAmount);
    }

    /**
     * Updates state. State is updates based on coordinates of cities and cities visited (negated coordinates).
     *
     */
    private void updateState() {
        state = new DMatrix(4 * cities.size(), 1);
        for (Integer index : cities.keySet()) {
            City city = cities.get(index);
            int visited = visitedCities.contains(index) ? 0 : cities.size();
            state.setValue(visited + 2 * index, 0, city.x / 10);
            state.setValue(visited + 2 * index + 1, 0, city.y / 10);
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
     * Returns total distance travelling salesman has travelled.
     *
     * @return total distance travelling salesman has travelled.
     */
    private double getTotalDistance() {
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
     * Returns list of cities added into hashmap by indices starting from zero.
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
            tsp = new TSP(numberOfCities);
            tsp.initWindow();
            for (int tour = 0; tour < 1000000; tour++) {
                tsp.route(tour % 10 == 0);
                System.out.println("Tour #" + (tour + 1) + " Total: " + tsp.getTotalDistance() + " Min: " + tsp.getMinDistance() + " Max: " + tsp.getMaxDistance());
            }
            tsp.stop();
        } catch (Exception exception) {
            exception.printStackTrace();
            System.exit(-1);
        }
    }

    /**
     * Returns current state of environment.
     *
     * @return state of environment.
     */
    public Matrix getState() {
        return state;
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
    public HashSet<Integer> getAvailableActions() {
        HashSet<Integer> availableActions = new HashSet<>();
        if (visitedCities.size() == cities.size()) availableActions.add(startCity);
        else {
            for (Integer city : cities.keySet()) {
                if (!visitedCities.contains(city)) availableActions.add(city);
            }
        }
        return availableActions;
    }

    /**
     * Takes specific action.
     *
     * @param agent  agent that is taking action.
     * @param action action to be taken.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void commitAction(Agent agent, int action) throws NeuralNetworkException, MatrixException, DynamicParamException {
        visitedCities.add(action);
        updateState();
        setReward(agent);
    }

    /**
     * Requests immediate reward from environment after taking action.
     *
     * @param agent agent that is asking for reward.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setReward(Agent agent) throws NeuralNetworkException, MatrixException, DynamicParamException {
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
            agent.respond(minDistance / totalDistance, true);
        }
        else agent.respond(0, false);
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
     * Returns minimum journey travelling salesman has taken.
     *
     * @return minimum journey travelling salesman has taken.
     */
    private double getMinDistance() {
        return minDistance;
    }

    /**
     * Returns maximum journey travelling salesman has taken.
     *
     * @return maximum journey travelling salesman has taken.
     */
    private double getMaxDistance() {
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
        void addCities(ArrayList<Integer> newDrawCities, ArrayList<Integer> newPreviousDrawCities) {
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
    private void initWindow() {
        JFrame.setDefaultLookAndFeelDecorated(true);
        jFrame = new JFrame("Travelling Salesman Problem (" + numberOfCities + " cities)");
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
     * Single journey of travelling salesman taken by deep agent.
     *
     * @param redraw if true current journey is drawn to window.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if building of neural network fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private void route(boolean redraw) throws MatrixException, NeuralNetworkException, DynamicParamException {
        resetRoute();

        getAgent().newEpisode();
        while (!isTerminalState()) {
            getAgent().newStep();
            getAgent().act();
        }

        if (redraw) {
            jFrame.remove(tspPanel);
            tspPanel = new TSPPanel();
            jFrame.add(tspPanel);
            tspPanel.addCities(visitedCities, visitedCitiesPrevious);
            jFrame.revalidate();
            tspPanel.paintImmediately(0, 0, 400, 400);
        }
    }

    /**
     * Stops deep agent.
     *
     */
    private void stop() {
        agent.stop();
    }

    /**
     * Creates agent (player).
     *
     * @return agent
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException throws exception if setting of dynamic parameter fails.
     */
    private Agent createAgent(int inputAmount, int outputAmount) throws MatrixException, NeuralNetworkException, DynamicParamException, IOException, ClassNotFoundException {
        boolean policyGradient = true;
        boolean stateValue = true;
        int policyType = 1;
        boolean nnPolicyEstimator = true;
        boolean nnValueEstimator = true;
        boolean basicPolicy = false;
        FunctionEstimator policyEstimator = nnPolicyEstimator ? new NNFunctionEstimator(buildNeuralNetwork(inputAmount, outputAmount, policyGradient, false)) : new TabularFunctionEstimator(outputAmount);
        FunctionEstimator valueEstimator = nnValueEstimator ? new NNFunctionEstimator(buildNeuralNetwork(inputAmount, (stateValue ? 1 : outputAmount), false, stateValue)) : new TabularFunctionEstimator(outputAmount);
        Policy policy = null;
        switch (policyType) {
            case 1:
                policy = new EpsilonGreedyPolicy("epsilonMin = 0");
                break;
            case 2:
                policy = new NoisyPolicy("minExplorationNoise = 0");
                break;
            case 3:
                policy = new WeightedRandomPolicy();
                break;
        }
        Agent agent;
        if (!policyGradient) {
            agent = new DDQNLearning(this, new ActionableBasicPolicy(policy, valueEstimator), new ReplayBuffer(), new QTargetValueFunctionEstimator(valueEstimator));
//            agent = new DQNLearning(this, new ActionableBasicPolicy(policy, valueEstimator), new OnlineBuffer(), new QValueFunctionEstimator(valueEstimator, "useBaseline = True"));
//            agent = new Sarsa(this, new ActionableBasicPolicy(policy, valueEstimator), new OnlineBuffer(), new ValueFunctionEstimator(valueEstimator));
        }
        else {
            UpdateablePolicy updateablePolicy = basicPolicy ? new UpdateableBasicPolicy(policy, policyEstimator) : new UpdateableProximalPolicy(policy, policyEstimator, "epsilon = 0.1");
//            agent = new PolicyGradient(this, updateablePolicy, new OnlineBuffer(), new PlainValueFunction(outputAmount));
            agent = new ActorCritic(this, updateablePolicy, new OnlineBuffer(), new ValueFunctionEstimator(valueEstimator, "useBaseline = True, bootstrap = True"));
        }
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
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    private static NeuralNetwork buildNeuralNetwork(int inputSize, int outputSize, boolean policyFunction, boolean stateValue) throws DynamicParamException, NeuralNetworkException, MatrixException {
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.addInputLayer("width = " + inputSize);
        if (!policyFunction) {
            neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.ELU), "width = 60");
            neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.ELU), "width = 60");
            neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.ELU), "width = " + (stateValue ? 1 : outputSize));
            neuralNetwork.addOutputLayer(BinaryFunctionType.HUBER);
            neuralNetwork.build();
            neuralNetwork.addNormalizer(3, NormalizationType.BATCH_NORMALIZATION, "beta = 0.9");
            neuralNetwork.setOptimizer(OptimizationType.ADAM);
            neuralNetwork.verboseTraining(10);
        }
        else {
            neuralNetwork.addHiddenLayer(LayerType.GRAVESLSTM, "width = 60");
            neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.ELU), "width = 60");
            neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.SOFTMAX), "width = " + outputSize);
            neuralNetwork.addOutputLayer(BinaryFunctionType.DIRECT_GRADIENT);
            neuralNetwork.build();
            neuralNetwork.addNormalizer(3, NormalizationType.WEIGHT_NORMALIZATION);
            neuralNetwork.setOptimizer(OptimizationType.ADAM);
        }
        return neuralNetwork;
    }

}
