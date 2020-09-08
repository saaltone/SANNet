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
import core.regularization.RegularizationType;
import core.reinforcement.*;
import core.reinforcement.algorithm.*;
import core.reinforcement.function.DirectFunctionEstimator;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.function.NNFunctionEstimator;
import core.reinforcement.function.TabularFunctionEstimator;
import core.reinforcement.memory.Memory;
import core.reinforcement.memory.OnlineMemory;
import core.reinforcement.memory.PriorityMemory;
import core.reinforcement.policy.*;
import core.reinforcement.policy.executablepolicy.*;
import core.reinforcement.policy.executablepolicy.NoisyNextBestPolicy;
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

        /**
         * Calculates distance to another city.
         *
         * @param city another city.
         * @return distance to another city.
         */
        double distanceTo(City city) {
            return Math.sqrt(Math.pow(x - city.x, 2) + Math.pow(y - city.y, 2));
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
     * Episode ID
     *
     */
    private int episodeID = 0;

    /**
     * Current time stamp of episode.
     *
     */
    private int timeStamp = 0;

    /**
     * State of travelling salesman problem. State contains time stamp of episode, coordinates of cities as state and available actions.<br>
     * If city has been marked with negative coordinates it means city has been already visited during journey.<br>
     *
     */
    private EnvironmentState environmentState;

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
        agent = createAgent(4 * cityAmount, cityAmount - 1);
    }

    /**
     * Updates state. State is updates based on coordinates of cities and cities visited (negated coordinates).
     *
     */
    private void updateState() {
        Matrix state = new DMatrix(4 * cities.size(), 1);
        for (Integer index : cities.keySet()) {
            City city = cities.get(index);
            int visited = visitedCities.contains(index) ? 0 : cities.size();
            state.setValue(visited + 2 * index, 0, city.x / 10);
            state.setValue(visited + 2 * index + 1, 0, city.y / 10);
        }

        HashSet<Integer> availableActions = new HashSet<>();
        for (Integer city : cities.keySet()) {
            if (!visitedCities.contains(city)) if (city != startCity) availableActions.add(city - 1);
        }

        environmentState = new EnvironmentState(episodeID, timeStamp, state, availableActions);
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
     * True if state is terminal. This is usually true if episode is completed.
     *
     * @return true if state is terminal.
     */
    public boolean isTerminalState() {
        return visitedCities.size() == cities.size();
    }

    /**
     * Takes specific action.
     *
     * @param agent  agent that is taking action.
     * @param action action to be taken.
     */
    public void commitAction(Agent agent, int action) {
        visitedCities.add(action + 1);
        updateState();
        setReward(agent);
    }

    /**
     * Requests immediate reward from environment after taking action.
     *
     * @param agent agent that is asking for reward.
     */
    public void setReward(Agent agent) {
        boolean isTerminalState = isTerminalState();
        int cityIndex = visitedCities.size() - 1;
        int fromCity = isTerminalState ? visitedCities.get(cityIndex) : visitedCities.get(cityIndex - 1);
        int toCity = isTerminalState ? visitedCities.get(0) : visitedCities.get(cityIndex);
        double distance = cities.get(fromCity).distanceTo(cities.get(toCity));
        totalDistance += distance;
        if (isTerminalState) {
            if (totalDistance < minDistance) {
                minDistance = totalDistance;
                visitedCitiesMin = visitedCities;
            }
            if (totalDistance > maxDistance) {
                maxDistance = totalDistance;
                visitedCitiesMax = visitedCities;
            }
            agent.respond(totalDistance == minDistance ? 1 : 0.75 * (1 - totalDistance / maxDistance));
//            agent.respond(1 - totalDistance / 100);
//            agent.respond(1 - 10 / totalDistance);
//            agent.respond((totalDistance - minDistance) / (maxDistance - minDistance));
//            agent.respond(1 - totalDistance / maxDistance);
//            agent.respond(Math.pow(1 - totalDistance / maxDistance, 2));
        }
//        else agent.respond(1 - totalDistance / 100);
        else agent.respond(0);
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
    private void route(boolean redraw) throws MatrixException, NeuralNetworkException, DynamicParamException, AgentException {
        resetRoute();

        getAgent().newEpisode();
        episodeID++;
        timeStamp = 0;
        while (!isTerminalState()) {
            timeStamp++;
            getAgent().newStep();
            getAgent().act();
        }
        getAgent().endEpisode();

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
        Memory estimatorMemory = true ? new OnlineMemory() : new PriorityMemory();
        FunctionEstimator policyEstimator;
        FunctionEstimator valueEstimator;
        if (false) {
            // Uses single neural network estimator for both policy and value functions (works for policy gradients).
            NeuralNetwork stateActionValueNN = buildNeuralNetwork(inputAmount, outputAmount);
            policyEstimator = new NNFunctionEstimator(estimatorMemory, stateActionValueNN, outputAmount);
            valueEstimator = new NNFunctionEstimator(estimatorMemory, stateActionValueNN, 1);
        }
        else {
            // Uses separate estimators for value and policy functions.
            policyEstimator = nnPolicyEstimator ? new NNFunctionEstimator(estimatorMemory, buildNeuralNetwork(inputAmount, outputAmount, policyGradient, false), outputAmount) : new TabularFunctionEstimator(estimatorMemory, outputAmount);
            valueEstimator = nnValueEstimator ? new NNFunctionEstimator(estimatorMemory, buildNeuralNetwork(inputAmount, outputAmount, false, stateValue), (stateValue ? 1 : outputAmount)) : new TabularFunctionEstimator(estimatorMemory, outputAmount);
        }
        ExecutablePolicy executablePolicy = null;
        switch (policyType) {
            case 1:
                executablePolicy = new EpsilonGreedyPolicy("epsilonDecayRate = 0.999, epsilonMin = 0");
                break;
            case 2:
                executablePolicy = new NoisyNextBestPolicy("explorationNoiseDecay = 0.999, minExplorationNoise = 0");
                break;
            case 3:
                executablePolicy = new SampledPolicy("thresholdMin = 0");
                break;
        }
        Agent agent;
        if (!policyGradient) {
//            agent = new DDQNLearning(this, new ActionableBasicPolicy(executablePolicy, valueEstimator), new QTargetValueFunctionEstimator(valueEstimator));
//            agent = new DQNLearning(this, new ActionableBasicPolicy(executablePolicy, valueEstimator), new QValueFunctionEstimator(valueEstimator));
            agent = new Sarsa(this, new ActionableBasicPolicy(executablePolicy, valueEstimator), new ActionValueFunctionEstimator(valueEstimator));
        }
        else {
            ActionablePolicy actionablePolicy = basicPolicy ? new UpdateableBasicPolicy(executablePolicy, policyEstimator) : new UpdateableProximalPolicy(executablePolicy, policyEstimator);
//            agent = new PolicyGradient(this, actionablePolicy,new PlainValueFunction(outputAmount, new DirectFunctionEstimator(estimatorMemory, outputAmount)));
            agent = new ActorCritic(this, actionablePolicy, new StateValueFunctionEstimator(valueEstimator));
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
        neuralNetwork.addHiddenLayer(LayerType.GRU, "width = 100");
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), "width = 100");
        if (!policyFunction) {
            neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), "width = " + (stateValue ? 1 : outputSize));
            neuralNetwork.addOutputLayer(BinaryFunctionType.HUBER);
            neuralNetwork.build();
            neuralNetwork.verboseTraining(10);
        }
        else {
            neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.SOFTMAX), "width = " + outputSize);
            neuralNetwork.addOutputLayer(BinaryFunctionType.DIRECT_GRADIENT);
            neuralNetwork.build();
        }
        neuralNetwork.setOptimizer(OptimizationType.RADAM);
        return neuralNetwork;
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
    private static NeuralNetwork buildNeuralNetwork(int inputSize, int outputSize) throws DynamicParamException, NeuralNetworkException, MatrixException {
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        neuralNetwork.addInputLayer("width = " + inputSize);
        neuralNetwork.addHiddenLayer(LayerType.GRU, "width = 100");
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), "width = 100");
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), "width = " + (1 + outputSize));
        neuralNetwork.addOutputLayer(BinaryFunctionType.POLICY_VALUE);
        neuralNetwork.build();
        neuralNetwork.setOptimizer(OptimizationType.RADAM);
        neuralNetwork.verboseTraining(10);
        return neuralNetwork;
    }

}
