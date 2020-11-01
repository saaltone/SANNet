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

/**
 * Class that implements deep reinforcement learning solution solving travelling salesman problem.
 *
 */
public class TSP implements Environment {

    /**
     * Class that defines tour.
     *
     */
    private static class Tour {

        /**
         * Number of cities for travelling salesman.
         *
         */
        private final int numberOfCities;

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
         * Total distance in formalized form that travelling salesman has travelled.
         *
         */
        private double totalNormalizedDistance = 0;

        /**
         * Total distance form that travelling salesman has travelled.
         *
         */
        private double totalDistance = 0;

        /**
         * Length of shortest route found as normalized distance.
         *
         */
        private double minNormalizedDistance = Double.MAX_VALUE;

        /**
         * Length of shortest route found.
         *
         */
        private double minDistance = Double.MAX_VALUE;

        /**
         * Length of longest route found as normalized distance.
         *
         */
        private double maxNormalizedDistance = Double.MIN_VALUE;

        /**
         * Length of longest route found.
         *
         */
        private double maxDistance = Double.MIN_VALUE;

        /**
         * Distance from previous tour as normalized distance.
         *
         */
        private double lastNormalizedDistance = Double.MIN_VALUE;

        /**
         * Distance from previous tour.
         *
         */
        private double lastDistance = Double.MIN_VALUE;

        /**
         * Constructor for tour
         *
         * @param numberOfCities number of cities for a tour.
         */
        Tour(int numberOfCities) {
            this.numberOfCities = numberOfCities;
            for (int index = 0; index < numberOfCities; index++) cities.put(index, new City(Math.random(), Math.random()));
            normalize();
        }

        /**
         * Size of tour as number of cities.
         *
         * @return size of tour as number of cities.
         */
        int size() {
            return numberOfCities;
        }

        /**
         * Normalizes tour coordinates.
         *
         */
        void normalize() {
            double xMin = Double.MAX_VALUE;
            double xMax = Double.MIN_VALUE;
            double yMin = Double.MAX_VALUE;
            double yMax = Double.MIN_VALUE;
            for (City city : cities.values()) {
                xMin = Math.min(xMin, city.x);
                xMax = Math.max(xMax, city.x);
                yMin = Math.min(yMin, city.y);
                yMax = Math.max(yMax, city.y);
            }
            for (City city : cities.values()) city.normalize(xMin, xMax, yMin, yMax);
        }

        /**
         * Resets tour.
         *
         */
        void reset() {
            if (visitedCities != null) {
                visitedCitiesPrevious = new ArrayList<>(visitedCities);
                lastNormalizedDistance = totalNormalizedDistance;
                lastDistance = totalDistance;
            }
            visitedCities = new ArrayList<>();
            visitedCities.add(startCity);
            totalNormalizedDistance = 0;
            totalDistance = 0;
        }

        /**
         * Add visited city to tour.
         *
         * @param cityIndex city index.
         */
        void addVisitedCity(int cityIndex) {
            visitedCities.add(cityIndex);
            int lastIndex = visitedCities.size() - 1;
            City cityFrom = cities.get(visitedCities.size() < cities.size() ? visitedCities.get(lastIndex - 1) : visitedCities.get(lastIndex));
            City cityTo = cities.get(visitedCities.size() < cities.size() ? visitedCities.get(lastIndex) : visitedCities.get(0));
            totalNormalizedDistance += cityFrom.distanceTo(cityTo, true);
            totalDistance += cityFrom.distanceTo(cityTo, false);
            if (isCompleteTour()) record();
        }

        /**
         * Records tour.
         *
         */
        void record() {
            if (minNormalizedDistance == Double.MAX_VALUE || totalNormalizedDistance < minNormalizedDistance) {
                minNormalizedDistance = totalNormalizedDistance;
                minDistance = totalDistance;
                visitedCitiesMin = new ArrayList<>(visitedCities);
            }
            if (maxNormalizedDistance == Double.MIN_VALUE || totalNormalizedDistance > maxNormalizedDistance) {
                maxNormalizedDistance = totalNormalizedDistance;
                maxDistance = totalDistance;
                visitedCitiesMax = new ArrayList<>(visitedCities);
            }
        }

        /**
         * Returns true if tour is completed.
         *
         * @return true if tour is completed.
         */
        boolean isCompleteTour() {
            return visitedCities.size() == cities.size();
        }

        /**
         * Returns unvisited cities.
         *
         * @return unvisited cities.
         */
        HashSet<Integer> getUnvisitedCities() {
            HashSet<Integer> unvisitedCities = new HashSet<>();
            for (Integer index : cities.keySet()) if (!visitedCities.contains(index) && index != startCity) unvisitedCities.add(index - 1);
            return unvisitedCities;
        }

    }

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
         * Coordinate x of city.
         *
         */
        double xNormalized;

        /**
         * Coordinate y of city.
         *
         */
        double yNormalized;

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
         * Normalizes x and y coordinates
         *
         * @param xMin minimum value for x.
         * @param xMax maximum value for x.
         * @param yMin minimum value for y.
         * @param yMax maximum value for y.
         */
        void normalize(double xMin, double xMax, double yMin, double yMax) {
            xNormalized = (x - xMin) / (xMax - xMin);
            yNormalized = (y - yMin) / (yMax - yMin);
        }

        /**
         * Calculates distance to another city.
         *
         * @param city another city.
         * @param asNormalized as normalized distance
         * @return distance to another city.
         */
        double distanceTo(City city, boolean asNormalized) {
            return asNormalized ? Math.sqrt(Math.pow(xNormalized - city.xNormalized, 2) + Math.pow(yNormalized - city.yNormalized, 2)) : Math.sqrt(Math.pow(x - city.x, 2) + Math.pow(y - city.y, 2));
        }

    }


    /**
     * Number of cities for travelling salesman.
     *
     */
    private static final int numberOfCities = 5;

    /**
     * Current tour.
     *
     */
    private final Tour tour;

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
     * If true uses compact state representation i.e. defines visited cities as negative coordinates.
     *
     */
    private final boolean compactState = true;

    /**
     * Window size in x-coordinate direction.
     *
     */
    private final int xWindowSize = 500;

    /**
     * Window size in y-coordinate direction.
     *
     */
    private final int yWindowSize = 500;

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
        tour = new Tour(cityAmount);
        agent = compactState ? createAgent(2 * tour.size(), tour.size() - 1) : createAgent(4 * tour.size(), tour.size() - 1);
    }

    /**
     * Updates state. State is updates based on coordinates of cities and cities visited (negated coordinates).
     *
     */
    private void updateState() {
        Matrix state;
        if (compactState) {
            state = new DMatrix(2 * tour.cities.size(), 1);
            for (Integer index : tour.cities.keySet()) {
                City city = tour.cities.get(index);
                boolean visited = tour.visitedCities.contains(index);
                state.setValue(2 * index, 0, city.xNormalized * (visited ? -1 : 1));
                state.setValue(2 * index + 1, 0, city.yNormalized * (visited ? -1 : 1));
            }
        }
        else {
            state = new DMatrix(4 * tour.cities.size(), 1);
            for (Integer index : tour.cities.keySet()) {
                City city = tour.cities.get(index);
                int visited = tour.visitedCities.contains(index) ? 0 : tour.cities.size();
                state.setValue(visited + 2 * index, 0, city.xNormalized);
                state.setValue(visited + 2 * index + 1, 0, city.yNormalized);
            }
        }

        environmentState = new EnvironmentState(episodeID, timeStamp, state, tour.getUnvisitedCities());
    }

    /**
     * Clears route for new journey.
     *
     */
    private void resetRoute() {
        tour.reset();
        updateState();
    }

    /**
     * Returns total distance travelling salesman has travelled as normalized distance.
     *
     * @return total distance travelling salesman has travelled as normalized distance.
     */
    private double getTotalNormalizedDistance() {
        return tour.totalNormalizedDistance;
    }

    /**
     * Returns total distance travelling salesman has travelled.
     *
     * @return total distance travelling salesman has travelled.
     */
    private double getTotalDistance() {
        return tour.totalDistance;
    }

    /**
     * Returns minimum journey travelling salesman has taken as normalized distance.
     *
     * @return minimum journey travelling salesman has taken as normalized distance.
     */
    private double getMinNormalizedDistance() {
        return tour.minNormalizedDistance;
    }

    /**
     * Returns minimum journey travelling salesman has taken.
     *
     * @return minimum journey travelling salesman has taken.
     */
    private double getMinDistance() {
        return tour.minDistance;
    }

    /**
     * Returns maximum journey travelling salesman has taken as normalized distance.
     *
     * @return maximum journey travelling salesman has taken as normalized distance.
     */
    private double getMaxNormalizedDistance() {
        return tour.maxNormalizedDistance;
    }

    /**
     * Returns maximum journey travelling salesman has taken.
     *
     * @return maximum journey travelling salesman has taken.
     */
    private double getMaxDistance() {
        return tour.maxDistance;
    }

    /**
     * Returns shortest route found by deep agent (travelling salesman) as indices of cities.
     *
     * @return shortest route as indices of cities.
     */
    public ArrayList<Integer> getShortestRoute() {
        return tour.visitedCitiesMin;
    }

    /**
     * Returns longest route found by deep agent (travelling salesman) as indices of cities.
     *
     * @return longest route as indices of cities.
     */
    public ArrayList<Integer> getLongestRoute() {
        return tour.visitedCitiesMax;
    }

    /**
     * Returns latest (shortest) route found by deep agent (travelling salesman) as indices of cities.
     *
     * @return latest (shortest) route as indices of cities.
     */
    public ArrayList<Integer> getLatestRoute() {
        return tour.visitedCities;
    }

    /**
     * Returns list of cities added into hashmap by indices starting from zero.
     *
     * @return list of cities.
     */
    public HashMap<Integer, City> getCities() {
        return tour.cities;
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
                System.out.println("Tour #" + (tour + 1) + " Total: " + tsp.getTotalDistance() + " (" + tsp.getTotalNormalizedDistance() +")" + " Min: " + tsp.getMinDistance() + " (" + tsp.getMinNormalizedDistance() + ")" + " Max: " + tsp.getMaxDistance() + " (" + tsp.getMaxNormalizedDistance() + ")");
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
        return tour.isCompleteTour();
    }

    /**
     * Takes specific action.
     *
     * @param agent  agent that is taking action.
     * @param action action to be taken.
     */
    public void commitAction(Agent agent, int action) {
        tour.addVisitedCity(action + 1);
        updateState();
        setReward(agent);
    }

    /**
     * Requests immediate reward from environment after taking action.
     *
     * @param agent agent that is asking for reward.
     */
    public void setReward(Agent agent) {
        if (isTerminalState()) agent.respond((getMaxNormalizedDistance() - getMinNormalizedDistance() != 0 ? Math.pow(1 - (getTotalNormalizedDistance() - getMinNormalizedDistance()) / (getMaxNormalizedDistance() - getMinNormalizedDistance()), 5) : 0));
        else agent.respond(0);
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
                City city1 = tour.cities.get(drawCities.get(index));
                City city2 = tour.cities.get(drawCities.get(index + 1));
                lastCity = city2;
                lastCityIndex = drawCities.get(index + 1);
                if (previousExists) {
                    if (drawCities.get(index).equals(previousDrawCities.get(index)) && drawCities.get(index + 1).equals(previousDrawCities.get(index + 1))) g.setColor(Color.BLACK);
                    else g.setColor(Color.RED);
                    previousLastCityIndex = previousDrawCities.get(index + 1);
                }
                else g.setColor(Color.BLACK);
                g.drawLine((int)(0.1 * xWindowSize) + (int)(city1.xNormalized * (int)(0.8 * xWindowSize)), (int)(0.1 * yWindowSize) + (int)(city1.yNormalized * (int)(0.8 * yWindowSize)), (int)(0.1 * xWindowSize) + (int)(city2.xNormalized * (int)(0.8 * xWindowSize)), (int)(0.1 * yWindowSize) + (int)(city2.yNormalized * (int)(0.8 * yWindowSize)));
            }
            if (lastCity != null) {
                if (previousExists) {
                    if (lastCityIndex == previousLastCityIndex) g.setColor(Color.BLACK);
                    else g.setColor(Color.RED);
                }
                else g.setColor(Color.BLACK);
                City initialCity = tour.cities.get(tour.startCity);
                g.drawLine((int)(0.1 * xWindowSize) + (int)(lastCity.xNormalized * (int)(0.8 * xWindowSize)), (int)(0.1 * yWindowSize) + (int)(lastCity.yNormalized * (int)(0.8 * yWindowSize)), (int)(0.1 * xWindowSize) + (int)(initialCity.xNormalized * (int)(0.8 * xWindowSize)), (int)(0.1 * yWindowSize) + (int)(initialCity.yNormalized * (int)(0.8 * yWindowSize)));
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
        jFrame = new JFrame("Travelling Salesman Problem (" + tour.size() + " cities)");
        jFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        jFrame.setBackground(Color.white);
        jFrame.setSize(xWindowSize, yWindowSize);
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
            tspPanel.addCities(tour.visitedCities, tour.visitedCitiesPrevious);
            jFrame.revalidate();
            tspPanel.paintImmediately(0, 0, (int)(0.8 * xWindowSize), (int)(0.8 * yWindowSize));
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
        if (true) {
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
//        policyEstimator = nnPolicyEstimator ? new NNFunctionEstimator(estimatorMemory, buildNeuralNetwork(inputAmount, outputAmount), outputAmount) : new TabularFunctionEstimator(estimatorMemory, outputAmount);
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
//            agent = new DQNLearning(this, new ActionablePolicy(executablePolicy, valueEstimator), new QValueFunctionEstimator(valueEstimator));
//            agent = new DDQNLearning(this, new ActionablePolicy(executablePolicy, valueEstimator), new QTargetValueFunctionEstimator(valueEstimator));
            agent = new Sarsa(this, new ActionablePolicy(executablePolicy, valueEstimator), new ActionValueFunctionEstimator(valueEstimator));
        }
        else {
            Policy policy = basicPolicy ? new UpdateableBasicPolicy(executablePolicy, policyEstimator) : new UpdateableProximalPolicy(executablePolicy, policyEstimator);
//            agent = new PolicyGradient(this, policy,new PlainValueFunction(outputAmount, new DirectFunctionEstimator(estimatorMemory, outputAmount)));
            agent = new ActorCritic(this, policy, new StateValueFunctionEstimator(valueEstimator));
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
        String width = "width = " + (inputSize + 20);
        neuralNetwork.addHiddenLayer(LayerType.GRU, width);
//        neuralNetwork.addHiddenLayer(LayerType.GRU, width);
//        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), width);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), width);
        neuralNetwork.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), "width = " + (1 + outputSize));
        neuralNetwork.addOutputLayer(BinaryFunctionType.POLICY_VALUE);
        neuralNetwork.build();
        neuralNetwork.setOptimizer(OptimizationType.ADAM);
        neuralNetwork.verboseTraining(10);
        return neuralNetwork;
    }

}
