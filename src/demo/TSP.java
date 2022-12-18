/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
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
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

/**
 * Implements deep reinforcement learning solution solving travelling salesman problem.<br>
 *
 */
public class TSP implements Environment, AgentFunctionEstimator {

    /**
     * Implements tour.
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
         * Distance for shortest route as normalized distance.
         *
         */
        private double minNormalizedDistance = Double.POSITIVE_INFINITY;

        /**
         * Distance for shortest route.
         *
         */
        private double minDistance = Double.POSITIVE_INFINITY;

        /**
         * Distance for longest route found as normalized distance.
         *
         */
        private double maxNormalizedDistance = Double.NEGATIVE_INFINITY;

        /**
         * Distance for longest route found.
         *
         */
        private double maxDistance = Double.NEGATIVE_INFINITY;

        /**
         * Distance from previous tour as normalized distance.
         *
         */
        private double lastNormalizedDistance = Double.NEGATIVE_INFINITY;

        /**
         * Distance from previous tour.
         *
         */
        private double lastDistance = Double.NEGATIVE_INFINITY;

        /**
         * Constructor for tour.
         *
         * @param numberOfCities number of cities for tour.
         */
        Tour(int numberOfCities) {
            this.numberOfCities = numberOfCities;
            for (int index = 0; index < numberOfCities; index++) cities.put(index, new City(Math.random(), Math.random()));
            normalize();
        }

        /**
         * Constructor for tour.
         *
         * @param cities array of cities for tour.
         */
        Tour(City[] cities) {
            this.numberOfCities = cities.length;
            for (int index = 0; index < cities.length; index++) this.cities.put(index, cities[index]);
            normalize();
        }

        /**
         * Constructor for tour.
         *
         * @param cities number of cities in list format.
         */
        Tour(String[] cities) {
            this.numberOfCities = cities.length;
            for (int index = 0; index < cities.length; index++) {
                String[] city = cities[index].split(", ");
                double x = Double.parseDouble(city[0]);
                double y = Double.parseDouble(city[1]);
                this.cities.put(index, new City(x, y));
            }
            normalize();
        }

        /**
         * Returns size of tour.
         *
         * @return size of tour.
         */
        int size() {
            return numberOfCities;
        }

        /**
         * Normalizes coordinates of cities.
         *
         */
        void normalize() {
            double xMin = 0;
            double xMax = 0;
            double yMin = 0;
            double yMax = 0;
            boolean firstCity = true;
            for (City city : cities.values()) {
                if (firstCity) {
                    xMin = city.x;
                    xMax = city.x;
                    yMin = city.y;
                    yMax = city.y;
                    firstCity = false;
                }
                else {
                    xMin = Math.min(xMin, city.x);
                    xMax = Math.max(xMax, city.x);
                    yMin = Math.min(yMin, city.y);
                    yMax = Math.max(yMax, city.y);
                }
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
         * Add visited city to tour. Updates total tour distance.
         *
         * @param cityIndex index of visited city.
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
         * Records complete tour. Updates statistics.
         *
         */
        void record() {
            if (totalNormalizedDistance < minNormalizedDistance) {
                minNormalizedDistance = totalNormalizedDistance;
                minDistance = totalDistance;
                visitedCitiesMin = new ArrayList<>(visitedCities);
            }
            if (totalNormalizedDistance > maxNormalizedDistance) {
                maxNormalizedDistance = totalNormalizedDistance;
                maxDistance = totalDistance;
                visitedCitiesMax = new ArrayList<>(visitedCities);
            }
        }

        /**
         * Checks if tour is complete.
         *
         * @return true if tour is complete otherwise false.
         */
        boolean isCompleteTour() {
            return visitedCities.size() == cities.size();
        }

        /**
         * Returns set of unvisited cities.
         *
         * @return set of unvisited cities.
         */
        HashSet<Integer> getUnvisitedCities() {
            HashSet<Integer> unvisitedCities = new HashSet<>();
            for (Integer index : cities.keySet()) if (!visitedCities.contains(index) && index != startCity) unvisitedCities.add(index - 1);
            return unvisitedCities;
        }

    }

    /**
     * Implements city with coordinates x and y.
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
         * Normalized coordinate x of city.
         *
         */
        double xNormalized;

        /**
         * Normalized coordinate y of city.
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
            double diagonal = Math.sqrt(Math.pow(xMax - xMin, 2) + Math.pow(yMax - yMin, 2));
            xNormalized = (x - xMin) / diagonal;
            yNormalized = (y - yMin) / diagonal;
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
    private static final int numberOfCities = 15;

    /**
     * Current tour.
     *
     */
    private final Tour tour;

    /**
     * Count for how many times distance was same between tours.
     *
     */
    private int unchangedDistanceCount = 0;

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
     * @throws DynamicParamException throws exception if setting of dynamic parameter fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws IOException throws exception if cloning of neural network fails.
     * @throws ClassNotFoundException throws exception if cloning of neural network fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public TSP(int cityAmount) throws NeuralNetworkException, MatrixException, DynamicParamException, IOException, ClassNotFoundException, AgentException {
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
     * Returns route that is shortest found by deep agent (travelling salesman) as indices of cities.
     *
     * @return shortest route as indices of cities.
     */
    public ArrayList<Integer> getShortestRoute() {
        return tour.visitedCitiesMin;
    }

    /**
     * Returns route that is longest found by deep agent (travelling salesman) as indices of cities.
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
     * Main function for travelling salesman.
     *
     * @param args not used.
     */
    public static void main(String[] args) {
        TSP tsp;
        String stringFormat = "%.5f";
        try {
            tsp = new TSP(numberOfCities);
            tsp.initWindow();
            for (int tour = 0; tour < 1000000; tour++) {
                tsp.route(tour % 10 == 0);
                System.out.println("Tour #" + (tour + 1) + " Total: " + String.format(stringFormat, tsp.getTotalDistance()) + " (" + String.format(stringFormat, tsp.getTotalNormalizedDistance()) +")" + " Min: " + String.format(stringFormat, tsp.getMinDistance()) + " (" + String.format(stringFormat, tsp.getMinNormalizedDistance()) + ")" + " Max: " + String.format(stringFormat, tsp.getMaxDistance()) + " (" + String.format(stringFormat, tsp.getMaxNormalizedDistance()) + ")" + " Cumul. reward learning " + String.format(stringFormat, tsp.getCumulativeReward(true)) + " Cumul. reward non-learning " + String.format(stringFormat, tsp.getCumulativeReward(false)));
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
        agent.respond(isTerminalState() ? (getMaxNormalizedDistance() - getMinNormalizedDistance() != 0 ? Math.pow(1 - (getTotalNormalizedDistance() - getMinNormalizedDistance()) / (getMaxNormalizedDistance() - getMinNormalizedDistance()), 6) : 0) : 0);
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
         * List of cities forming shortest found route.
         *
         */
        private final ArrayList<Integer> shortestDrawCities = new ArrayList<>();

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
        void addCities(ArrayList<Integer> newDrawCities, ArrayList<Integer> newPreviousDrawCities, ArrayList<Integer> newShortestDrawCities) {
            drawCities.clear();
            drawCities.addAll(newDrawCities);
            if (newPreviousDrawCities != null) {
                previousDrawCities.clear();
                previousDrawCities.addAll(newPreviousDrawCities);
                previousExists = true;
            }
            else previousExists = false;
            if (newShortestDrawCities != null) {
                shortestDrawCities.clear();
                shortestDrawCities.addAll(newShortestDrawCities);
            }
        }

        /**
         * Paints journey to JPanel.
         *
         * @param g graphics.
         */
        public void paintComponent(Graphics g) {
            Graphics2D g2 = (Graphics2D)g;
            super.paintComponent(g2);
            if (drawCities.isEmpty()) return;
            City lastCity = null;
            int lastCityIndex = -1;
            int previousLastCityIndex = -2;
            g2.setStroke(new BasicStroke(1));
            int size = drawCities.size() - 1;
            for (int index = 0; index < size; index++) {
                City city1 = tour.cities.get(drawCities.get(index));
                City city2 = tour.cities.get(drawCities.get(index + 1));
                lastCity = city2;
                lastCityIndex = drawCities.get(index + 1);
                if (previousExists) {
                    if (drawCities.get(index).equals(previousDrawCities.get(index)) && drawCities.get(index + 1).equals(previousDrawCities.get(index + 1))) g.setColor(Color.DARK_GRAY);
                    else g2.setColor(Color.LIGHT_GRAY);
                    previousLastCityIndex = previousDrawCities.get(index + 1);
                }
                else g2.setColor(Color.DARK_GRAY);
                g2.drawLine((int)(0.1 * xWindowSize) + (int)(city1.xNormalized * (int)(0.8 * xWindowSize)), (int)(0.1 * yWindowSize) + (int)(city1.yNormalized * (int)(0.8 * yWindowSize)), (int)(0.1 * xWindowSize) + (int)(city2.xNormalized * (int)(0.8 * xWindowSize)), (int)(0.1 * yWindowSize) + (int)(city2.yNormalized * (int)(0.8 * yWindowSize)));
            }
            if (lastCity != null) {
                if (previousExists) {
                    if (lastCityIndex == previousLastCityIndex) g.setColor(Color.DARK_GRAY);
                    else g2.setColor(Color.LIGHT_GRAY);
                }
                else g2.setColor(Color.DARK_GRAY);
                City initialCity = tour.cities.get(tour.startCity);
                g2.drawLine((int)(0.1 * xWindowSize) + (int)(lastCity.xNormalized * (int)(0.8 * xWindowSize)), (int)(0.1 * yWindowSize) + (int)(lastCity.yNormalized * (int)(0.8 * yWindowSize)), (int)(0.1 * xWindowSize) + (int)(initialCity.xNormalized * (int)(0.8 * xWindowSize)), (int)(0.1 * yWindowSize) + (int)(initialCity.yNormalized * (int)(0.8 * yWindowSize)));
            }
            if (shortestDrawCities.isEmpty()) return;
            g2.setStroke(new BasicStroke(2));
            g2.setColor(Color.RED);
            lastCity = null;
            int shortestDrawCitiesSize = shortestDrawCities.size() - 1;
            for (int index = 0; index < shortestDrawCitiesSize; index++) {
                City city1 = tour.cities.get(shortestDrawCities.get(index));
                City city2 = tour.cities.get(shortestDrawCities.get(index + 1));
                lastCity = city2;
                g2.drawLine((int)(0.1 * xWindowSize) + (int)(city1.xNormalized * (int)(0.8 * xWindowSize)), (int)(0.1 * yWindowSize) + (int)(city1.yNormalized * (int)(0.8 * yWindowSize)), (int)(0.1 * xWindowSize) + (int)(city2.xNormalized * (int)(0.8 * xWindowSize)), (int)(0.1 * yWindowSize) + (int)(city2.yNormalized * (int)(0.8 * yWindowSize)));
            }
            if (lastCity != null) {
                City initialCity = tour.cities.get(tour.startCity);
                g2.drawLine((int)(0.1 * xWindowSize) + (int)(lastCity.xNormalized * (int)(0.8 * xWindowSize)), (int)(0.1 * yWindowSize) + (int)(lastCity.yNormalized * (int)(0.8 * yWindowSize)), (int)(0.1 * xWindowSize) + (int)(initialCity.xNormalized * (int)(0.8 * xWindowSize)), (int)(0.1 * yWindowSize) + (int)(initialCity.yNormalized * (int)(0.8 * yWindowSize)));
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
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    private void route(boolean redraw) throws MatrixException, NeuralNetworkException, DynamicParamException, AgentException, IOException, ClassNotFoundException {
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

        if (tour.lastDistance > 0 && tour.lastDistance == getTotalDistance()) unchangedDistanceCount++;
        else unchangedDistanceCount = 0;

        if (redraw) {
            jFrame.remove(tspPanel);
            tspPanel = new TSPPanel();
            jFrame.add(tspPanel);
            tspPanel.addCities(tour.visitedCities, tour.visitedCitiesPrevious, tour.visitedCitiesMin);
            jFrame.revalidate();
            tspPanel.paintImmediately(0, 0, (int)(0.8 * xWindowSize), (int)(0.8 * yWindowSize));
        }
    }

    /**
     * Returns cumulative reward.
     *
     * @param isLearning if true returns cumulative reward during learning otherwise return cumulative reward when not learning
     * @return cumulative reward.
     */
    public double getCumulativeReward(boolean isLearning) {
        return getAgent().getCumulativeReward(isLearning);
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
     * @param inputSize number of inputs for agent.
     * @param outputSize number of outputs for agent.
     * @return agent
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException throws exception if setting of dynamic parameter fails.
     * @throws IOException throws exception if cloning of neural network fails.
     * @throws ClassNotFoundException throws exception if cloning of neural network fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    private Agent createAgent(int inputSize, int outputSize) throws MatrixException, NeuralNetworkException, DynamicParamException, IOException, ClassNotFoundException, AgentException {
        int policyType = 0;
        ExecutablePolicyType executablePolicyType = null;
        String policyTypeParams = "";
        switch (policyType) {
            case 0 -> executablePolicyType = ExecutablePolicyType.GREEDY;
            case 1 -> {
                executablePolicyType = ExecutablePolicyType.EPSILON_GREEDY;
                policyTypeParams = "epsilonInitial = 0.4, epsilonDecayRate = 0.999, epsilonMin = 0.00";
            }
            case 2 -> {
                executablePolicyType = ExecutablePolicyType.NOISY_NEXT_BEST;
                policyTypeParams = "initialExplorationNoise = 0.2, explorationNoiseDecay = 0.999, minExplorationNoise = 0";
            }
            case 3 -> {
                executablePolicyType = ExecutablePolicyType.SAMPLED;
                policyTypeParams = "thresholdInitial = 0.7, thresholdMin = 0.0";
            }
            case 4 -> executablePolicyType = ExecutablePolicyType.ENTROPY_GREEDY;
            case 5 -> executablePolicyType = ExecutablePolicyType.ENTROPY_NOISY_NEXT_BEST;
        }
        boolean singleFunctionEstimator = false;
        AgentFactory.AgentAlgorithmType agentAlgorithmType = AgentFactory.AgentAlgorithmType.MCTS;
        boolean onlineMemory = switch (agentAlgorithmType) {
            case DDQN, DDPG, SACDiscrete -> false;
            default -> true;
        };
        boolean applyDueling = switch (agentAlgorithmType) {
            case DQN, DDQN -> true;
            default -> false;
        };
        String algorithmParams = switch (agentAlgorithmType) {
            case QN -> "agentUpdateCycle = 10";
            case DDQN -> "applyImportanceSamplingWeights = true, applyUniformSampling = false, capacity = 20000, targetFunctionUpdateCycle = 0, targetFunctionTau = 0.01";
            case SACDiscrete -> "applyImportanceSamplingWeights = false, applyUniformSampling = true, capacity = 20000, targetFunctionUpdateCycle = 0, targetFunctionTau = 0.01";
            case MCTS -> "gamma = 1, updateValuePerEpisode = true";
            default -> "";
        };

        String params = "";
        if (policyTypeParams.isEmpty() && !algorithmParams.isEmpty()) params = algorithmParams;
        if (!policyTypeParams.isEmpty() && algorithmParams.isEmpty()) params = policyTypeParams;
        if (!policyTypeParams.isEmpty() && !algorithmParams.isEmpty()) params = policyTypeParams + ", " + algorithmParams;

        Agent agent = AgentFactory.createAgent(this, agentAlgorithmType, this, inputSize, outputSize, onlineMemory, singleFunctionEstimator, applyDueling, executablePolicyType, params);
        agent.start();
        return agent;
    }

    /**
     * Build neural network for travelling salesman (agent).
     *
     * @param inputSize input size of neural network (number of states)
     * @param outputSize output size of neural network (number of actions and their values).
     * @param policyGradient if true neural network is policy gradient network.
     * @param applyDueling if true applied dueling layer to non policy gradient network otherwise not.
     * @return built neural network
     * @throws DynamicParamException throws exception if setting of dynamic parameters fails.
     * @throws NeuralNetworkException throws exception if building of neural network fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public NeuralNetwork buildNeuralNetwork(int inputSize, int outputSize, boolean policyGradient, boolean applyDueling) throws DynamicParamException, NeuralNetworkException, MatrixException {
        NeuralNetwork neuralNetwork = new NeuralNetwork();

        NeuralNetworkConfiguration neuralNetworkConfiguration = new NeuralNetworkConfiguration();
        neuralNetworkConfiguration.addInputLayer("width = " + inputSize);
        String width = "width = " + (4 * inputSize);
        neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), width);
        neuralNetworkConfiguration.addHiddenLayer(LayerType.CONNECTOR, "joinInputs = false");
        neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), width);
        neuralNetworkConfiguration.addHiddenLayer(LayerType.CONNECTOR, "joinInputs = false");
        neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, !policyGradient ? new ActivationFunction(UnaryFunctionType.SINACT) : new ActivationFunction(UnaryFunctionType.RELU), "width = " + outputSize);
        if (!policyGradient && applyDueling) neuralNetworkConfiguration.addHiddenLayer(LayerType.DUELING, "width = " + outputSize);
        neuralNetworkConfiguration.addOutputLayer(!policyGradient ? BinaryFunctionType.MEAN_SQUARED_ERROR : BinaryFunctionType.DIRECT_GRADIENT);
        neuralNetworkConfiguration.connectLayersSerially();
        neuralNetworkConfiguration.connectLayers(0, 2);
        neuralNetworkConfiguration.connectLayers(0, 4);
        neuralNetworkConfiguration.connectLayers(1, 4);

        neuralNetwork.build(neuralNetworkConfiguration);

        neuralNetwork.setOptimizer(OptimizationType.ADAM);
        if (!policyGradient) neuralNetwork.verboseTraining(10);
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
    public NeuralNetwork buildNeuralNetwork(int inputSize, int outputSize) throws DynamicParamException, NeuralNetworkException, MatrixException {
        NeuralNetwork neuralNetwork = new NeuralNetwork();

        NeuralNetworkConfiguration neuralNetworkConfiguration = new NeuralNetworkConfiguration();
        neuralNetworkConfiguration.addInputLayer("width = " + inputSize);
        String width = "width = " + (6 * inputSize);
        neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), width);
        neuralNetworkConfiguration.addHiddenLayer(LayerType.CONNECTOR, "joinInputs = false");
        neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), width);
        neuralNetworkConfiguration.addHiddenLayer(LayerType.CONNECTOR, "joinInputs = false");
        neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.RELU), "width = " + outputSize);
        neuralNetworkConfiguration.addOutputLayer(BinaryFunctionType.DIRECT_GRADIENT);
        neuralNetworkConfiguration.connectLayersSerially();
        neuralNetworkConfiguration.connectLayers(0, 2);
        neuralNetworkConfiguration.connectLayers(0, 4);
        neuralNetworkConfiguration.connectLayers(1, 4);
        neuralNetworkConfiguration.addHiddenLayer(LayerType.FEEDFORWARD, new ActivationFunction(UnaryFunctionType.SINACT), "width = 1");
        neuralNetworkConfiguration.addOutputLayer(BinaryFunctionType.MEAN_SQUARED_ERROR);
        neuralNetworkConfiguration.connectLayers(4, 7);
        neuralNetworkConfiguration.connectLayers(7, 8);

        neuralNetwork.build(neuralNetworkConfiguration);

        neuralNetwork.setOptimizer(OptimizationType.RADAM);
        neuralNetwork.verboseTraining(10);
        return neuralNetwork;
    }

}
