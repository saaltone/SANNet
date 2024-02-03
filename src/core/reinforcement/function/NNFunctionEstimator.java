/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.function;

import core.layer.InputLayer;
import core.network.NeuralNetwork;
import core.network.NeuralNetworkException;
import core.reinforcement.agent.State;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.Initialization;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.sampling.BasicSampler;

import java.io.IOException;
import java.util.*;

/**
 * Implements neural network based function estimator.<br>
 *
 */
public class NNFunctionEstimator extends AbstractFunctionEstimator {

    /**
     * Parameter name types for neural network based function estimator.
     *     - numberOfIterations: number of training or validation iterations executed during step. Default value 1.<br>
     *     - targetFunctionUpdateCycle; target function update cycle. Default value 0 (smooth update).<br>
     *     - targetFunctionTau: update rate of target function. Default value 0.001.<br>
     *
     */
    private final static String paramNameTypes = "(numberOfIterations:INT), " +
            "(targetFunctionUpdateCycle:INT), " +
            "(targetFunctionTau:DOUBLE)";

    /**
     * Neural network function estimator.
     *
     */
    private final NeuralNetwork neuralNetwork;

    /**
     * Target neural network function estimator.
     *
     */
    private final NeuralNetwork targetNeuralNetwork;

    /**
     * Number of validation cycles.
     *
     */
    private int numberOfIterations;

    /**
     * Update cycle (in episodes) for target function estimator. If update cycle is zero then smooth parameter updates are applied with update rate tau.
     *
     */
    private int targetFunctionUpdateCycle;

    /**
     * Update count for target function update cycle.
     *
     */
    private transient int targetFunctionUpdateCount = 0;

    /**
     * Update rate of target function.
     *
     */
    private double targetFunctionTau;

    /**
     * Size of state history.
     *
     */
    private final int stateHistorySize;

    /**
     * Size of action history.
     *
     */
    private final int actionHistorySize;

    /**
     * Zero state input for empty history entries.
     *
     */
    private final Matrix zeroStateInputReference;

    /**
     * Zero action input for empty history entries.
     *
     */
    private final Matrix zeroActionInputReference;

    /**
     * Intermediate map for state value pairs as value cache.
     *
     */
    private final HashMap<State, TreeMap<Integer, Matrix>> stateCache = new HashMap<>();

    /**
     * Intermediate map for state value pairs for update.
     *
     */
    private final TreeMap<State, TreeMap<Integer, Matrix>> stateValueMap = new TreeMap<>();

    /**
     * Constructor for neural network based function estimator.
     *
     * @param neuralNetwork              neural network reference.
     * @param hasTargetFunctionEstimator if true has target function estimator otherwise not
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public NNFunctionEstimator(NeuralNetwork neuralNetwork, boolean hasTargetFunctionEstimator) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException {
        this (neuralNetwork, hasTargetFunctionEstimator, null);
    }

    /**
     * Constructor for neural network based function estimator.
     *
     * @param neuralNetwork              neural network reference.
     * @param hasTargetFunctionEstimator if true has target function estimator otherwise not
     * @param params                     parameters for function
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public NNFunctionEstimator(NeuralNetwork neuralNetwork, boolean hasTargetFunctionEstimator, String params) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException {
        super (neuralNetwork.getInputLayerGroups().get(0).get(0).getLayerWidth(), neuralNetwork.getOutputLayers().get(0).getLayerWidth(), neuralNetwork.getOutputLayers().size() == 2, params);
        this.neuralNetwork = neuralNetwork;
        this.targetNeuralNetwork = hasTargetFunctionEstimator ? neuralNetwork.copy() : null;

        TreeMap<Integer, InputLayer> stateInputLayers = neuralNetwork.getInputLayerGroups().get(0);
        stateHistorySize = stateInputLayers.size();
        int firstStateKey = stateInputLayers.firstKey();
        zeroStateInputReference = new DMatrix(stateInputLayers.get(firstStateKey).getLayerWidth(), stateInputLayers.get(firstStateKey).getLayerHeight(), stateInputLayers.get(firstStateKey).getLayerDepth(), Initialization.ONE);

        TreeMap<Integer, InputLayer> actionInputLayers = neuralNetwork.getInputLayerGroups().get(1);
        if (actionInputLayers != null) {
            actionHistorySize = actionInputLayers.size();
            int firstActionKey = actionInputLayers.firstKey();
            zeroActionInputReference = new DMatrix(actionInputLayers.get(firstActionKey).getLayerWidth(), actionInputLayers.get(firstActionKey).getLayerHeight(), actionInputLayers.get(firstActionKey).getLayerDepth(), Initialization.ONE);
        }
        else {
            actionHistorySize = 0;
            zeroActionInputReference = null;
        }
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        numberOfIterations = 1;
        targetFunctionUpdateCycle = 0;
        targetFunctionTau = 0.001;
    }

    /**
     * Returns parameters used for neural network based function estimator.
     *
     * @return parameters used for neural network based function estimator.
     */
    public String getParamDefs() {
        return NNFunctionEstimator.paramNameTypes;
    }

    /**
     * Sets parameters used for neural network based function estimator.<br>
     * <br>
     * Supported parameters are:<br>
     *     - numberOfIterations: number of training or validation iterations executed during step. Default value 1.<br>
     *     - targetFunctionUpdateCycle; target function update cycle. Default value 0 (smooth update).<br>
     *     - targetFunctionTau: update rate of target function. Default value 0.001.<br>
     *
     * @param params parameters used for neural network based function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("numberOfIterations")) {
            numberOfIterations = params.getValueAsInteger("numberOfIterations");
            if (numberOfIterations < 1) throw new DynamicParamException("Number of iterations must be at least 1.");
        }
        if (params.hasParam("targetFunctionUpdateCycle")) targetFunctionUpdateCycle = params.getValueAsInteger("targetFunctionUpdateCycle");
        if (params.hasParam("targetFunctionTau")) targetFunctionTau = params.getValueAsDouble("targetFunctionTau");
    }

    /**
     * Returns reference to function estimator.
     *
     * @return reference to value function.
     * @throws IOException throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public FunctionEstimator reference() throws DynamicParamException, MatrixException, IOException, ClassNotFoundException {
        return new NNFunctionEstimator(getNeuralNetwork().reference(), getTargetNeuralNetwork() != null, getParams());
    }

    /**
     * Returns copy of neural network based function estimator.
     *
     * @return copy of neural network based function estimator.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public FunctionEstimator copy() throws IOException, ClassNotFoundException, DynamicParamException, MatrixException {
        return new NNFunctionEstimator(getNeuralNetwork().copy(), getTargetNeuralNetwork() != null, getParams());
    }

    /**
     * Starts function estimator.
     *
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void start() throws NeuralNetworkException, MatrixException, DynamicParamException {
        if (!neuralNetwork.isStarted()) neuralNetwork.start();
        if (targetNeuralNetwork != null) {
            if (!targetNeuralNetwork.isStarted()) targetNeuralNetwork.start();
        }
    }

    /**
     * Stops function estimator.
     *
     */
    public void stop() {
        if (neuralNetwork.isStarted()) neuralNetwork.stop();
        if (targetNeuralNetwork != null) if (!targetNeuralNetwork.isStarted()) targetNeuralNetwork.stop();
    }

    /**
     * Returns neural network used by neural network based function estimator.
     *
     * @return neural network.
     */
    public NeuralNetwork getNeuralNetwork() {
        return neuralNetwork;
    }

    /**
     * Returns target neural network used by neural network based function estimator.
     *
     * @return neural network.
     */
    public NeuralNetwork getTargetNeuralNetwork() {
        return targetNeuralNetwork;
    }

    /**
     * Resets neural network based function estimator.
     *
     */
    protected void reset() {
        stateCache.clear();
        stateValueMap.clear();
    }

    /**
     * Predicts values corresponding to a state.
     *
     * @param state state.
     * @return values corresponding to a state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private TreeMap<Integer, Matrix> predictValues(NeuralNetwork currentNeuralNetwork, State state) throws NeuralNetworkException, MatrixException {
        if (currentNeuralNetwork == getNeuralNetwork()) {
            TreeMap<Integer, Matrix> values = stateCache.get(state);
            if (values == null)  {
                values = currentNeuralNetwork.predictMatrix(new TreeMap<>() {{ putAll(getInputs(state)); }});
                stateCache.put(state, values);
            }
            return values;
        }
        else {
            return currentNeuralNetwork.predictMatrix(new TreeMap<>() {{ putAll(getInputs(state)); }});
        }
    }

    /**
     * Returns inputs based on state
     *
     * @param state state
     * @return inputs
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private TreeMap<Integer, Matrix> getInputs(State state) throws MatrixException {
        TreeMap<Integer, Matrix> inputs = new TreeMap<>();

        int stateInputIndex = 0;
        int actionInputIndex = 0;
        State currentState = state;
        while (currentState != null) {
            if (stateInputIndex < stateHistorySize) inputs.put(stateInputIndex++, currentState.environmentState.state());
            if (actionInputIndex < actionHistorySize) inputs.put(stateHistorySize + actionInputIndex++, state.action < getNumberOfActions() ? DMatrix.getOneHotVector(getNumberOfActions(), state.action) : new DMatrix(state.tdTarget));
            currentState = currentState.previousState;
            if (stateInputIndex >= stateHistorySize && actionInputIndex >= actionHistorySize) break;
        }

        for (int paddedStateInputIndex = stateInputIndex; paddedStateInputIndex < stateHistorySize; paddedStateInputIndex++) {
            inputs.put(paddedStateInputIndex, zeroStateInputReference);
        }
        if (actionHistorySize > 0) {
            for (int paddedActionInputIndex = stateInputIndex; paddedActionInputIndex < stateHistorySize; paddedActionInputIndex++) {
                inputs.put(stateHistorySize + paddedActionInputIndex, zeroActionInputReference);
            }
        }

        return inputs;
    }

    /**
     * Predicts state values corresponding to a state.
     *
     * @param state state.
     * @return state values corresponding to a state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     */
    public Matrix predictPolicyValues(State state) throws NeuralNetworkException, MatrixException {
        return predictValues(getNeuralNetwork(), state).get(0);
    }

    /**
     * Predicts target policy values corresponding to a state.
     *
     * @param state state.
     * @return policy values corresponding to a state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     */
    public Matrix predictTargetPolicyValues(State state) throws NeuralNetworkException, MatrixException {
        return predictValues(getTargetNeuralNetwork(), state).get(0);
    }

    /**
     * Predicts state action values corresponding to a state.
     *
     * @param state state.
     * @return state action values corresponding to a state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix predictStateActionValues(State state) throws NeuralNetworkException, MatrixException {
        return predictValues(getNeuralNetwork(), state).get(isStateActionValueFunction() ? 1 : 0);
    }

    /**
     * Predicts target state action values corresponding to a state.
     *
     * @param state state.
     * @return state action values corresponding to a state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix predictTargetStateActionValues(State state) throws NeuralNetworkException, MatrixException {
        return predictValues(getTargetNeuralNetwork(), state).get(isStateActionValueFunction() ? 1 : 0);
    }

    /**
     * Stores values.
     *
     * @param state state.
     * @param values values.
     * @param outputIndex is 1 in case of state action values otherwise 0 (policy value / action values).
     */
    private void storeValues(State state, Matrix values, int outputIndex) {
        if (!stateValueMap.containsKey(state)) stateValueMap.put(state, new TreeMap<>());
        stateValueMap.get(state).put(outputIndex, values);
    }

    /**
     * Stores policy state values pair.
     * @param state state.
     * @param values values.
     */
    public void storePolicyValues(State state, Matrix values) {
        storeValues(state, values, 0);
    }

    /**
     * Stores state action values pair.
     *
     * @param state state.
     * @param values values.
     */
    public void storeStateActionValues(State state, Matrix values) {
        storeValues(state, values, isStateActionValueFunction() ? 1 : 0);
    }

    /**
     * Updates (trains) neural network.
     *
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     */
    public void update() throws NeuralNetworkException, DynamicParamException, MatrixException {
        HashMap<Integer, HashMap<Integer, Matrix>> inputs = new HashMap<>();
        HashMap<Integer, HashMap<Integer, Matrix>> outputs = new HashMap<>();

        for (int inputIndex = 0; inputIndex < stateHistorySize; inputIndex++) inputs.put(inputIndex, new HashMap<>());
        if (actionHistorySize > 0) for (int inputIndex = 0; inputIndex < actionHistorySize; inputIndex++) inputs.put(stateHistorySize + inputIndex, new HashMap<>());

        outputs.put(0, new HashMap<>());
        if (isStateActionValueFunction()) outputs.put(1, new HashMap<>());

        HashMap<Integer, Double> importanceSamplingWeights = new HashMap<>();

        int index = 0;
        boolean applyImportanceSamplingWeights = false;
        for (Map.Entry<State, TreeMap<Integer, Matrix>> entry: stateValueMap.entrySet()) {
            State state = entry.getKey();

            TreeMap<Integer, Matrix> currentInputs = getInputs(state);
            for (Map.Entry<Integer, Matrix> inputEntry : currentInputs.entrySet()) inputs.get(inputEntry.getKey()).put(index, inputEntry.getValue());

            TreeMap<Integer, Matrix> currentOutputs = entry.getValue();
            for (Map.Entry<Integer, Matrix> outputEntry : currentOutputs.entrySet()) outputs.get(outputEntry.getKey()).put(index, outputEntry.getValue());

            if (state.applyImportanceSamplingWeight) {
                importanceSamplingWeights.put(index, state.importanceSamplingWeight);
                applyImportanceSamplingWeights = true;
            }

            index++;
        }

        if (applyImportanceSamplingWeights) neuralNetwork.setImportanceSamplingWeights(new TreeMap<>() {{ put(0, importanceSamplingWeights); }});
        neuralNetwork.train(new BasicSampler(new HashMap<>() {{ putAll(inputs); }}, new HashMap<>() {{ putAll(outputs); }}, "fullSet = true, randomOrder = false, numberOfIterations = " + numberOfIterations));

        if (getTargetNeuralNetwork() != null) appendTargetNeuralNetwork();

        updateComplete();
    }

    /**
     * Appends parameters to this neural network based function estimator from another neural network based function estimator.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private void appendTargetNeuralNetwork() throws MatrixException {
        if (targetFunctionUpdateCycle == 0) getTargetNeuralNetwork().append(neuralNetwork, targetFunctionTau);
        else {
            if (++targetFunctionUpdateCount >= targetFunctionUpdateCycle) {
                getTargetNeuralNetwork().append(neuralNetwork, 1);
                targetFunctionUpdateCount = 0;
            }
        }
    }

    /**
     * Appends from function estimator.
     *
     * @param functionEstimator function estimator.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void append(FunctionEstimator functionEstimator) throws MatrixException {
        getNeuralNetwork().append(((NNFunctionEstimator)functionEstimator).getNeuralNetwork(), 1);
    }

}
