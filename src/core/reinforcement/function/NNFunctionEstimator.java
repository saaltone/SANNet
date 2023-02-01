/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.function;

import core.network.NeuralNetwork;
import core.network.NeuralNetworkException;
import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.StateTransition;
import core.reinforcement.memory.Memory;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.MMatrix;
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
     *     - targetFunctionTau: update rate of target function. Default value 0.01.<br>
     *
     */
    private final static String paramNameTypes = "(numberOfIterations:INT), " +
            "(targetFunctionTau:DOUBLE)";

    /**
     * Neural network function estimator.
     *
     */
    private final NeuralNetwork neuralNetwork;

    /**
     * Number of validation cycles.
     *
     */
    private int numberOfIterations;

    /**
     * Update rate of target function.
     *
     */
    private double targetFunctionTau;

    /**
     * If true applies importance sampling weights.
     *
     */
    private boolean applyImportanceSamplingWeights;

    /**
     * Size of state history.
     *
     */
    private final int stateHistorySize;

    /**
     * Zero input for empty history entries.
     *
     */
    private final Matrix zeroInputReference;

    /**
     * Intermediate map for state value pairs as value cache.
     *
     */
    private final HashMap<StateTransition, TreeMap<Integer, Matrix>> stateTransitionCache = new HashMap<>();

    /**
     * Intermediate map for state transition value pairs for update.
     *
     */
    private final TreeMap<StateTransition, TreeMap<Integer, Matrix>> stateTransitionValueMap = new TreeMap<>();

    /**
     * Constructor for neural network based function estimator.
     *
     * @param memory memory reference.
     * @param neuralNetwork neural network reference.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public NNFunctionEstimator(Memory memory, NeuralNetwork neuralNetwork) throws DynamicParamException {
        this (memory, neuralNetwork, null);
    }

    /**
     * Constructor for neural network based function estimator.
     *
     * @param memory memory reference.
     * @param neuralNetwork neural network reference.
     * @param params parameters for function
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public NNFunctionEstimator(Memory memory, NeuralNetwork neuralNetwork, String params) throws DynamicParamException {
        super (memory, neuralNetwork.getInputLayers().get(0).getLayerWidth(), neuralNetwork.getOutputLayers().get(0).getLayerWidth(), neuralNetwork.getOutputLayers().size() == 2, params);
        this.neuralNetwork = neuralNetwork;
        stateHistorySize = neuralNetwork.getInputLayers().size();
        zeroInputReference = new DMatrix(neuralNetwork.getInputLayers().get(0).getLayerWidth(), 1);
        applyImportanceSamplingWeights = memory.applyImportanceSamplingWeights();
    }

    /**
     * Initializes default params.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void initializeDefaultParams() throws DynamicParamException {
        super.initializeDefaultParams();
        numberOfIterations = 1;
        targetFunctionTau = 0.01;
    }

    /**
     * Returns parameters used for neural network based function estimator.
     *
     * @return parameters used for neural network based function estimator.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + NNFunctionEstimator.paramNameTypes;
    }

    /**
     * Sets parameters used for neural network based function estimator.<br>
     * <br>
     * Supported parameters are:<br>
     *     - numberOfIterations: number of training or validation iterations executed during step. Default value 1.<br>
     *     - targetFunctionTau: update rate of target function. Default value 0.01.<br>
     *
     * @param params parameters used for neural network based function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
        if (params.hasParam("numberOfIterations")) {
            numberOfIterations = params.getValueAsInteger("numberOfIterations");
            if (numberOfIterations < 1) throw new DynamicParamException("Number of iterations must be at least 1.");
        }
        if (params.hasParam("targetFunctionTau")) targetFunctionTau = params.getValueAsDouble("targetFunctionTau");
        applyImportanceSamplingWeights = memory.applyImportanceSamplingWeights();
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
        return new NNFunctionEstimator(getMemory().reference(), getNeuralNetwork().reference(), getParams());
    }

    /**
     * Returns reference to function estimator.
     *
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to value function.
     * @throws IOException throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public FunctionEstimator reference(boolean sharedMemory) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException {
        return new NNFunctionEstimator(sharedMemory ? getMemory() : getMemory().reference(), getNeuralNetwork().reference(), getParams());
    }

    /**
     * Returns reference to function estimator.
     *
     * @param memory reference to memory.
     * @return reference to value function.
     * @throws IOException throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public FunctionEstimator reference(Memory memory) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException {
        return new NNFunctionEstimator(memory, getNeuralNetwork().reference(), getParams());
    }

    /**
     * Starts function estimator.
     *
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void start() throws NeuralNetworkException, MatrixException, DynamicParamException, IOException, ClassNotFoundException {
        super.start();
        if (!neuralNetwork.isStarted()) neuralNetwork.start();
    }

    /**
     * Stops function estimator.
     *
     */
    public void stop() {
        super.stop();
        if (neuralNetwork.isStarted()) neuralNetwork.stop();
    }

    /**
     * Checks if function estimator is started.
     *
     * @return true if function estimator is started otherwise false.
     */
    public boolean isStarted() {
        return neuralNetwork.isStarted();
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
     * Returns copy of neural network based function estimator.
     *
     * @return copy of neural network based function estimator.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public FunctionEstimator copy() throws IOException, ClassNotFoundException, DynamicParamException, MatrixException {
        return new NNFunctionEstimator(memory, neuralNetwork.copy(), getParams());
    }

    /**
     * Resets neural network based function estimator.
     *
     */
    public void reset() {
        super.reset();
        stateTransitionCache.clear();
        stateTransitionValueMap.clear();
        if (getTargetFunctionEstimator() != null) getTargetFunctionEstimator().reset();
    }

    /**
     * Reinitializes neural network based function estimator.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void reinitialize() throws MatrixException, DynamicParamException {
        neuralNetwork.reinitialize();
    }

    /**
     * Predicts values corresponding to a state.
     *
     * @param stateTransition state.
     * @return values corresponding to a state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     */
    private TreeMap<Integer, Matrix> predictValues(StateTransition stateTransition) throws NeuralNetworkException, MatrixException {
        TreeMap<Integer, Matrix> values = stateTransitionCache.get(stateTransition);
        if (values == null)  {
            values = neuralNetwork.predictMatrix(new TreeMap<>() {{ putAll(getInputs(stateTransition)); }});
            stateTransitionCache.put(stateTransition, values);
        }
        return values;
    }

    private TreeMap<Integer, Matrix> getInputs(StateTransition stateTransition) {
        TreeMap<Integer, Matrix> states = new TreeMap<>();
        StateTransition currentStateTransition = stateTransition;
        for (int inputIndex = stateHistorySize - 1; inputIndex >= 0; inputIndex--) {
            if (currentStateTransition != null) {
                states.put(inputIndex, currentStateTransition.environmentState.state());
                currentStateTransition = currentStateTransition.previousStateTransition;
            }
            else states.put(inputIndex, zeroInputReference);
        }
        return states;
    }

    /**
     * Predicts state values corresponding to a state.
     *
     * @param stateTransition state.
     * @param isAction true if prediction is for taking other otherwise false.
     * @return state values corresponding to a state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     */
    public Matrix predictPolicyValues(StateTransition stateTransition, boolean isAction) throws NeuralNetworkException, MatrixException {
        return predictValues(stateTransition).get(0);
    }

    /**
     * Predicts state action values corresponding to a state.
     *
     * @param stateTransition state.
     * @return state action values corresponding to a state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix predictStateActionValues(StateTransition stateTransition) throws NeuralNetworkException, MatrixException {
        return predictValues(stateTransition).get(!isStateActionValueFunction() ? 0 : 1);
    }

    /**
     * Stores policy state transition values pair.
     * @param stateTransition state transition.
     * @param values values.
     */
    public void storePolicyValues(StateTransition stateTransition, Matrix values) {
        if (!stateTransitionValueMap.containsKey(stateTransition)) stateTransitionValueMap.put(stateTransition, new TreeMap<>());
        stateTransitionValueMap.get(stateTransition).put(0, values);
    }

    /**
     * Stores state action state transition values pair.
     *
     * @param stateTransition state transition.
     * @param values values.
     */
    public void storeStateActionValues(StateTransition stateTransition, Matrix values) {
        if (!isStateActionValueFunction()) storePolicyValues(stateTransition, values);
        else {
            if (!stateTransitionValueMap.containsKey(stateTransition)) stateTransitionValueMap.put(stateTransition, new TreeMap<>());
            stateTransitionValueMap.get(stateTransition).put(1, values);
        }
    }

    /**
     * Updates (trains) neural network.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    public void update() throws NeuralNetworkException, DynamicParamException, AgentException, MatrixException, IOException, ClassNotFoundException {
        HashMap<Integer, HashMap<Integer, MMatrix>> states = new HashMap<>();
        HashMap<Integer, HashMap<Integer, MMatrix>> stateValues = new HashMap<>();
        ArrayDeque<Matrix> stateHistory = new ArrayDeque<>();
        for (int inputIndex = 0; inputIndex < stateHistorySize; inputIndex++) {
            states.put(inputIndex, new HashMap<>());
            stateHistory.add(new DMatrix(neuralNetwork.getInputLayers().get(inputIndex).getLayerWidth(), 1));
        }
        stateValues.put(0, new HashMap<>());
        if (isStateActionValueFunction()) stateValues.put(1, new HashMap<>());
        HashMap<Integer, Double> importanceSamplingWeights = new HashMap<>();
        int index = 0;
        for (Map.Entry<StateTransition, TreeMap<Integer, Matrix>> entry: stateTransitionValueMap.entrySet()) {
            StateTransition stateTransition = entry.getKey();
            stateHistory.add(stateTransition.environmentState.state());
            stateHistory.poll();
            int inputIndex = 0;
            for (Matrix state : stateHistory) states.get(inputIndex++).put(index, new MMatrix(state));
            TreeMap<Integer, Matrix> matrix = entry.getValue();
            for (Map.Entry<Integer, Matrix> entry1 : matrix.entrySet()) {
                stateValues.get(entry1.getKey()).put(index, new MMatrix(entry1.getValue()));
            }
            if (applyImportanceSamplingWeights) importanceSamplingWeights.put(index, stateTransition.importanceSamplingWeight);
            index++;
        }
        if (applyImportanceSamplingWeights) neuralNetwork.setImportanceSamplingWeights(new TreeMap<>() {{ put(0, importanceSamplingWeights); }});
        neuralNetwork.train(new BasicSampler(new HashMap<>() {{ putAll(states); }}, new HashMap<>() {{ putAll(stateValues); }}, "fullSet = true, randomOrder = false, numberOfIterations = " + numberOfIterations));

        updateComplete();
    }

    /**
     * Appends parameters to this neural network based function estimator from another neural network based function estimator.
     *
     * @param functionEstimator function estimator used to update current function estimator.
     * @param fullUpdate if true full update is done.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    public void append(FunctionEstimator functionEstimator, boolean fullUpdate) throws MatrixException, AgentException {
        super.append();
        neuralNetwork.append(((NNFunctionEstimator)functionEstimator).getNeuralNetwork(), fullUpdate ? 1 : targetFunctionTau);
        if (fullUpdate) neuralNetwork.append(((NNFunctionEstimator)functionEstimator).getNeuralNetwork(), 1);
        else neuralNetwork.append(((NNFunctionEstimator)functionEstimator).getNeuralNetwork(), targetFunctionTau);
    }

    /**
     * Appends parameters to this function estimator from another function estimator.
     *
     * @param functionEstimator function estimator used to update current function estimator.
     * @param tau tau which controls contribution of other function estimator.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    public void append(FunctionEstimator functionEstimator, double tau) throws MatrixException, AgentException, NeuralNetworkException, IOException, DynamicParamException, ClassNotFoundException {
        super.append();
        neuralNetwork.append(((NNFunctionEstimator)functionEstimator).getNeuralNetwork(), tau);
        finalizeAppend();
    }

    /**
     * Sets if importance sampling weights are applied.
     *
     * @param applyImportanceSamplingWeights if true importance sampling weights are applied otherwise not.
     */
    public void setEnableImportanceSamplingWeights(boolean applyImportanceSamplingWeights) {
        this.applyImportanceSamplingWeights = applyImportanceSamplingWeights;
    }

}
