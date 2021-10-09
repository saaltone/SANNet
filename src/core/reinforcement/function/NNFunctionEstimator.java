/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.function;

import core.network.NeuralNetwork;
import core.network.NeuralNetworkException;
import core.reinforcement.agent.AgentException;
import core.reinforcement.memory.Memory;
import core.reinforcement.memory.StateTransition;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.sampling.BasicSampler;

import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.TreeMap;

/**
 * Class that defines NNFunctionEstimator.<br>
 *
 */
public class NNFunctionEstimator extends AbstractFunctionEstimator {

    /**
     * Parameter name types for NNFunctionEstimator.
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
    private int numberOfIterations = 1;

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
     * Intermediate map for state value pairs as value cache.
     *
     */
    private HashMap<Matrix, Matrix> stateValueMap = new HashMap<>();

    /**
     * Intermediate map for state transition value pairs for update.
     *
     */
    private HashMap<StateTransition, Matrix> stateTransitionValueMap = new HashMap<>();

    /**
     * Parameters for NNFunctionEstimator.
     *
     */
    private String params;

    /**
     * Constructor for NNFunctionEstimator.
     *
     * @param memory memory reference.
     * @param neuralNetwork neural network reference.
     */
    public NNFunctionEstimator(Memory memory, NeuralNetwork neuralNetwork) {
        super (memory, neuralNetwork.getInputLayer().getLayerWidth(), neuralNetwork.getOutputLayer().isMultiOutput() ? neuralNetwork.getOutputLayer().getLayerWidth() - 1 : neuralNetwork.getOutputLayer().getLayerWidth(), neuralNetwork.getOutputLayer().isMultiOutput());
        this.neuralNetwork = neuralNetwork;
        applyImportanceSamplingWeights = memory.applyImportanceSamplingWeights();
    }

    /**
     * Constructor for NNFunctionEstimator.
     *
     * @param memory memory reference.
     * @param neuralNetwork neural network reference.
     * @param params parameters for function
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public NNFunctionEstimator(Memory memory, NeuralNetwork neuralNetwork, String params) throws DynamicParamException {
        this(memory, neuralNetwork);
        this.params = params;
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        targetFunctionTau = 0.01;
    }

    /**
     * Returns parameters used for NNFunctionEstimator.
     *
     * @return parameters used for NNFunctionEstimator.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + NNFunctionEstimator.paramNameTypes;
    }

    /**
     * Sets parameters used for NNFunctionEstimator.<br>
     * <br>
     * Supported parameters are:<br>
     *     - numberOfIterations: number of training or validation iterations executed during step. Default value 1.<br>
     *     - targetFunctionTau: update rate of target function. Default value 0.01.<br>
     *
     * @param params parameters used for NNFunctionEstimator.
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
     */
    public FunctionEstimator reference() {
        return new NNFunctionEstimator(getMemory(), getNeuralNetwork());
    }

    /**
     * Returns reference to function estimator.
     *
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public FunctionEstimator reference(boolean sharedMemory) throws DynamicParamException {
        return new NNFunctionEstimator(sharedMemory ? getMemory() : getMemory().reference(), getNeuralNetwork());
    }

    /**
     * Starts function estimator.
     *
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void start() throws NeuralNetworkException, MatrixException, DynamicParamException {
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
     * Returns neural network used by NNFunctionEstimator.
     *
     * @return neural network.
     */
    public NeuralNetwork getNeuralNetwork() {
        return neuralNetwork;
    }

    /**
     * Returns copy of NNFunctionEstimator.
     *
     * @return copy of NNFunctionEstimator.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public FunctionEstimator copy() throws IOException, ClassNotFoundException, DynamicParamException {
        return params == null ? new NNFunctionEstimator(memory, neuralNetwork.copy()) : new NNFunctionEstimator(memory, neuralNetwork.copy(), params);
    }

    /**
     * Resets NNFunctionEstimator.
     *
     */
    public void reset() {
        super.reset();
        stateValueMap = new HashMap<>();
        stateTransitionValueMap = new HashMap<>();
    }

    /**
     * Reinitializes NNFunctionEstimator.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void reinitialize() throws NeuralNetworkException, MatrixException {
        neuralNetwork.reinitialize();
    }

    /**
     * Predicts state values corresponding to a state.
     *
     * @param state state.
     * @return state values corresponding to a state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     */
    public Matrix predict(Matrix state) throws NeuralNetworkException, MatrixException {
        if (stateValueMap.containsKey(state)) return stateValueMap.get(state);
        Matrix values = neuralNetwork.predict(new MMatrix(state)).get(0);
        stateValueMap.put(state, values);
        return values;
    }

    /**
     * Stores state transition values pair.
     * @param stateTransition state transition.
     * @param values values.
     */
    public void store(StateTransition stateTransition, Matrix values) {
        stateTransitionValueMap.put(stateTransition, values);
    }

    /**
     * Updates (trains) neural network.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void update() throws NeuralNetworkException, DynamicParamException, AgentException, MatrixException {
        LinkedHashMap<Integer, MMatrix> states = new LinkedHashMap<>();
        LinkedHashMap<Integer, MMatrix> stateValues = new LinkedHashMap<>();
        TreeMap<Integer, Double> importanceSamplingWeights = new TreeMap<>();
        int index = 0;
        for (StateTransition stateTransition : stateTransitionValueMap.keySet()) {
            states.put(index, new MMatrix(stateTransition.environmentState.state()));
            stateValues.put(index, new MMatrix(stateTransitionValueMap.get(stateTransition)));
            if (applyImportanceSamplingWeights) importanceSamplingWeights.put(index++, stateTransition.importanceSamplingWeight);
            index++;
        }
        if (applyImportanceSamplingWeights) neuralNetwork.setImportanceSamplingWeights(importanceSamplingWeights);
        neuralNetwork.train(new BasicSampler(states, stateValues, "fullSet = true, numberOfIterations = " + numberOfIterations));

        updateComplete();
    }

    /**
     * Appends parameters to this NNFunctionEstimator from another NNFunctionEstimator.
     *
     * @param functionEstimator function estimator used to update current function estimator.
     * @param fullUpdate if true full update is done.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    public void append(FunctionEstimator functionEstimator, boolean fullUpdate) throws MatrixException, AgentException {
        super.append();
        if (fullUpdate) neuralNetwork.append(((NNFunctionEstimator)functionEstimator).getNeuralNetwork(), 1);
        else neuralNetwork.append(((NNFunctionEstimator)functionEstimator).getNeuralNetwork(), targetFunctionTau);
    }

}
