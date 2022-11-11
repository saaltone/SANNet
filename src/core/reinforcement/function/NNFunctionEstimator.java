/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.reinforcement.function;

import core.network.NeuralNetwork;
import core.network.NeuralNetworkException;
import core.reinforcement.agent.AgentException;
import core.reinforcement.memory.Memory;
import core.reinforcement.agent.StateTransition;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.sampling.BasicSampler;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

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
     * Intermediate map for state value pairs as value cache.
     *
     */
    private final HashMap<StateTransition, Matrix> stateTransitionCache = new HashMap<>();

    /**
     * Intermediate map for state transition value pairs for update.
     *
     */
    private final HashMap<StateTransition, Matrix> stateTransitionValueMap = new HashMap<>();

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
        super (memory, neuralNetwork.getInputLayer().getLayerWidth(), neuralNetwork.getOutputLayer().isMultiOutput() ? neuralNetwork.getOutputLayer().getLayerWidth() - 1 : neuralNetwork.getOutputLayer().getLayerWidth(), neuralNetwork.getOutputLayer().isMultiOutput(), params);
        this.neuralNetwork = neuralNetwork;
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
     * Predicts state values corresponding to a state.
     *
     * @param stateTransition state.
     * @return state values corresponding to a state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     */
    public Matrix predict(StateTransition stateTransition) throws NeuralNetworkException, MatrixException {
        Matrix values = stateTransitionCache.get(stateTransition);
        if (values == null)  {
            values = neuralNetwork.predict(new MMatrix(stateTransition.environmentState.state())).get(0);
            stateTransitionCache.put(stateTransition, values);
        }
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
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    public void update() throws NeuralNetworkException, DynamicParamException, AgentException, MatrixException, IOException, ClassNotFoundException {
        HashMap<Integer, MMatrix> states = new HashMap<>();
        HashMap<Integer, MMatrix> stateValues = new HashMap<>();
        HashMap<Integer, Double> importanceSamplingWeights = new HashMap<>();
        int index = 0;
        for (Map.Entry<StateTransition, Matrix> entry: stateTransitionValueMap.entrySet()) {
            StateTransition stateTransition = entry.getKey();
            Matrix matrix = entry.getValue();
            states.put(index, new MMatrix(stateTransition.environmentState.state()));
            stateValues.put(index, new MMatrix(matrix));
            if (applyImportanceSamplingWeights) importanceSamplingWeights.put(index, stateTransition.importanceSamplingWeight);
            index++;
        }
        if (applyImportanceSamplingWeights) neuralNetwork.setImportanceSamplingWeights(importanceSamplingWeights);
        neuralNetwork.train(new BasicSampler(states, stateValues, "fullSet = true, randomOrder = false, numberOfIterations = " + numberOfIterations));

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
