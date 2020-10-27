/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.function;

import core.NeuralNetwork;
import core.NeuralNetworkException;
import core.reinforcement.Agent;
import core.reinforcement.AgentException;
import core.reinforcement.memory.Memory;
import core.reinforcement.memory.StateTransition;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.BinaryFunctionType;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.sampling.BasicSampler;

import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.TreeMap;

/**
 * Class that defines NNFunctionEstimator.
 *
 */
public class NNFunctionEstimator extends AbstractFunctionEstimator {

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
    private double tau = 0.001;

    /**
     * If true applies importance sampling weights.
     *
     */
    private final boolean applyImportanceSamplingWeights;

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
     * @param numberOfActions number of actions.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public NNFunctionEstimator(Memory memory, NeuralNetwork neuralNetwork, int numberOfActions) throws MatrixException {
        super (memory, numberOfActions);
        if (neuralNetwork.getOutputLayer().getLayerWidth() < numberOfActions) throw new MatrixException("Neural network has less output than number of actions.");
        this.neuralNetwork = neuralNetwork;
        this.isStateActionValueFunction = getNeuralNetwork().getOutputLayer().getLossFunctionType() == BinaryFunctionType.POLICY_VALUE;
        applyImportanceSamplingWeights = memory.applyImportanceSamplingWeights();
    }

    /**
     * Constructor for NNFunctionEstimator.
     *
     * @param memory memory reference.
     * @param neuralNetwork neural network reference.
     * @param numberOfActions number of actions.
     * @param isStateActionValueFunction true if estimator is state action value function otherwise false.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public NNFunctionEstimator(Memory memory, NeuralNetwork neuralNetwork, int numberOfActions, boolean isStateActionValueFunction) throws MatrixException {
        super (memory, numberOfActions);
        if (neuralNetwork.getOutputLayer().getLayerWidth() < numberOfActions) throw new MatrixException("Neural network has less output than number of actions.");
        this.neuralNetwork = neuralNetwork;
        this.isStateActionValueFunction = isStateActionValueFunction;
        applyImportanceSamplingWeights = memory.applyImportanceSamplingWeights();
    }

    /**
     * Constructor for NNFunctionEstimator.
     *
     * @param memory memory reference.
     * @param neuralNetwork neural network reference.
     * @param numberOfActions number of actions.
     * @param params parameters for function
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public NNFunctionEstimator(Memory memory, NeuralNetwork neuralNetwork, int numberOfActions, String params) throws MatrixException, DynamicParamException {
        this(memory, neuralNetwork, numberOfActions);
        this.params = params;
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Constructor for NNFunctionEstimator.
     *
     * @param memory memory reference.
     * @param neuralNetwork neural network reference.
     * @param numberOfActions number of actions.
     * @param isStateActionValueFunction true if estimator is state action value function otherwise false.
     * @param params parameters for function
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public NNFunctionEstimator(Memory memory, NeuralNetwork neuralNetwork, int numberOfActions, boolean isStateActionValueFunction, String params) throws MatrixException, DynamicParamException {
        this(memory, neuralNetwork, numberOfActions, isStateActionValueFunction);
        this.params = params;
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for NNFunctionEstimator.
     *
     * @return parameters used for NNFunctionEstimator.
     */
    protected HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("numberOfIterations", DynamicParam.ParamType.INT);
        paramDefs.put("tau", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for NNFunctionEstimator.<br>
     * <br>
     * Supported parameters are:<br>
     *     - numberOfIterations: number of training or validation iterations executed during step. Default value 1.<br>
     *     - tau: update rate pf target function. Default value 0.001.<br>
     *
     * @param params parameters used for NNFunctionEstimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("numberOfIterations")) {
            numberOfIterations = params.getValueAsInteger("numberOfIterations");
            if (numberOfIterations < 1) throw new DynamicParamException("Number of iterations must be at least 1.");
        }
        if (params.hasParam("tau")) tau = params.getValueAsDouble("tau");
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
    }

    /**
     * Stops function estimator.
     *
     */
    public void stop() {
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
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public FunctionEstimator copy() throws IOException, ClassNotFoundException, DynamicParamException, MatrixException {
        return params == null ? new NNFunctionEstimator(memory, neuralNetwork.copy(), numberOfActions) : new NNFunctionEstimator(memory, neuralNetwork.copy(), numberOfActions, params);
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
     *
     * @param agent deep agent.
     * @param stateTransition state transition.
     * @param values values.
     */
    public void store(Agent agent, StateTransition stateTransition, Matrix values) throws AgentException {
        super.store(agent);
        stateTransitionValueMap.put(stateTransition, values);
    }

    /**
     * Updates (trains) neural network.
     *
     * @param agent agent.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if agent is not registered for ongoing update cycle.
     */
    public void update(Agent agent) throws NeuralNetworkException, DynamicParamException, AgentException {
        if (!super.updateAndCheck(agent)) return;

        LinkedHashMap<Integer, MMatrix> states = new LinkedHashMap<>();
        LinkedHashMap<Integer, MMatrix> stateValues = new LinkedHashMap<>();
        TreeMap<Integer, Double> importanceSamplingWeights = new TreeMap<>();
        int index = 0;
        for (StateTransition stateTransition : stateTransitionValueMap.keySet()) {
            states.put(index, new MMatrix(stateTransition.environmentState.state));
            stateValues.put(index, new MMatrix(stateTransitionValueMap.get(stateTransition)));
            if (applyImportanceSamplingWeights) importanceSamplingWeights.put(index++, stateTransition.importanceSamplingWeight);
            index++;
        }
        if (applyImportanceSamplingWeights) neuralNetwork.setImportanceSamplingWeights(importanceSamplingWeights);
        neuralNetwork.train(new BasicSampler(states, stateValues, "fullSet = true, numberOfIterations = " + numberOfIterations));
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
        else neuralNetwork.append(((NNFunctionEstimator)functionEstimator).getNeuralNetwork(), tau);
    }

}
