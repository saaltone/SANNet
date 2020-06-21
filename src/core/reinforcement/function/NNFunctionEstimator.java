/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.function;

import core.NeuralNetwork;
import core.NeuralNetworkException;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.sampling.BasicSampler;

import java.io.IOException;
import java.io.Serializable;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.TreeMap;

/**
 * Class that defines neural network as function estimator.
 *
 */
public class NNFunctionEstimator implements FunctionEstimator, Serializable {

    private static final long serialVersionUID = 233674344549590971L;

    /**
     * Neural network function estimator.
     *
     */
    private final NeuralNetwork neuralNetwork;

    /**
     * Number of actions for function estimator.
     *
     */
    private final int numberOfActions;

    /**
     * Update rate for target function estimator.
     *
     */
    private double tau = 0.001;

    /**
     * If true applies importance sampling weights.
     *
     */
    private boolean applyImportanceSamplingWeights = true;

    /**
     * Constructor for NNFunctionEstimator.
     *
     * @param neuralNetwork neural network reference.
     */
    public NNFunctionEstimator(NeuralNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
        numberOfActions = neuralNetwork.getOutputLayer().getLayerWidth();
    }

    /**
     * Constructor for NNFunctionEstimator.
     *
     * @param neuralNetwork neural network reference.
     * @param params parameters for function
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public NNFunctionEstimator(NeuralNetwork neuralNetwork, String params) throws DynamicParamException {
        this(neuralNetwork);
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for NNFunctionEstimator.
     *
     * @return parameters used for NNFunctionEstimator.
     */
    protected HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("tau", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("applyImportanceSamplingWeights", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for NNFunctionEstimator.<br>
     * <br>
     * Supported parameters are:<br>
     *     - tau: update rate for for target function (applied is updateCycle = 0). Default value 0.001.<br>
     *     - applyImportanceSamplingWeights: if true applies importance sampling weights as applicable. Default value True.<br>
     *
     * @param params parameters used for NNFunction.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("tau")) tau = params.getValueAsDouble("tau");
        if (params.hasParam("applyImportanceSamplingWeights")) applyImportanceSamplingWeights = params.getValueAsBoolean("applyImportanceSamplingWeights");
    }

    /**
     * Returns number of actions for NNFunctionEstimator.
     *
     * @return number of actions for NNFunctionEstimator.
     */
    public int getNumberOfActions() {
        return numberOfActions;
    }

    /**
     * Starts function estimator.
     *
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     */
    public void start() throws NeuralNetworkException, MatrixException {
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
     * Returns neural network as function estimator.
     *
     * @return neural network.
     */
    public NeuralNetwork getNeuralNetwork() {
        return neuralNetwork;
    }

    /**
     * Returns copy of function estimator.
     *
     * @return copy of function estimator.
     * @throws IOException throws exception if creation of function estimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of function estimator copy fails.
     */
    public FunctionEstimator copy() throws IOException, ClassNotFoundException {
        return new NNFunctionEstimator(neuralNetwork.copy());
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
        return neuralNetwork.predict(new MMatrix(state)).get(0);
    }

    /**
     * Sets importance sampling weights.
     *
     * @param ISWeights importance sampling weights.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    public void setImportanceSamplingWeights(TreeMap<Integer, Double> ISWeights) throws NeuralNetworkException {
        if (applyImportanceSamplingWeights) neuralNetwork.setImportanceSamplingWeights(ISWeights);
    }

    /**
     * Updates (trains) neural network.
     *
     * @param states states to be updated.
     * @param stateValues state values to be updated.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void train(LinkedHashMap<Integer, MMatrix> states, LinkedHashMap<Integer, MMatrix> stateValues) throws NeuralNetworkException, DynamicParamException {
        neuralNetwork.train(new BasicSampler(states, stateValues, "fullSet = true"));
    }

    /**
     * Appends parameters of this neural network estimator from another estimator function.
     *
     * @param functionEstimator function estimator used to update current function estimator.
     * @param fullUpdate if true full update is done.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void append(FunctionEstimator functionEstimator, boolean fullUpdate) throws MatrixException {
        if (fullUpdate) neuralNetwork.append(((NNFunctionEstimator)functionEstimator).getNeuralNetwork(), 1);
        else neuralNetwork.append(((NNFunctionEstimator)functionEstimator).getNeuralNetwork(), tau);
    }

    /**
     * Returns error of FunctionEstimator.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return error of FunctionEstimator.
     */
    public double getError() throws MatrixException {
        return neuralNetwork.getOutputError();
    }

}
