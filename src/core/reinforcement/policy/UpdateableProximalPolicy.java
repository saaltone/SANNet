/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement.policy;

import core.NeuralNetworkException;
import core.reinforcement.RLSample;
import core.reinforcement.function.FunctionEstimator;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.util.HashMap;

/**
 * Class that defines UpdateableProximalPolicy. Implements Proximal Policy Optimization (PPO).
 *
 */
public class UpdateableProximalPolicy extends AbstractUpdateablePolicy {

    /**
     * Reference to previous policy function estimator.
     *
     */
    private final FunctionEstimator previousFunctionEstimator;

    /**
     * Epsilon value for Proximal Policy clipping function.
     *
     */
    private double epsilon = 0.2;

    /**
     * Constructor for UpdateableProximalPolicy.
     *
     * @param policy reference to policy.
     * @param functionEstimator reference to FunctionEstimator.
     * @throws IOException throws exception if creation of previousFunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of previousFunctionEstimator fails.
     */
    public UpdateableProximalPolicy(Policy policy, FunctionEstimator functionEstimator) throws IOException, ClassNotFoundException {
        super(policy, functionEstimator);
        previousFunctionEstimator = functionEstimator.copy();
    }

    /**
     * Constructor for UpdateableProximalPolicy.
     *
     * @param policy reference to policy.
     * @param functionEstimator reference to FunctionEstimator.
     * @param params parameters for policy.
     * @throws IOException throws exception if creation of previousFunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of previousFunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public UpdateableProximalPolicy(Policy policy, FunctionEstimator functionEstimator, String params) throws IOException, ClassNotFoundException, DynamicParamException {
        this(policy, functionEstimator);
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for UpdateableProximalPolicy.
     *
     * @return parameters used for UpdateableProximalPolicy.
     */
    protected HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("epsilon", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for UpdateableProximalPolicy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - updateEntropy: Epsilon value for Proximal Policy clipping function. Default value 0.2.<br>
     *
     * @param params parameters used for UpdateableProximalPolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("epsilon")) epsilon = params.getValueAsDouble("epsilon");
    }

    /**
     * Starts FunctionEstimator
     *
     * @throws NeuralNetworkException throws exception if start of neural network estimator(s) fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     */
    public void start() throws NeuralNetworkException, MatrixException {
        super.start();
        previousFunctionEstimator.start();
    }

    /**
     * Stops FunctionEstimator
     *
     */
    public void stop() {
        super.stop();
        previousFunctionEstimator.stop();
    }

    /**
     * Preprocesses policy gradient setting.
     *
     */
    protected void preProcess() {
    }

    /**
     * Returns policy gradient value for sample.
     *
     * @param sample sample
     * @param hasImportanceSamplingWeight if true sample has importance sample weight set.
     * @return policy gradient value for sample.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected double getPolicyGradientValue(RLSample sample, boolean hasImportanceSamplingWeight) throws NeuralNetworkException, MatrixException {
        double previousActionValue = previousFunctionEstimator.predict(sample.state.stateMatrix).getValue(sample.state.action, 0);
        double rValue = sample.policyValue / previousActionValue;
        double clippedRValue = Math.min(Math.max(rValue, 1 - epsilon), 1 + epsilon);
        double advantage = sample.tdTarget - sample.baseline;
        return Math.min(rValue * advantage, clippedRValue * advantage);
    }

    /**
     * Postprocesses policy gradient setting.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void postProcess() throws MatrixException {
        previousFunctionEstimator.append(functionEstimator, true);
    }

}
