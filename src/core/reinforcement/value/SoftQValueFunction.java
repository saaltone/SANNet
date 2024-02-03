/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.value;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.agent.State;
import core.reinforcement.policy.updateablepolicy.UpdateableSoftQPolicy;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.io.IOException;
import java.util.Random;
import java.util.TreeSet;

/**
 * Implements soft Q value function.<br>
 *
 */
public class SoftQValueFunction extends QTargetValueFunction {

    /**
     * Parameter name types for soft Q value function.
     *     - dualFunctionEstimation: if true uses dual function estimation for value function. Default value true.<br>
     *     - minMaxBalance: defines balance (probability) between choosing maximal and minimal value between estimator (experimental parameter). Default value 1.<br>
     */
    private final static String paramNameTypes = "(dualFunctionEstimation:BOOLEAN), " +
            "(minMaxBalance:DOUBLE)";

    /**
     * Reference to second function estimator.
     *
     */
    private FunctionEstimator functionEstimator2;

    /**
     * Reference to current policy.
     *
     */
    private UpdateableSoftQPolicy updateableSoftQPolicy;

    /**
     * Reference to soft Q alpha matrix.
     *
     */
    private final Matrix softQAlphaMatrix;

    /**
     * If true soft Q value function estimator has dual function estimator.
     *
     */
    private boolean dualFunctionEstimation;

    /**
     * Defines balance (probability) between choosing maximal and minimal value between estimator (experimental parameter).
     *
     */
    private double minMaxBalance;

    /**
     * Log function.
     *
     */
    private final UnaryFunction logFunction = new UnaryFunction(UnaryFunctionType.LOG);

    /**
     * Random function.
     *
     */
    private final Random random = new Random();

    /**
     * Constructor for soft Q value function.
     *
     * @param functionEstimator       reference to value FunctionEstimator.
     * @param params                  parameters for QTargetValueFunctionEstimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public SoftQValueFunction(FunctionEstimator functionEstimator, String params) throws MatrixException, DynamicParamException {
        super(functionEstimator, params);
        this.functionEstimator2 = null;
        softQAlphaMatrix = new DMatrix(0);
    }

    /**
     * Constructor for soft Q value function.
     *
     * @param functionEstimator       reference to value FunctionEstimator.
     * @param functionEstimator2      reference to second value FunctionEstimator.
     * @param softQAlphaMatrix        reference to soft Q alpha matrix.
     * @param params                  parameters for QTargetValueFunctionEstimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException       throws exception if matrix operation fails.
     */
    protected SoftQValueFunction(FunctionEstimator functionEstimator, FunctionEstimator functionEstimator2, Matrix softQAlphaMatrix, String params) throws MatrixException, DynamicParamException {
        super(functionEstimator, params);
        this.functionEstimator2 = functionEstimator2;
        this.softQAlphaMatrix = softQAlphaMatrix;
    }

    /**
     * Sets updateable soft Q policy.
     *
     * @param updateableSoftQPolicy updateable soft Q policy.
     */
    public void setUpdateableSoftQPolicy(UpdateableSoftQPolicy updateableSoftQPolicy) {
        this.updateableSoftQPolicy = updateableSoftQPolicy;
    }

    /**
     * Returns policy function estimator.
     *
     * @return policy function estimator.
     */
    private UpdateableSoftQPolicy getUpdateableSoftQPolicy() {
        return updateableSoftQPolicy;
    }

    /**
     * Returns Soft Q Alpha Matrix.
     *
     * @return Soft Q Alpha Matrix.
     */
    public Matrix getSoftQAlphaMatrix() {
        return softQAlphaMatrix;
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        dualFunctionEstimation = true;
        minMaxBalance = 1;
    }

    /**
     * Returns parameters used for soft Q value function.
     *
     * @return parameters used for soft Q value function.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + SoftQValueFunction.paramNameTypes;
    }

    /**
     * Sets parameters used for soft Q value function.<br>
     * <br>
     * Supported parameters are:<br>
     *     - dualFunctionEstimation: if true uses dual function estimation for value function. Default value true.<br>
     *     - minMaxBalance: defines balance (probability) between choosing maximal and minimal value between estimator (experimental parameter). Default value 1.<br>
     *
     * @param params parameters used for soft Q value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
        if (params.hasParam("dualFunctionEstimation")) dualFunctionEstimation = params.getValueAsBoolean("dualFunctionEstimation");
        if (params.hasParam("minMaxBalance")) minMaxBalance = params.getValueAsDouble("minMaxBalance");
    }

    /**
     * Returns 2nd function estimator.
     *
     * @return 2nd function estimator.
     */
    private FunctionEstimator getFunctionEstimator2() {
        return functionEstimator2;
    }

    /**
     * Returns reference to value function.
     *
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @return reference to value function.
     * @throws IOException            throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     * @throws MatrixException        throws exception if neural network has less output than actions.
     */
    public ValueFunction reference(boolean sharedValueFunctionEstimator) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException {
        return new SoftQValueFunction(sharedValueFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(), sharedValueFunctionEstimator ? getFunctionEstimator2() : null, sharedValueFunctionEstimator ? softQAlphaMatrix : new DMatrix(0), getParams());
    }

    /**
     * Starts function estimator
     *
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void start(Agent agent) throws NeuralNetworkException, MatrixException, DynamicParamException, IOException, ClassNotFoundException {
        super.start(agent);
        if (functionEstimator2 == null) {
            functionEstimator2 = dualFunctionEstimation ? getFunctionEstimator().reference() : null;
            if (getFunctionEstimator2() != null) {
                getFunctionEstimator2().start();
            }
        }
        if (functionEstimator2 != null) getFunctionEstimator2().registerAgent(agent);
    }

    /**
     * Stops function estimator
     *
     * @throws NeuralNetworkException throws exception is neural network is not started.
     */
    public void stop() throws NeuralNetworkException {
        super.stop();
        if (getFunctionEstimator2() != null) getFunctionEstimator2().stop();
    }

    /**
     * Notifies that agent is ready to update.
     *
     * @param agent current agent.
     * @throws AgentException throws exception if agent is not registered for function estimator.
     * @return true if all registered agents are ready to update.
     */
    public boolean readyToUpdate(Agent agent) throws AgentException {
        return super.readyToUpdate(agent) && (getFunctionEstimator2() == null || getFunctionEstimator2().readyToUpdate(agent));
    }

    /**
     * Updates function estimator.
     *
     * @param sampledStates sampled states.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     */
    public void updateFunctionEstimator(TreeSet<State> sampledStates) throws NeuralNetworkException, MatrixException, DynamicParamException {
        super.updateFunctionEstimator(sampledStates);
        if (getFunctionEstimator2() != null) {
            updateTargetValues(getFunctionEstimator2(), sampledStates);
            if (!isStateActionValueFunction()) getFunctionEstimator2().update();
        }
    }

    /**
     * Returns target value based on next state. Uses target action with maximal value as defined by target FunctionEstimator.
     *
     * @param nextState next state.
     * @return target value based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     */
    public double getTargetValue(State nextState) throws NeuralNetworkException, MatrixException, DynamicParamException {
        Matrix policyValues = getUpdateableSoftQPolicy().getFunctionEstimator().predictPolicyValues(nextState);
        return policyValues.multiply(getTargetValues(nextState, false).subtract(softQAlphaMatrix.multiply(policyValues.apply(logFunction)))).getValue(nextState.targetAction, 0, 0);
    }

    /**
     * Returns target Q value.
     *
     * @param nextState next state.
     * @param useDefaultEstimator if true uses default estimator otherwise uses target estimator.
     * @return target Q value.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     */
    public Matrix getTargetValues(State nextState, boolean useDefaultEstimator) throws MatrixException, NeuralNetworkException, DynamicParamException {
        if (getFunctionEstimator2() != null) {
            Matrix values1 = useDefaultEstimator ? getFunctionEstimator().predictStateActionValues(nextState) : getFunctionEstimator().predictTargetStateActionValues(nextState);
            Matrix values2 = useDefaultEstimator ? getFunctionEstimator2().predictStateActionValues(nextState) : getFunctionEstimator2().predictTargetStateActionValues(nextState);
            return random.nextDouble() < minMaxBalance ? values1.min(values2) : values1.max(values2);
        }
        else {
            return useDefaultEstimator ? getFunctionEstimator().predictStateActionValues(nextState) : getFunctionEstimator().predictTargetStateActionValues(nextState);
        }
    }

    /**
     * Returns target action based on next state.
     *
     * @param nextState next state.
     * @return target action based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     */
    protected int getTargetAction(State nextState) throws NeuralNetworkException, MatrixException {
        return getUpdateableSoftQPolicy().getFunctionEstimator().argmax(getUpdateableSoftQPolicy().getFunctionEstimator().predictPolicyValues(nextState), nextState.environmentState.availableActions());
    }

}
