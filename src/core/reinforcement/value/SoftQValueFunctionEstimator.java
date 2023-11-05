/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.value;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.Memory;
import core.reinforcement.agent.State;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.util.Random;
import java.util.TreeSet;

/**
 * Implements soft Q value function estimator.<br>
 *
 */
public class SoftQValueFunctionEstimator extends QTargetValueFunctionEstimator {

    /**
     * Parameter name types for soft Q value function estimator.
     *     - dualFunctionEstimation: if true uses dual function estimation for value function. Default value true.<br>
     *     - minMaxBalance: defines balance (probability) between choosing maximal and minimal value between estimator (experimental parameter). Default value 1.<br>
     */
    private final static String paramNameTypes = "(dualFunctionEstimation:BOOLEAN), " +
            "(minMaxBalance:DOUBLE)";

    /**
     * Reference to second function estimator.
     *
     */
    protected FunctionEstimator functionEstimator2;

    /**
     * Reference to current policy.
     *
     */
    protected final FunctionEstimator policyFunctionEstimator;

    /**
     * Reference to soft Q alpha matrix.
     *
     */
    private final Matrix softQAlphaMatrix = new DMatrix(0);

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
     * Random function.
     *
     */
    private final Random random = new Random();

    /**
     * Constructor for soft Q value function estimator.
     *
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param functionEstimator reference to function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public SoftQValueFunctionEstimator(FunctionEstimator policyFunctionEstimator, FunctionEstimator functionEstimator) throws DynamicParamException {
        this(policyFunctionEstimator, functionEstimator, null);
    }

    /**
     * Constructor for soft Q value function estimator.
     *
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param functionEstimator reference to value FunctionEstimator.
     * @param params parameters for QTargetValueFunctionEstimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public SoftQValueFunctionEstimator(FunctionEstimator policyFunctionEstimator, FunctionEstimator functionEstimator, String params) throws DynamicParamException {
        super(functionEstimator, params);
        this.policyFunctionEstimator = policyFunctionEstimator;
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
     * Returns parameters used for soft Q value function estimator.
     *
     * @return parameters used for soft Q value function estimator.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + SoftQValueFunctionEstimator.paramNameTypes;
    }

    /**
     * Sets parameters used for soft Q value function estimator.<br>
     * <br>
     * Supported parameters are:<br>
     *     - dualFunctionEstimation: if true uses dual function estimation for value function. Default value true.<br>
     *     - minMaxBalance: defines balance (probability) between choosing maximal and minimal value between estimator (experimental parameter). Default value 1.<br>
     *
     * @param params parameters used for soft Q value function estimator.
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
     * Returns policy function estimator.
     *
     * @return policy function estimator.
     */
    private FunctionEstimator getPolicyFunctionEstimator() {
        return policyFunctionEstimator;
    }

    /**
     * Returns reference to value function.
     *
     * @return reference to value function.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public ValueFunction reference() throws DynamicParamException, MatrixException, IOException, ClassNotFoundException {
        return new SoftQValueFunctionEstimator(getPolicyFunctionEstimator(), getFunctionEstimator().reference(), getParams());
    }

    /**
     * Returns reference to value function.
     *
     * @param policyFunctionEstimator reference to policy function estimator.
     * @return reference to value function.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public ValueFunction reference(FunctionEstimator policyFunctionEstimator) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException {
        return new SoftQValueFunctionEstimator(policyFunctionEstimator, getFunctionEstimator().reference(), getParams());
    }

    /**
     * Returns reference to value function.
     *
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to value function.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public ValueFunction reference(boolean sharedValueFunctionEstimator, boolean sharedMemory) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException {
        return new SoftQValueFunctionEstimator(getPolicyFunctionEstimator(), sharedValueFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(sharedMemory), getParams());
    }

    /**
     * Returns reference to value function.
     *
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to value function.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public ValueFunction reference(FunctionEstimator policyFunctionEstimator, boolean sharedValueFunctionEstimator, boolean sharedMemory) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException {
        return new SoftQValueFunctionEstimator(policyFunctionEstimator, sharedValueFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(sharedMemory), getParams());
    }

    /**
     * Returns reference to value function.
     *
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @param memory reference to memory.
     * @return reference to value function.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public ValueFunction reference(boolean sharedValueFunctionEstimator, Memory memory) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException {
        return new SoftQValueFunctionEstimator(getPolicyFunctionEstimator(), sharedValueFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(memory), getParams());
    }

    /**
     * Returns reference to value function.
     *
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @param memory reference to memory.
     * @return reference to value function.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public ValueFunction reference(FunctionEstimator policyFunctionEstimator, boolean sharedValueFunctionEstimator, Memory memory) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException {
        return new SoftQValueFunctionEstimator(getPolicyFunctionEstimator(), sharedValueFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(memory), getParams());
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
        functionEstimator2 = dualFunctionEstimation ? getFunctionEstimator().copy() : null;
        if (getFunctionEstimator2() != null) {
            if (!isStateActionValueFunction()) getFunctionEstimator2().registerAgent(agent);
            getFunctionEstimator2().start();
        }
    }

    /**
     * Stops function estimator
     *
     */
    public void stop() {
        super.stop();
        if (getFunctionEstimator2() != null) getFunctionEstimator2().stop();
    }

    /**
     * Resets function estimator.
     *
     */
    public void resetFunctionEstimator() {
        super.resetFunctionEstimator();
        if (getFunctionEstimator2() != null) getFunctionEstimator2().reset();
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
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void updateFunctionEstimator() throws NeuralNetworkException, MatrixException, DynamicParamException {
        TreeSet<State> sampledStates = getSampledStates();
        if (sampledStates == null || sampledStates.isEmpty()) {
            getFunctionEstimator().abortUpdate();
            if (getFunctionEstimator2() != null) getFunctionEstimator2().abortUpdate();
        }
        else {
            updateTargetValues(getFunctionEstimator(), sampledStates);
            if (getFunctionEstimator2() !=  null) updateTargetValues(getFunctionEstimator2(), sampledStates);
            if (!isStateActionValueFunction()) {
                getFunctionEstimator().update();
                if (getFunctionEstimator2() !=  null) getFunctionEstimator2().update();
            }
        }
    }

    /**
     * Returns target value based on next state. Uses target action with maximal value as defined by target FunctionEstimator.
     *
     * @param nextState next state.
     * @return target value based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double getTargetValue(State nextState) throws NeuralNetworkException, MatrixException {
        Matrix targetPolicyValues = getPolicyFunctionEstimator().predictPolicyValues(nextState);
        int targetAction = getPolicyFunctionEstimator().argmax(targetPolicyValues, nextState.environmentState.availableActions());
        return getFunctionEstimator2() != null ? getClippedValue(nextState) : getTargetValue(getFunctionEstimator(), nextState) - softQAlphaMatrix.getValue(0, 0, 0) * Math.log(targetPolicyValues.getValue(targetAction, 0, 0));
    }

    /**
     * Returns clipped value.
     *
     * @param nextState next state.
     * @return clipped value.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private double getClippedValue(State nextState) throws MatrixException, NeuralNetworkException {
        double value1 = getTargetValue(getFunctionEstimator(), nextState);
        double value2 = getTargetValue(getFunctionEstimator2(), nextState);
        return random.nextDouble() < minMaxBalance ? Math.min(value1, value2) : Math.max(value1, value2);
    }

}
