/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.reinforcement.value;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.AgentException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.Memory;
import core.reinforcement.memory.StateTransition;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.util.Random;
import java.util.TreeSet;

/**
 * Implements soft Q value function estimator.<br>
 *
 */
public class SoftQValueFunctionEstimator extends AbstractActionValueFunctionEstimator {

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
     * Random function.
     *
     */
    private final Random random = new Random();

    /**
     * Constructor for soft Q value function estimator.
     *
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param functionEstimator reference to function estimator.
     * @param softQAlphaMatrix reference to softQAlphaMatrix.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if soft Q alpha matrix is non-scalar matrix.
     * @throws NeuralNetworkException throws exception if starting of function estimator fails.
     */
    public SoftQValueFunctionEstimator(FunctionEstimator policyFunctionEstimator, FunctionEstimator functionEstimator, Matrix softQAlphaMatrix) throws IOException, ClassNotFoundException, DynamicParamException, MatrixException, AgentException, NeuralNetworkException {
        super(functionEstimator);
        functionEstimator.createTargetFunctionEstimator();
        this.policyFunctionEstimator = policyFunctionEstimator;
        functionEstimator2 = dualFunctionEstimation ? functionEstimator.reference() : null;
        if (functionEstimator2 != null) functionEstimator2.createTargetFunctionEstimator();
        this.softQAlphaMatrix = softQAlphaMatrix;
        if (!softQAlphaMatrix.isScalar()) throw new AgentException("Soft Q Alpha matrix must be scalar matrix.");
    }

    /**
     * Constructor for soft Q value function estimator.
     *
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param functionEstimator reference to value FunctionEstimator.
     * @param softQAlphaMatrix reference to softQAlphaMatrix.
     * @param params parameters for QTargetValueFunctionEstimator.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if soft Q alpha matrix is non-scalar matrix.
     * @throws NeuralNetworkException throws exception if starting of function estimator fails.
     */
    public SoftQValueFunctionEstimator(FunctionEstimator policyFunctionEstimator, FunctionEstimator functionEstimator, Matrix softQAlphaMatrix, String params) throws IOException, ClassNotFoundException, DynamicParamException, MatrixException, AgentException, NeuralNetworkException {
        super(functionEstimator, params);
        functionEstimator.createTargetFunctionEstimator();
        this.policyFunctionEstimator = policyFunctionEstimator;
        functionEstimator2 = dualFunctionEstimation ? functionEstimator.reference() : null;
        if (functionEstimator2 != null) functionEstimator2.createTargetFunctionEstimator();
        this.softQAlphaMatrix = softQAlphaMatrix;
        if (!softQAlphaMatrix.isScalar()) throw new AgentException("Soft Q Alpha matrix must be scalar matrix.");
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
     * Returns reference to value function.
     *
     * @return reference to value function.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if soft Q alpha matrix is non-scalar matrix.
     * @throws NeuralNetworkException throws exception if starting of function estimator fails.
     */
    public ValueFunction reference() throws DynamicParamException, MatrixException, IOException, ClassNotFoundException, AgentException, NeuralNetworkException {
        return new SoftQValueFunctionEstimator(policyFunctionEstimator.reference(), functionEstimator.reference(), softQAlphaMatrix, getParams());
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
     * @throws AgentException throws exception if soft Q alpha matrix is non-scalar matrix.
     * @throws NeuralNetworkException throws exception if starting of function estimator fails.
     */
    public ValueFunction reference(boolean sharedValueFunctionEstimator, boolean sharedMemory) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException, AgentException, NeuralNetworkException {
        return new SoftQValueFunctionEstimator(policyFunctionEstimator.reference(), sharedValueFunctionEstimator ? functionEstimator : functionEstimator.reference(sharedMemory), softQAlphaMatrix, getParams());
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
     * @throws AgentException throws exception if soft Q alpha matrix is non-scalar matrix.
     * @throws NeuralNetworkException throws exception if starting of function estimator fails.
     */
    public ValueFunction reference(boolean sharedValueFunctionEstimator, Memory memory) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException, AgentException, NeuralNetworkException {
        return new SoftQValueFunctionEstimator(policyFunctionEstimator.reference(), sharedValueFunctionEstimator ? functionEstimator : functionEstimator.reference(memory), softQAlphaMatrix, getParams());
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
    public void start() throws NeuralNetworkException, MatrixException, DynamicParamException, IOException, ClassNotFoundException {
        super.start();
        if (dualFunctionEstimation) functionEstimator2.start();
    }

    /**
     * Stops function estimator
     *
     */
    public void stop() {
        super.stop();
        if (dualFunctionEstimation) functionEstimator2.stop();
    }

    /**
     * Resets function estimator.
     *
     */
    public void resetFunctionEstimator() {
        super.resetFunctionEstimator();
        if (dualFunctionEstimation) functionEstimator2.reset();
    }

    /**
     * Updates function estimator.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    public void updateFunctionEstimator() throws NeuralNetworkException, MatrixException, DynamicParamException, AgentException, IOException, ClassNotFoundException {
        TreeSet<StateTransition> sampledStateTransitions = getSampledStateTransitions();
        if (sampledStateTransitions == null || sampledStateTransitions.isEmpty()) {
            functionEstimator.abortUpdate();
            return;
        }

        updateFunctionEstimatorMemory(sampledStateTransitions);

        if (!isStateActionValueFunction()) {
            for (StateTransition stateTransition : sampledStateTransitions) {
                updateTargetValues(functionEstimator, stateTransition);
                if (dualFunctionEstimation) updateTargetValues(functionEstimator2, stateTransition);
            }
            functionEstimator.update();
            if (dualFunctionEstimation) functionEstimator2.update();
        }

    }

    /**
     * Returns target values as estimated by function estimator.
     *
     * @param currentFunctionEstimator function estimator.
     * @param stateTransition state transition.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private void updateTargetValues(FunctionEstimator currentFunctionEstimator, StateTransition stateTransition) throws MatrixException, NeuralNetworkException {
        Matrix targetValues = currentFunctionEstimator.predict(stateTransition);
        targetValues.setValue(getFunctionIndex(stateTransition), 0, stateTransition.tdTarget);
        currentFunctionEstimator.store(stateTransition, targetValues);
    }

    /**
     * Returns target value based on next state. Uses target action with maximal value as defined by target FunctionEstimator.
     *
     * @param nextStateTransition next state transition.
     * @return target value based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double getTargetValue(StateTransition nextStateTransition) throws NeuralNetworkException, MatrixException {
        Matrix targetPolicyValues = policyFunctionEstimator.predict(nextStateTransition);
        int targetAction = policyFunctionEstimator.argmax(targetPolicyValues, nextStateTransition.environmentState.availableActions());
        return (dualFunctionEstimation ? getClippedValue(nextStateTransition, targetAction) : getTargetValue(functionEstimator, nextStateTransition, targetAction)) - softQAlphaMatrix.getValue(0, 0) * Math.log(targetPolicyValues.getValue(targetAction, 0));
    }

    /**
     * Returns clipped value.
     *
     * @param nextStateTransition next state transition.
     * @param targetAction target action.
     * @return clipped value.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private double getClippedValue(StateTransition nextStateTransition, int targetAction) throws MatrixException, NeuralNetworkException {
        double value1 = getTargetValue(functionEstimator, nextStateTransition, targetAction);
        double value2 = getTargetValue(functionEstimator2, nextStateTransition, targetAction);
        return random.nextDouble() < minMaxBalance ? Math.min(value1, value2) : Math.max(value1, value2);
    }

    /**
     * Return target value as estimated by target function estimator.
     *
     * @param currentFunctionEstimator function estimator.
     * @param nextStateTransition next state transition
     * @param targetAction target action.
     * @return target value.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private double getTargetValue(FunctionEstimator currentFunctionEstimator, StateTransition nextStateTransition, int targetAction) throws MatrixException, NeuralNetworkException {
        return getValues(currentFunctionEstimator.getTargetFunctionEstimator(), nextStateTransition).getValue(targetAction, 0);
    }

    /**
     * Appends parameters to this value function from another value function.
     *
     * @param valueFunction value function used to update current value function.
     * @param tau tau which controls contribution of other value function.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    public void append(ValueFunction valueFunction, double tau) throws MatrixException, AgentException, NeuralNetworkException, IOException, DynamicParamException, ClassNotFoundException {
        super.append(valueFunction, tau);
        functionEstimator.createTargetFunctionEstimator();
        functionEstimator2 = dualFunctionEstimation ? functionEstimator.reference() : null;
        if (functionEstimator2 != null) {
            functionEstimator2.start();
            functionEstimator2.createTargetFunctionEstimator();
        }
    }

}
