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
import java.util.TreeSet;

/**
 * Implements soft Q value function estimator.<br>
 *
 */
public class SoftQValueFunctionEstimator extends AbstractActionValueFunctionEstimator {

    /**
     * Parameter name types for soft Q value function estimator.
     *     - dualFunctionEstimation: if true uses dual function estimation for value function. Default value true.<br>
     *
     */
    private final static String paramNameTypes = "(dualFunctionEstimation:BOOLEAN)";

    /**
     * Reference to second function estimator.
     *
     */
    protected final FunctionEstimator functionEstimator2;

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
     */
    public SoftQValueFunctionEstimator(FunctionEstimator policyFunctionEstimator, FunctionEstimator functionEstimator, Matrix softQAlphaMatrix) throws IOException, ClassNotFoundException, DynamicParamException, MatrixException, AgentException {
        super(functionEstimator);
        this.policyFunctionEstimator = policyFunctionEstimator;
        functionEstimator.setTargetFunctionEstimator();
        functionEstimator2 = functionEstimator.reference();
        functionEstimator2.setTargetFunctionEstimator();
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
     */
    public SoftQValueFunctionEstimator(FunctionEstimator policyFunctionEstimator, FunctionEstimator functionEstimator, Matrix softQAlphaMatrix, String params) throws IOException, ClassNotFoundException, DynamicParamException, MatrixException, AgentException {
        super(functionEstimator, params);
        this.policyFunctionEstimator = policyFunctionEstimator;
        functionEstimator.setTargetFunctionEstimator();
        functionEstimator2 = functionEstimator.reference();
        functionEstimator2.setTargetFunctionEstimator();
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
     *
     * @param params parameters used for soft Q value function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
        if (params.hasParam("dualFunctionEstimation")) dualFunctionEstimation = params.getValueAsBoolean("dualFunctionEstimation");
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
     */
    public ValueFunction reference() throws DynamicParamException, MatrixException, IOException, ClassNotFoundException, AgentException {
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
     */
    public ValueFunction reference(boolean sharedValueFunctionEstimator, boolean sharedMemory) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException, AgentException {
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
     */
    public ValueFunction reference(boolean sharedValueFunctionEstimator, Memory memory) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException, AgentException {
        return new SoftQValueFunctionEstimator(policyFunctionEstimator.reference(), sharedValueFunctionEstimator ? functionEstimator : functionEstimator.reference(memory), softQAlphaMatrix, getParams());
    }

    /**
     * Starts function estimator
     *
     * @throws NeuralNetworkException throws exception if starting of value function estimator fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void start() throws NeuralNetworkException, MatrixException, DynamicParamException {
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
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if function estimator update fails.
     */
    public void updateFunctionEstimator() throws NeuralNetworkException, MatrixException, DynamicParamException, AgentException {
        TreeSet<StateTransition> sampledStateTransitions = getSampledStateTransitions();
        if (sampledStateTransitions == null || sampledStateTransitions.isEmpty()) return;

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
        return (dualFunctionEstimation ? Math.min(getTargetValue(functionEstimator, nextStateTransition, targetAction), getTargetValue(functionEstimator2, nextStateTransition, targetAction)) : getTargetValue(functionEstimator, nextStateTransition, targetAction)) - softQAlphaMatrix.getValue(0, 0) * Math.log(targetPolicyValues.getValue(targetAction, 0));
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

}
