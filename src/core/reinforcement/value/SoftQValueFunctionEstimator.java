package core.reinforcement.value;


import core.NeuralNetworkException;
import core.reinforcement.AgentException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.StateTransition;
import utils.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.util.TreeSet;

/**
 * Class that defines SoftQValueFunctionEstimator (Soft Q value function with target function estimator).<br>
 *
 */
public class SoftQValueFunctionEstimator extends AbstractValueFunctionEstimator {

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
     * Reference to softQ alpha matrix.
     *
     */
    private final Matrix softQAlphaMatrix;

    /**
     * Constructor for SoftQValueFunctionEstimator.
     *
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param functionEstimator reference to FunctionEstimator.
     * @param softQAlphaMatrix reference to softQAlphaMatrix.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    public SoftQValueFunctionEstimator(FunctionEstimator policyFunctionEstimator, FunctionEstimator functionEstimator, Matrix softQAlphaMatrix) throws IOException, ClassNotFoundException, DynamicParamException, MatrixException, NeuralNetworkException {
        super(functionEstimator.getNumberOfActions(), functionEstimator);
        this.policyFunctionEstimator = policyFunctionEstimator;
        functionEstimator.setTargetFunctionEstimator();
        functionEstimator2 = functionEstimator.copy();
        functionEstimator2.reinitialize();
        functionEstimator2.setTargetFunctionEstimator();
        this.softQAlphaMatrix = softQAlphaMatrix;
    }

    /**
     * Constructor for SoftQValueFunctionEstimator.
     *
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param functionEstimator reference to value FunctionEstimator.
     * @param softQAlphaMatrix reference to softQAlphaMatrix.
     * @param params parameters for QTargetValueFunctionEstimator.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    public SoftQValueFunctionEstimator(FunctionEstimator policyFunctionEstimator, FunctionEstimator functionEstimator, Matrix softQAlphaMatrix, String params) throws IOException, ClassNotFoundException, DynamicParamException, MatrixException, NeuralNetworkException {
        super(functionEstimator.getNumberOfActions(), functionEstimator, params);
        this.policyFunctionEstimator = policyFunctionEstimator;
        functionEstimator.setTargetFunctionEstimator();
        functionEstimator2 = functionEstimator.copy();
        functionEstimator2.reinitialize();
        functionEstimator2.setTargetFunctionEstimator();
        this.softQAlphaMatrix = softQAlphaMatrix;
    }

    /**
     * Starts FunctionEstimator
     *
     * @throws NeuralNetworkException throws exception if starting of value FunctionEstimator fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void start() throws NeuralNetworkException, MatrixException, DynamicParamException {
        super.start();
        functionEstimator2.start();
    }

    /**
     * Stops FunctionEstimator
     *
     */
    public void stop() {
        super.stop();
        functionEstimator2.stop();
    }

    /**
     * Updates state value.
     *
     * @param stateTransition state transition.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void updateValue(StateTransition stateTransition) throws NeuralNetworkException, MatrixException {
        double value1 = functionEstimator.predict(stateTransition.environmentState.state).getValue(getAction(stateTransition.action), 0);
        double value2 = functionEstimator2.predict(stateTransition.environmentState.state).getValue(getAction(stateTransition.action), 0);
        stateTransition.value = Math.min(value1, value2);
    }

    /**
     * Resets FunctionEstimator.
     *
     */
    public void resetFunctionEstimator() {
        super.resetFunctionEstimator();
        functionEstimator2.reset();
    }

    /**
     * Updates FunctionEstimator.
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
                Matrix targetValues1 = functionEstimator.predict(stateTransition.environmentState.state).copy();
                targetValues1.setValue(getAction(stateTransition.action), 0, stateTransition.tdTarget);
                functionEstimator.store(stateTransition, targetValues1);
                Matrix targetValues2 = functionEstimator2.predict(stateTransition.environmentState.state).copy();
                targetValues2.setValue(getAction(stateTransition.action), 0, stateTransition.tdTarget);
                functionEstimator2.store(stateTransition, targetValues2);
            }
            functionEstimator.update();
            functionEstimator2.update();
        }

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
        Matrix targetPolicyValues = policyFunctionEstimator.predict(nextStateTransition.environmentState.state);
        int targetAction = policyFunctionEstimator.argmax(targetPolicyValues, nextStateTransition.environmentState.availableActions);
        double value1 = functionEstimator.getTargetFunctionEstimator().predict(nextStateTransition.environmentState.state).getValue(targetAction, 0);
        double value2 = functionEstimator2.getTargetFunctionEstimator().predict(nextStateTransition.environmentState.state).getValue(targetAction, 0);
        return Math.min(value1, value2) - softQAlphaMatrix.getValue(0, 0) * Math.log(targetPolicyValues.getValue(targetAction, 0));
    }

}
