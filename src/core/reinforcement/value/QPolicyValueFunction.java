package core.reinforcement.value;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.State;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.updateablepolicy.UpdateableQPolicy;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements Q policy value function (Q policy value function with function estimator).<br>
 *
 */
public class QPolicyValueFunction extends AbstractActionValueFunction {

    /**
     * Reference to policy.
     *
     */
    private UpdateableQPolicy updateableQPolicy;

    /**
     * Constructor for Q policy value function.
     *
     * @param functionEstimator reference to function estimator.
     * @param params            parameters for value function.
     */
    public QPolicyValueFunction(FunctionEstimator functionEstimator, String params) {
        super(functionEstimator, params);
    }

    /**
     * Sets updateable Q policy.
     *
     * @param updateableQPolicy updateable Q policy.
     */
    public void setUpdateableQPolicy(UpdateableQPolicy updateableQPolicy) {
        this.updateableQPolicy = updateableQPolicy;
    }

    /**
     * Returns policy function estimator.
     *
     * @return policy function estimator.
     */
    private UpdateableQPolicy getUpdateableQPolicy() {
        return updateableQPolicy;
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
        return new QPolicyValueFunction(sharedValueFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(), getParams());
    }

    /**
     * Returns target value based on next state.
     *
     * @param nextState next state.
     * @return target value based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     */
    public double getTargetValue(State nextState) throws NeuralNetworkException, MatrixException {
        return getUpdateableQPolicy().getFunctionEstimator().predictTargetPolicyValues(nextState).multiply(getTargetValues(nextState, false)).getValue(nextState.targetAction, 0, 0);
    }

    /**
     * Returns target values based on next state.
     *
     * @param state state.
     * @param useDefaultEstimator if true uses default estimator otherwise uses target estimator.
     * @return target value based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     */
    public Matrix getTargetValues(State state, boolean useDefaultEstimator) throws NeuralNetworkException, MatrixException {
        return useDefaultEstimator ? getFunctionEstimator().predictStateActionValues(state) : getFunctionEstimator().predictTargetStateActionValues(state);
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
        return getUpdateableQPolicy().getFunctionEstimator().argmax(getUpdateableQPolicy().getFunctionEstimator().predictTargetPolicyValues(nextState), nextState.environmentState.availableActions());
    }

}
