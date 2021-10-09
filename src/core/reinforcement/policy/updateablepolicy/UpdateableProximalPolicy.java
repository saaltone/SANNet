/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.policy.updateablepolicy;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.AgentException;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.Policy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Class that defines UpdateableProximalPolicy. Implements Proximal Policy Optimization (PPO).<br>
 *
 */
public class UpdateableProximalPolicy extends AbstractUpdateablePolicy {

    /**
     * Parameter name types for UpdateableProximalPolicy.
     *     - epsilon: epsilon value for proximal policy value clipping. Default value 0.2.<br>
     *     - updateCycle: update cycle for previous estimator function update. Default value 1.<br>
     *
     */
    private final static String paramNameTypes = "(epsilon:DOUBLE), " +
            "(updateCycle:INT)";

    /**
     * Reference to previous policy function estimator.
     *
     */
    private final FunctionEstimator previousFunctionEstimator;

    /**
     * Epsilon value for proximal policy value clipping.
     *
     */
    private double epsilon;

    /**
     * Update cycle for previous function estimator.
     *
     */
    private int updateCycle;

    /**
     * Update count for previous function estimator updates.
     *
     */
    private int updateCount = 0;

    /**
     * Constructor for UpdateableProximalPolicy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to FunctionEstimator.
     * @throws IOException throws exception if creation of FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public UpdateableProximalPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator) throws IOException, ClassNotFoundException, DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator);
        previousFunctionEstimator = functionEstimator.copy();
    }

    /**
     * Constructor for UpdateableProximalPolicy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to FunctionEstimator.
     * @param params parameters for UpdateableProximalPolicy.
     * @throws IOException throws exception if creation of FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public UpdateableProximalPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, String params) throws IOException, ClassNotFoundException, DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator, params);
        previousFunctionEstimator = functionEstimator.copy();
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        epsilon = 0.2;
        updateCycle = 1;
    }

    /**
     * Returns parameters used for UpdateableProximalPolicy.
     *
     * @return parameters used for UpdateableProximalPolicy.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + UpdateableProximalPolicy.paramNameTypes;
    }

    /**
     * Sets parameters used for UpdateableProximalPolicy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - epsilon: epsilon value for proximal policy value clipping. Default value 0.2.<br>
     *     - updateCycle: update cycle for previous estimator function update. Default value 1.<br>
     *
     * @param params parameters used for UpdateableProximalPolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
        if (params.hasParam("epsilon")) epsilon = params.getValueAsDouble("epsilon");
        if (params.hasParam("updateCycle")) updateCycle = params.getValueAsInteger("updateCycle");
    }

    /**
     * Returns reference to policy.
     *
     * @return reference to policy.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public Policy reference() throws DynamicParamException, IOException, ClassNotFoundException, AgentException {
        return new UpdateableProximalPolicy(executablePolicy.getExecutablePolicyType(), functionEstimator, params);
    }

    /**
     * Returns reference to policy.
     *
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to policy.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if optimizer is of an unknown type.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public Policy reference(boolean sharedPolicyFunctionEstimator, boolean sharedMemory) throws DynamicParamException, IOException, ClassNotFoundException, NeuralNetworkException, AgentException {
        return new UpdateableProximalPolicy(executablePolicy.getExecutablePolicyType(), sharedPolicyFunctionEstimator ? functionEstimator : functionEstimator.reference(sharedMemory), params);
    }

    /**
     * Starts FunctionEstimator
     *
     * @throws NeuralNetworkException throws exception if start of neural network estimator(s) fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void start() throws NeuralNetworkException, MatrixException, DynamicParamException {
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
     * Returns policy gradient value for update.
     *
     * @param stateTransition state transition.
     * @return policy gradient value.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected double getPolicyValue(StateTransition stateTransition) throws NeuralNetworkException, MatrixException {
        double currentActionValue = functionEstimator.predict(stateTransition.environmentState.state()).getValue(stateTransition.action, 0);
        double previousActionValue = previousFunctionEstimator.predict(stateTransition.environmentState.state()).getValue(stateTransition.action, 0);
        double rValue = previousActionValue == 0 ? 1 : currentActionValue / previousActionValue;
        double clippedRValue = Math.min(Math.max(rValue, 1 - epsilon), 1 + epsilon);
        return -Math.min(rValue * stateTransition.advantage, clippedRValue * stateTransition.advantage);
    }

    /**
     * Postprocesses policy gradient update.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    protected void postProcess() throws MatrixException, AgentException {
        if (++updateCount >= updateCycle) {
            previousFunctionEstimator.append(functionEstimator, true);
            updateCount = 0;
        }
    }

}
