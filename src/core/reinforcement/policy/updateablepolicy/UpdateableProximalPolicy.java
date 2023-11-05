/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.policy.updateablepolicy;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.State;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.Memory;
import core.reinforcement.policy.Policy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.ComputableMatrix;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements updateable proximal policy. Implements Proximal Policy Optimization (PPO).<br>
 *
 */
public class UpdateableProximalPolicy extends AbstractUpdateablePolicy {

    /**
     * Parameter name types for updateable proximal policy.
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
     * Constructor for updateable proximal policy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to function estimator.
     * @throws IOException throws exception if creation of function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public UpdateableProximalPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator) throws IOException, ClassNotFoundException, DynamicParamException, AgentException, MatrixException {
        this(executablePolicyType, functionEstimator, null);
    }

    /**
     * Constructor for updateable proximal policy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to function estimator.
     * @param params parameters for updateable proximal policy.
     * @throws IOException throws exception if creation of function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public UpdateableProximalPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, String params) throws IOException, ClassNotFoundException, DynamicParamException, AgentException, MatrixException {
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
     * Returns parameters used for updateable proximal policy.
     *
     * @return parameters used for updateable proximal policy.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + UpdateableProximalPolicy.paramNameTypes;
    }

    /**
     * Sets parameters used for updateable proximal policy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - epsilon: epsilon value for proximal policy value clipping. Default value 0.2.<br>
     *     - updateCycle: update cycle for previous estimator function update. Default value 1.<br>
     *
     * @param params parameters used for updateable proximal policy.
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
     * @param valueFunctionEstimator value function estimator.
     * @return reference to policy.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Policy reference(FunctionEstimator valueFunctionEstimator) throws DynamicParamException, IOException, ClassNotFoundException, AgentException, MatrixException {
        return new UpdateableProximalPolicy(executablePolicy.getExecutablePolicyType(), getFunctionEstimator(), params);
    }

    /**
     * Returns reference to policy.
     *
     * @param valueFunctionEstimator        value function estimator.
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param sharedMemory                  if true shared memory is used between estimators.
     * @return reference to policy.
     * @throws IOException            throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     * @throws AgentException         throws exception if state action value function is applied to non-updateable policy.
     */
    public Policy reference(FunctionEstimator valueFunctionEstimator, boolean sharedPolicyFunctionEstimator, boolean sharedMemory) throws DynamicParamException, IOException, ClassNotFoundException, AgentException, MatrixException {
        return new UpdateableProximalPolicy(executablePolicy.getExecutablePolicyType(), sharedPolicyFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(sharedMemory), params);
    }

    /**
     * Returns reference to policy.
     *
     * @param valueFunctionEstimator        reference to value function estimator.
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param memory                  if true policy will use shared memory otherwise dedicated memory.
     * @return reference to policy.
     * @throws IOException            throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     * @throws AgentException         throws exception if state action value function is applied to non-updateable policy.
     */
    public Policy reference(FunctionEstimator valueFunctionEstimator, boolean sharedPolicyFunctionEstimator, Memory memory) throws DynamicParamException, IOException, ClassNotFoundException, AgentException, MatrixException {
        return new UpdateableProximalPolicy(executablePolicy.getExecutablePolicyType(), sharedPolicyFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(memory), params);
    }

    /**
     * Returns previous function estimator.
     *
     * @return previous function estimator.
     */
    private FunctionEstimator getPreviousFunctionEstimator() {
        return previousFunctionEstimator;
    }

    /**
     * Starts function estimator
     *
     * @throws NeuralNetworkException throws exception if start of neural network estimator(s) fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void start(Agent agent) throws NeuralNetworkException, MatrixException, DynamicParamException, IOException, ClassNotFoundException {
        super.start(agent);
        getPreviousFunctionEstimator().start();
    }

    /**
     * Stops function estimator
     *
     */
    public void stop() {
        super.stop();
        getPreviousFunctionEstimator().stop();
    }

    /**
     * Resets function estimator.
     *
     */
    public void resetFunctionEstimator() {
        super.resetFunctionEstimator();
        if(getPreviousFunctionEstimator() != null) getPreviousFunctionEstimator().reset();
    }

    /**
     * Returns policy gradient value for update.
     *
     * @param state state.
     * @return policy gradient value.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected double getPolicyValue(State state) throws NeuralNetworkException, MatrixException {
        if (getPreviousFunctionEstimator() == getFunctionEstimator()) return state.tdError;
        else {
            double currentActionValue = getFunctionEstimator().predictPolicyValues(state).getValue(state.action, 0, 0);
            double previousActionValue = getPreviousFunctionEstimator().predictPolicyValues(state).getValue(state.action, 0, 0);
            double rValue = previousActionValue != 0 ? currentActionValue / previousActionValue : 1;
            return Math.min(rValue, ComputableMatrix.clipValue(rValue, 1 - epsilon, 2 + epsilon)) * state.tdError;
        }
    }

    /**
     * Postprocesses policy gradient update.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void postProcess() throws MatrixException {
        if (++updateCount >= updateCycle) {
            getPreviousFunctionEstimator().append(getFunctionEstimator());
            updateCount = 0;
        }
    }

}
