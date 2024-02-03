/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.policy.updateablepolicy;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.Agent;
import core.reinforcement.agent.State;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.Memory;
import core.reinforcement.policy.Policy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
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
     * @param functionEstimator    reference to function estimator.
     * @param memory               reference to memory.
     * @param params               parameters for updateable proximal policy.
     * @throws IOException            throws exception if creation of function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of function estimator fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     */
    public UpdateableProximalPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, Memory memory, String params) throws IOException, ClassNotFoundException, DynamicParamException, MatrixException {
        super(executablePolicyType, functionEstimator, memory, params);
        previousFunctionEstimator = functionEstimator.reference();
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
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param memory                        if true policy will use shared memory otherwise dedicated memory.
     * @return reference to policy.
     * @throws IOException            throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     */
    public Policy reference(boolean sharedPolicyFunctionEstimator, Memory memory) throws DynamicParamException, IOException, ClassNotFoundException, MatrixException {
        return new UpdateableProximalPolicy(executablePolicy.getExecutablePolicyType(), sharedPolicyFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(), memory, params);
    }

    /**
     * Returns reference to policy.
     *
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param memory                  if true policy will use shared memory otherwise dedicated memory.
     * @return reference to policy.
     * @throws IOException            throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     */
    public Policy reference(FunctionEstimator policyFunctionEstimator, Memory memory) throws DynamicParamException, IOException, ClassNotFoundException, MatrixException {
        return new UpdateableProximalPolicy(executablePolicy.getExecutablePolicyType(), policyFunctionEstimator, memory, params);
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
     * @throws NeuralNetworkException throws exception is neural network is not started.
     */
    public void stop() throws NeuralNetworkException {
        super.stop();
        getPreviousFunctionEstimator().stop();
    }

    /**
     * Returns policy gradient value for update.
     *
     * @param state state.
     * @return policy gradient value.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     */
    protected Matrix getPolicyGradient(State state) throws NeuralNetworkException, MatrixException {
        if (getPreviousFunctionEstimator() == getFunctionEstimator()) return state.policyValues.getNewMatrix(1);
        else {
            double currentPolicyValue = getFunctionEstimator().predictPolicyValues(state).getValue(state.action, 0, 0);
            double previousPolicyValue = getPreviousFunctionEstimator().predictPolicyValues(state).getValue(state.action, 0, 0);
            double policyGradientValue = Math.min(currentPolicyValue / previousPolicyValue * state.tdError, Math.min(1 + epsilon, Math.max(1 - epsilon, state.tdError)));
            Matrix policyGradient = state.policyValues.getNewMatrix();
            policyGradient.setValue(state.action, 0, 0, policyGradientValue);
            return policyGradient;
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
