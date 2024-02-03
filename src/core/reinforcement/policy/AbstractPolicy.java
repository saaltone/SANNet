/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.policy;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.Agent;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.agent.State;
import core.reinforcement.memory.Memory;
import core.reinforcement.policy.executablepolicy.ExecutablePolicy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyFactory;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import utils.configurable.Configurable;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.io.Serial;
import java.io.Serializable;

/**
 * Implements abstract policy with common policy functions.<br>
 *
 */
public abstract class AbstractPolicy implements Policy, Configurable, Serializable {

    @Serial
    private static final long serialVersionUID = 7604226764648819354L;

    /**
     * Reference to executable policy.
     *
     */
    protected final ExecutablePolicy executablePolicy;

    /**
     * Reference to function estimator.
     *
     */
    protected final FunctionEstimator functionEstimator;

    /**
     * Reference to memory.
     *
     */
    protected final Memory memory;

    /**
     * If true agent is in learning mode.
     *
     */
    private transient boolean isLearning = true;

    /**
     * Parameters for policy.
     *
     */
    protected final String params;

    /**
     * Constructor for abstract policy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator    reference to function estimator.
     * @param memory               reference to memory.
     * @param params               parameters for AbstractExecutablePolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public AbstractPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, Memory memory, String params) throws DynamicParamException, MatrixException {
        this(ExecutablePolicyFactory.create(executablePolicyType), functionEstimator, memory, params);
    }

    /**
     * Constructor for abstract policy.
     *
     * @param executablePolicy  executable policy.
     * @param functionEstimator reference to function estimator.
     * @param memory            reference to memory.
     * @param params            parameters for AbstractExecutablePolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public AbstractPolicy(ExecutablePolicy executablePolicy, FunctionEstimator functionEstimator, Memory memory, String params) throws DynamicParamException, MatrixException {
        initializeDefaultParams();
        this.executablePolicy = executablePolicy;
        this.functionEstimator = functionEstimator;
        this.memory = memory;
        this.params = params;
    }

    /**
     * Returns parameters used for abstract policy.
     *
     * @return parameters used for abstract policy.
     */
    public String getParamDefs() {
        return getExecutablePolicy().getParamDefs() + (getFunctionEstimator().getParamDefs() == null ? "" : ", " + getFunctionEstimator().getParamDefs());
    }

    /**
     * Sets parameters used for abstract policy.<br>
     *
     * @param params parameters used for abstract policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        getExecutablePolicy().setParams(params);
        getFunctionEstimator().setParams(params);
    }

    /**
     * Starts abstract policy.
     *
     * @throws NeuralNetworkException throws exception if start of neural network estimator(s) fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void start(Agent agent) throws NeuralNetworkException, MatrixException, DynamicParamException, IOException, ClassNotFoundException {
        getFunctionEstimator().registerAgent(agent);
        getFunctionEstimator().start();
    }

    /**
     * Stops abstract policy.
     *
     * @throws NeuralNetworkException throws exception is neural network is not started.
     */
    public void stop() throws NeuralNetworkException {
        getFunctionEstimator().stop();
    }

    /**
     * Returns executable policy.
     *
     * @return executable policy.
     */
    public ExecutablePolicy getExecutablePolicy() {
        return executablePolicy;
    }

    /**
     * Sets flag if agent is in learning mode.
     *
     * @param isLearning if true agent is in learning mode.
     */
    public void setLearning(boolean isLearning) {
        this.isLearning = isLearning;
        getExecutablePolicy().setLearning(isLearning);
    }

    /**
     * Return flag is policy is in learning mode.
     *
     * @return if true agent is in learning mode.
     */
    public boolean isLearning() {
        return isLearning;
    }

    /**
     * Increments executable policy.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void increment() throws MatrixException {
        getExecutablePolicy().increment();
    }

    /**
     * Returns values for state.
     *
     * @param state state.
     * @return values for state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix getValues(State state) throws MatrixException, NeuralNetworkException {
        return getFunctionEstimator().predictPolicyValues(state);
    }

    /**
     * Takes action defined by external agent.
     *
     * @param state  state.
     * @param action action taken by external agent.
     */
    public void act(State state, int action) throws MatrixException, NeuralNetworkException {
        getExecutablePolicy().action(getValues(state), state.environmentState.availableActions(), action);
    }

    /**
     * Takes action defined by executable policy.
     *
     * @param state state.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     */
    public void act(State state) throws NeuralNetworkException, MatrixException {
        Matrix policyValues = getValues(state);
        state.action = getExecutablePolicy().action(policyValues, state.environmentState.availableActions());
        if (isLearning()) {
            state.policyValues = policyValues;
            state.policyValue = policyValues.getValue(state.action, 0, 0);
            memory.add(state);
            getExecutablePolicy().add(state);
        }
    }

    /**
     * Returns reference to function estimator.
     *
     * @return reference to function estimator.
     */
    public FunctionEstimator getFunctionEstimator() {
        return functionEstimator;
    }

}
