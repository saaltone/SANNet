/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.policy;

import core.network.NeuralNetworkException;
import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import core.reinforcement.agent.Environment;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.policy.executablepolicy.ExecutablePolicy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyFactory;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import core.reinforcement.value.ValueFunction;
import utils.Configurable;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;

/**
 * Class that defines AbstractPolicy with common policy functions.<br>
 *
 */
public abstract class AbstractPolicy implements Policy, Configurable, Serializable {

    @Serial
    private static final long serialVersionUID = 7604226764648819354L;

    /**
     * Reference to environment.
     *
     */
    protected Environment environment;

    /**
     * Value function for policy.
     *
     */
    protected ValueFunction valueFunction;

    /**
     * Reference to FunctionEstimator.
     *
     */
    protected final FunctionEstimator functionEstimator;

    /**
     * If true function estimator is state action value function.
     *
     */
    private final boolean isStateActionValueFunction;

    /**
     * Reference to executable policy.
     *
     */
    protected final ExecutablePolicy executablePolicy;

    /**
     * If true agent is in learning mode.
     *
     */
    private boolean isLearning = true;

    /**
     * Parameters for policy.
     *
     */
    protected final String params;

    /**
     * Constructor for AbstractPolicy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to FunctionEstimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public AbstractPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator) throws DynamicParamException, AgentException {
        initializeDefaultParams();
        this.executablePolicy = ExecutablePolicyFactory.create(executablePolicyType);
        this.functionEstimator = functionEstimator;
        isStateActionValueFunction = functionEstimator.isStateActionValueFunction();
        if (isStateActionValueFunction && !isUpdateablePolicy()) throw new AgentException("Non-updateable policy cannot be applied along state value function estimator.");
        params = null;
    }

    /**
     * Constructor for AbstractPolicy.
     *
     * @param executablePolicy executable policy.
     * @param functionEstimator reference to FunctionEstimator.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public AbstractPolicy(ExecutablePolicy executablePolicy, FunctionEstimator functionEstimator) throws AgentException {
        initializeDefaultParams();
        this.executablePolicy = executablePolicy;
        this.functionEstimator = functionEstimator;
        isStateActionValueFunction = functionEstimator.isStateActionValueFunction();
        if (isStateActionValueFunction && !isUpdateablePolicy()) throw new AgentException("Non-updateable policy cannot be applied along state value function estimator.");
        params = null;
    }

    /**
     * Constructor for AbstractPolicy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to FunctionEstimator.
     * @param params parameters for AbstractExecutablePolicy.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, String params) throws DynamicParamException, AgentException {
        initializeDefaultParams();
        this.executablePolicy = ExecutablePolicyFactory.create(executablePolicyType);
        this.functionEstimator = functionEstimator;
        isStateActionValueFunction = functionEstimator.isStateActionValueFunction();
        if (isStateActionValueFunction && !isUpdateablePolicy()) throw new AgentException("Non-updateable policy cannot be applied along state value function estimator.");
        this.params = params;
        if (params != null) setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Constructor for AbstractPolicy.
     *
     * @param executablePolicy executable policy.
     * @param functionEstimator reference to FunctionEstimator.
     * @param params parameters for AbstractExecutablePolicy.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractPolicy(ExecutablePolicy executablePolicy, FunctionEstimator functionEstimator, String params) throws AgentException, DynamicParamException {
        initializeDefaultParams();
        this.executablePolicy = executablePolicy;
        this.functionEstimator = functionEstimator;
        isStateActionValueFunction = functionEstimator.isStateActionValueFunction();
        if (isStateActionValueFunction && !isUpdateablePolicy()) throw new AgentException("Non-updateable policy cannot be applied along state value function estimator.");
        this.params = params;
        if (params != null) setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for AbstractPolicy.
     *
     * @return parameters used for AbstractPolicy.
     */
    public String getParamDefs() {
        return executablePolicy.getParamDefs() + ", " + functionEstimator.getParamDefs();
    }

    /**
     * Sets parameters used for AbstractPolicy.<br>
     *
     * @param params parameters used for AbstractPolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        executablePolicy.setParams(params);
        functionEstimator.setParams(params);
    }

    /**
     * Return true is function is state action value function.
     *
     * @return true is function is state action value function.
     */
    public boolean isStateActionValueFunction() {
        return isStateActionValueFunction;
    }

    /**
     * Starts AbstractPolicy.
     *
     * @throws NeuralNetworkException throws exception if start of neural network estimator(s) fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void start() throws NeuralNetworkException, MatrixException, DynamicParamException {
        functionEstimator.start();
    }

    /**
     * Stops AbstractPolicy.
     *
     */
    public void stop() {
        functionEstimator.stop();
    }

    /**
     * Registers agent for FunctionEstimator.
     *
     * @param agent agent.
     */
    public void registerAgent(Agent agent) {
        functionEstimator.registerAgent(agent);
    }

    /**
     * Sets reference to environment.
     *
     * @param environment reference to environment.
     */
    public void setEnvironment(Environment environment) {
        this.environment = environment;
    }

    /**
     * Returns reference to environment.
     *
     * @return reference to environment.
     */
    public Environment getEnvironment() {
        return environment;
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
     * Sets value function for policy-
     *
     * @param valueFunction value function.
     */
    public void setValueFunction(ValueFunction valueFunction) {
        this.valueFunction = valueFunction;
    }

    /**
     * Sets flag if agent is in learning mode.
     *
     * @param isLearning if true agent is in learning mode.
     */
    public void setLearning(boolean isLearning) {
        this.isLearning = isLearning;
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
     * Resets executable policy.
     *
     * @param forceReset forces to trigger reset.
     */
    public void reset(boolean forceReset) {
        executablePolicy.reset(forceReset);
    }

    /**
     * Increments executable policy.
     *
     */
    public void increment() {
        executablePolicy.increment();
    }

    /**
     * Takes action defined by external agent.
     *
     * @param stateTransition state transition.
     * @param action action.
     */
    public void act(StateTransition stateTransition, int action) throws MatrixException, NeuralNetworkException {
        executablePolicy.action(isStateActionValueFunction ? functionEstimator.predict(stateTransition.environmentState.state()).getSubMatrices().get(1) : functionEstimator.predict(stateTransition.environmentState.state()), stateTransition.environmentState.availableActions(), action);
    }

    /**
     * Takes action defined by executable policy.
     *
     * @param stateTransition state transition.
     * @param alwaysGreedy if true greedy action is always taken.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void act(StateTransition stateTransition, boolean alwaysGreedy) throws NeuralNetworkException, MatrixException {
        stateTransition.action = executablePolicy.action(isStateActionValueFunction ? functionEstimator.predict(stateTransition.environmentState.state()).getSubMatrices().get(1) : functionEstimator.predict(stateTransition.environmentState.state()), stateTransition.environmentState.availableActions(), alwaysGreedy);
        if (isLearning()) functionEstimator.add(stateTransition);
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
