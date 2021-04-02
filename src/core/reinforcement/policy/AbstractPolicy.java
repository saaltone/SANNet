package core.reinforcement.policy;

import core.NeuralNetworkException;
import core.reinforcement.Agent;
import core.reinforcement.AgentException;
import core.reinforcement.Environment;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.policy.executablepolicy.ExecutablePolicy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyFactory;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import core.reinforcement.value.ValueFunction;
import utils.Configurable;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Class that defines AbstractPolicy with common policy functions.
 *
 */
public abstract class AbstractPolicy implements Policy, Configurable, Serializable {

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
    protected final boolean isStateActionValueFunction;

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
     * Constructor for AbstractPolicy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to FunctionEstimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if creation of executable policy fails.
     */
    public AbstractPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator) throws DynamicParamException, AgentException {
        this.executablePolicy = ExecutablePolicyFactory.create(executablePolicyType);
        this.functionEstimator = functionEstimator;
        isStateActionValueFunction = functionEstimator.isStateActionValueFunction();
    }

    /**
     * Constructor for AbstractPolicy.
     *
     * @param executablePolicy executable policy.
     * @param functionEstimator reference to FunctionEstimator.
     */
    public AbstractPolicy(ExecutablePolicy executablePolicy, FunctionEstimator functionEstimator) {
        this.executablePolicy = executablePolicy;
        this.functionEstimator = functionEstimator;
        isStateActionValueFunction = functionEstimator.isStateActionValueFunction();
    }

    /**
     * Returns parameters used for AbstractPolicy.
     *
     * @return parameters used for AbstractPolicy.
     */
    public HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.putAll(executablePolicy.getParamDefs());
        paramDefs.putAll(functionEstimator.getParamDefs());
        return paramDefs;
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
     * Sets value function for policy-
     *
     * @param valueFunction value function.
     */
    public void setValueFunction(ValueFunction valueFunction) {
        this.valueFunction = valueFunction;
    }

    /**
     * Set flag if agent is in learning mode.
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
     * If true value function is combined state action value function.
     *
     * @return true if value function is combined state action value function.
     */
    protected int isStateActionValueFunction() {
        return isStateActionValueFunction ? 1 : 0;
    }

    /**
     * Returns action with potential state action value offset.
     *
     * @param action action.
     * @return updated action.
     */
    protected int getAction(int action) {
        return isStateActionValueFunction() + action;
    }

    /**
     * Takes action defined by external agent.
     *
     * @param stateTransition state transition.
     * @param action action.
     */
    public void act(StateTransition stateTransition, int action) throws MatrixException, NeuralNetworkException {
        executablePolicy.action(functionEstimator.predict(stateTransition.environmentState.state), stateTransition.environmentState.availableActions, isStateActionValueFunction(), action);
    }

    /**
     * Takes action by applying defined executable policy.
     *
     * @param stateTransition state transition.
     * @param alwaysGreedy if true greedy action is always taken.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void act(StateTransition stateTransition, boolean alwaysGreedy) throws NeuralNetworkException, MatrixException {
        Matrix currentPolicyValues = functionEstimator.predict(stateTransition.environmentState.state);
        stateTransition.action = executablePolicy.action(currentPolicyValues, stateTransition.environmentState.availableActions, isStateActionValueFunction(), alwaysGreedy);
        if (isLearning()) functionEstimator.add(stateTransition);
    }

}
