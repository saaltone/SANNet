/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement;

import core.NeuralNetworkException;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.policy.Policy;
import core.reinforcement.value.ValueFunction;
import utils.Configurable;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashMap;

/**
 * Class that defines DeepAgent.<br>
 *
 */
public abstract class DeepAgent implements Agent, Configurable, Serializable {

    @Serial
    private static final long serialVersionUID = -1720953512017473344L;

    /**
     * Reference to environment.
     *
     */
    private Environment environment;

    /**
     * True if environment is episodic.
     *
     */
    private boolean episodic;

    /**
     * If true updates value after each episode is completed.
     *
     */
    protected boolean updateValuePerEpisode = false;

    /**
     * Reference to current state transition.
     *
     */
    private transient StateTransition stateTransition;

    /**
     * Reference to current policy.
     *
     */
    protected Policy policy;

    /**
     * Reference to value function.
     *
     */
    protected ValueFunction valueFunction;

    /**
     * Update cycle in episode steps for function estimator.
     *
     */
    private int agentUpdateCycle = 1;

    /**
     * Average reward for non-episodic learning.
     *
     */
    private double averageReward = Double.NEGATIVE_INFINITY;

    /**
     * Tau value for reward averaging.
     *
     */
    private double rewardTau = 0.9;

    /**
     * Constructor for DeepAgent.
     *
     */
    public DeepAgent() {
    }

    /**
     * Initializes environment for DeepAgent.
     *
     * @param environment reference to environment.
     */
    private void initialize(Environment environment) {
        this.environment = environment;
        this.episodic = environment.isEpisodic();
        if (!episodic) agentUpdateCycle = 10;
    }

    /**
     * Initializes DeepAgent with policy
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     */
    public void initialize(Environment environment, Policy policy) {
        initialize(environment);

        this.policy = policy;
        policy.setEnvironment(environment);
        policy.registerAgent(this);
    }

    /**
     * Initializes DeepAgent with policy
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param params parameters for deep agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void initialize(Environment environment, Policy policy, String params) throws DynamicParamException {
        HashMap<String, DynamicParam.ParamType> nameTypes = new HashMap<>(getParamDefs());
        nameTypes.putAll(policy.getParamDefs());

        initialize(environment, policy);

        DynamicParam dynamicParam = new DynamicParam(params, nameTypes);
        setParams(dynamicParam);
        policy.setParams(dynamicParam);
    }

    /**
     * Intializes DeepAgent with policy and value function.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param valueFunction reference to value function.
     */
    public void initialize(Environment environment, Policy policy, ValueFunction valueFunction) {
        initialize(environment, policy);

        this.valueFunction = valueFunction;
        valueFunction.registerAgent(this);
        policy.setValueFunction(valueFunction);
    }

    /**
     * Intializes DeepAgent with policy and value function.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param valueFunction reference to value function.
     * @param params parameters for deep agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void initialize(Environment environment, Policy policy, ValueFunction valueFunction, String params) throws DynamicParamException {
        HashMap<String, DynamicParam.ParamType> nameTypes = new HashMap<>(getParamDefs());
        nameTypes.putAll(policy.getParamDefs());
        nameTypes.putAll(valueFunction.getParamDefs());

        initialize(environment, policy, valueFunction);

        DynamicParam dynamicParam = new DynamicParam(params, nameTypes);
        setParams(dynamicParam);
        policy.setParams(dynamicParam);
        valueFunction.setParams(dynamicParam);
    }

    /**
     * Returns parameters used for DeepAgent.
     *
     * @return parameters used for DeepAgent.
     */
    public HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("updateValuePerEpisode", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("agentUpdateCycle", DynamicParam.ParamType.INT);
        paramDefs.put("rewardTau", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for DeepAgent.<br>
     * <br>
     * Supported parameters are:<br>
     *     - updateValuePerEpisode: if true updates value after each episode is completed. Default value false.<br>
     *     - agentUpdateCycle: estimator update cycle. Default value 1.<br>
     *     - rewardTau: tau value for reward averaging in non-episodic learning. Default value 0.9.<br>
     *
     * @param params parameters used for DeepAgent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("updateValuePerEpisode")) updateValuePerEpisode = params.getValueAsBoolean("updateValuePerEpisode");
        if (params.hasParam("agentUpdateCycle")) agentUpdateCycle = params.getValueAsInteger("agentUpdateCycle");
        if (params.hasParam("rewardTau")) rewardTau = params.getValueAsDouble("rewardTau");
    }

    /**
     * Starts DeepAgent.
     *
     * @throws NeuralNetworkException throws exception if start of neural network estimator(s) fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void start() throws NeuralNetworkException, MatrixException, DynamicParamException {
        policy.start();
        if (valueFunction != null) valueFunction.start();
    }

    /**
     * Stops DeepAgent.
     *
     */
    public void stop() {
        policy.stop();
        if (valueFunction != null) valueFunction.stop();
    }

    /**
     * Starts new episode.
     *
     */
    public void newEpisode() {
        if (episodic) stateTransition = null;
    }

    /**
     * Begins new episode step for DeepAgent.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update of estimator fails.
     */
    public void newStep() throws MatrixException, DynamicParamException, NeuralNetworkException, AgentException {
        if (!episodic) {
            endEpisode();
            stateTransition = null;
        }
        stateTransition = stateTransition == null ? new StateTransition(environment.getState()) : stateTransition.getNextStateTransition(environment.getState());
    }

    /**
     * Ends episode and if end of update cycle is reached updates DeepAgent.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update of estimator fails.
     */
    public void endEpisode() throws MatrixException, NeuralNetworkException, DynamicParamException, AgentException {
        if (updateValuePerEpisode) valueFunction.update(stateTransition);
        policy.update();
        if (policy.isLearning() && environment.getState().episodeID() > 0 && environment.getState().episodeID() % agentUpdateCycle == 0) update();
    }

    /**
     * Disables learning.
     *
     */
    public void disableLearning() {
        policy.setLearning(false);
    }

    /**
     * Enables learning.
     *
     */
    public void enableLearning() {
        policy.setLearning(true);
    }

    /**
     * Takes action per defined agent policy.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void act() throws NeuralNetworkException, MatrixException {
        act(false);
    }

    /**
     * Takes action per defined agent policy.
     *
     * @param alwaysGreedy if true greedy action is always taken.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void act(boolean alwaysGreedy) throws NeuralNetworkException, MatrixException {
        policy.act(stateTransition, alwaysGreedy);
        environment.commitAction(this, stateTransition.action);
    }

    /**
     * Takes action defined by external agent.
     *
     * @param action action.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void act(int action) throws NeuralNetworkException, MatrixException {
        policy.act(stateTransition, action);
        stateTransition.action = action;
        environment.commitAction(this, stateTransition.action);
    }

    /**
     * Assigns immediate reward from environment in response to action agent executed.
     *
     * @param reward immediate reward.
     */
    public void respond(double reward) {
        stateTransition.reward = reward;
        if (!episodic) {
            averageReward = averageReward == Double.NEGATIVE_INFINITY ? reward : rewardTau * averageReward + (1 - rewardTau) * reward;
            stateTransition.reward -= averageReward;
        }
    }

    /**
     * Updates agent.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if function estimator update fails.
     */
    private void update() throws MatrixException, NeuralNetworkException, DynamicParamException, AgentException {
        updateFunctionEstimator();
        policy.increment();
        policy.reset(false);
    }

    /**
     * Updates policy and value functions of agent.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update of estimator fails.
     */
    protected abstract void updateFunctionEstimator() throws MatrixException, NeuralNetworkException, DynamicParamException, AgentException;

    /**
     * Resets policy.
     *
     */
    public void resetPolicy() {
        policy.reset(true);
    }

}
