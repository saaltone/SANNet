/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.agent;

import core.network.NeuralNetworkException;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.policy.Policy;
import core.reinforcement.value.ValueFunction;
import utils.Configurable;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;

/**
 * Class that defines DeepAgent.<br>
 *
 */
public abstract class DeepAgent implements Agent, Configurable, Serializable {

    @Serial
    private static final long serialVersionUID = -1720953512017473344L;

    /**
     * Parameter name types for DeepAgent.
     *     - updateValuePerEpisode: if true updates value after each episode is completed. Default value false.<br>
     *     - agentUpdateCycle: estimator update cycle. Default value 1.<br>
     *     - rewardTau: tau value for reward averaging in non-episodic learning. Default value 0.9.<br>
     *
     */
    private final static String paramNameTypes = "(updateValuePerEpisode:BOOLEAN), " +
            "(agentUpdateCycle:INT), " +
            "(rewardTau:DOUBLE)";

    /**
     * Parameters for deep agent.
     *
     */
    private String params = null;

    /**
     * Reference to environment.
     *
     */
    private final Environment environment;

    /**
     * True if environment is episodic.
     *
     */
    private final boolean episodic;

    /**
     * If true updates value after each episode is completed.
     *
     */
    protected boolean updateValuePerEpisode;

    /**
     * Reference to current state transition.
     *
     */
    private transient StateTransition stateTransition;

    /**
     * Reference to current policy.
     *
     */
    protected final Policy policy;

    /**
     * Reference to value function.
     *
     */
    protected final ValueFunction valueFunction;

    /**
     * Update cycle in episode steps for function estimator.
     *
     */
    private int agentUpdateCycle;

    /**
     * Average reward for non-episodic learning.
     *
     */
    private double averageReward = Double.NEGATIVE_INFINITY;

    /**
     * Tau value for reward averaging.
     *
     */
    private double rewardTau;

    /**
     * Constructor for DeepAgent.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param valueFunction reference to value function.
     */
    public DeepAgent(Environment environment, Policy policy, ValueFunction valueFunction) {
        initializeDefaultParams();

        this.environment = environment;
        this.episodic = environment.isEpisodic();
        if (!episodic) agentUpdateCycle = 10;

        this.policy = policy;
        policy.registerAgent(this);
        policy.setEnvironment(environment);

        this.valueFunction = valueFunction;
        valueFunction.registerAgent(this);
        policy.setValueFunction(valueFunction);
    }

    /**
     * Constructor for DeepAgent.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param valueFunction reference to value function.
     * @param params parameters for deep agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public DeepAgent(Environment environment, Policy policy, ValueFunction valueFunction, String params) throws DynamicParamException {
        this(environment, policy, valueFunction);

        if (params != null) {
            this.params = params;
            DynamicParam dynamicParam = new DynamicParam(params, getParamDefs() + ", " + policy.getParamDefs() + ", " + valueFunction.getParamDefs());
            setParams(dynamicParam);
            policy.setParams(dynamicParam);
            valueFunction.setParams(dynamicParam);
        }
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        updateValuePerEpisode = false;
        agentUpdateCycle = 1;
        rewardTau = 0.9;
    }

    /**
     * Returns parameters used for DeepAgent.
     *
     * @return parameters used for DeepAgent.
     */
    public String getParamDefs() {
        return DeepAgent.paramNameTypes;
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
     * Returns parameters of deep agent.
     *
     * @return parameters of deep agent.
     */
    protected String getParams() {
        return params;
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
