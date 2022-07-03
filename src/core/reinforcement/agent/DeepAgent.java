/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.reinforcement.agent;

import core.network.NeuralNetworkException;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.policy.Policy;
import core.reinforcement.value.ValueFunction;
import utils.configurable.Configurable;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.io.Serial;
import java.io.Serializable;

/**
 * Implements deep agent.<br>
 *
 */
public abstract class DeepAgent implements Agent, Configurable, Serializable {

    @Serial
    private static final long serialVersionUID = -1720953512017473344L;

    /**
     * Parameter name types for deep agent.
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
     * Constructor for deep agent.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param valueFunction reference to value function.
     */
    public DeepAgent(Environment environment, Policy policy, ValueFunction valueFunction) {
        this.environment = environment;
        this.episodic = environment.isEpisodic();

        this.policy = policy;
        policy.registerAgent(this);

        this.valueFunction = valueFunction;
        valueFunction.registerAgent(this);

        initializeDefaultParams();
    }

    /**
     * Constructor for deep agent.
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
        agentUpdateCycle = episodic ? 1 : 10;
        rewardTau = 0.9;
    }

    /**
     * Returns parameters used for deep agent.
     *
     * @return parameters used for deep agent.
     */
    public String getParamDefs() {
        return DeepAgent.paramNameTypes;
    }

    /**
     * Sets parameters used for deep agent.<br>
     * <br>
     * Supported parameters are:<br>
     *     - updateValuePerEpisode: if true updates value after each episode is completed. Default value false.<br>
     *     - agentUpdateCycle: estimator update cycle. Default value 1.<br>
     *     - rewardTau: tau value for reward averaging in non-episodic learning. Default value 0.9.<br>
     *
     * @param params parameters used for deep agent.
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
     * Starts deep agent.
     *
     * @throws NeuralNetworkException throws exception if start of neural network estimator(s) fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws IOException throws exception if creation of FunctionEstimator copy fails.
     * @throws ClassNotFoundException throws exception if creation of FunctionEstimator copy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void start() throws NeuralNetworkException, MatrixException, DynamicParamException, IOException, ClassNotFoundException {
        policy.start();
        if (valueFunction != null) valueFunction.start();
    }

    /**
     * Stops deep agent.
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
     * Begins new episode step for deep agent.
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
     * Ends episode and if end of update cycle is reached updates deep agent.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update of estimator fails.
     */
    public void endEpisode() throws MatrixException, NeuralNetworkException, DynamicParamException, AgentException {
        if (updateValuePerEpisode) valueFunction.update(stateTransition);

        policy.endEpisode();
        if (policy.isLearning() && environment.getState().episodeID() > 0 && environment.getState().episodeID() % agentUpdateCycle == 0) {
            updateFunctionEstimator();
            policy.increment();
        }
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
        stateTransition.action = action;
        policy.act(stateTransition);
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
     * Updates policy and value functions of agent.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update of estimator fails.
     */
    protected abstract void updateFunctionEstimator() throws MatrixException, NeuralNetworkException, DynamicParamException, AgentException;

}
