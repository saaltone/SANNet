/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement;

import core.NeuralNetworkException;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.policy.Policy;
import core.reinforcement.value.ValueFunction;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Class that defines DeepAgent.
 *
 */
public abstract class DeepAgent implements Agent, Serializable {

    private static final long serialVersionUID = -1720953512017473344L;

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
     * If true updates value for episode otherwise set sampled from memory.
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
    private int updateCycle = 1;

    /**
     * Average reward for non-episodic learning.
     *
     */
    private double averageReward = Double.NEGATIVE_INFINITY;

    /**
     * Tau value for reward averaging.
     *
     */
    private double tau = 0.9;

    /**
     * Constructor for deep agent.
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param valueFunction reference to value function.
     */
    public DeepAgent(Environment environment, Policy policy, ValueFunction valueFunction) {
        this.environment = environment;
        this.episodic = environment.isEpisodic();
        if (!episodic) updateCycle = 10;
        this.policy = policy;
        policy.setEnvironment(environment);
        policy.getFunctionEstimator().registerAgent(this);
        this.valueFunction = valueFunction;
        if (!valueFunction.getFunctionEstimator().isStateActionValue()) valueFunction.getFunctionEstimator().registerAgent(this);
    }

    /**
     * Constructor for deep agent.
     * @param environment reference to environment.
     * @param policy reference to policy.
     */
    public DeepAgent(Environment environment, Policy policy) {
        this.environment = environment;
        this.episodic = environment.isEpisodic();
        if (!episodic) updateCycle = 10;
        this.policy = policy;
        policy.setEnvironment(environment);
        policy.getFunctionEstimator().registerAgent(this);
        valueFunction = null;
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
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Constructor for DeepAgent.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param params parameters for deep agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public DeepAgent(Environment environment, Policy policy, String params) throws DynamicParamException {
        this(environment, policy);
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for DeepAgent.
     *
     * @return parameters used for DeepAgent.
     */
    protected HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("updateValuePerEpisode", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("updateCycle", DynamicParam.ParamType.INT);
        paramDefs.put("tau", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for DeepAgent.<br>
     * <br>
     * Supported parameters are:<br>
     *     - updateValuePerEpisode: if true updates value for episode otherwise set sampled from memory. Default value false.<br>
     *     - updateCycle: estimator update cycle. Default value 5.<br>
     *     - tau: tau value for reward averaging in non-episodic learning. Default value 0.99.<br>
     *
     * @param params parameters used for DeepAgent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("updateValuePerEpisode")) updateValuePerEpisode = params.getValueAsBoolean("updateValuePerEpisode");
        if (params.hasParam("updateCycle")) updateCycle = params.getValueAsInteger("updateCycle");
        if (params.hasParam("tau")) tau = params.getValueAsDouble("tau");
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
     * Begins new episode step for agent.
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
     * Ends episode and if end of update cycle is reached updates agent.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if update of estimator fails.
     */
    public void endEpisode() throws MatrixException, NeuralNetworkException, DynamicParamException, AgentException {
        if (updateValuePerEpisode) valueFunction.update(stateTransition);
        policy.update();
        if (policy.isLearning() && environment.getState().episodeID > 0 && environment.getState().episodeID % updateCycle == 0) update();
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
            averageReward = averageReward == Double.NEGATIVE_INFINITY ? reward : tau * averageReward + (1 - tau) * reward;
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
    protected abstract void update() throws MatrixException, NeuralNetworkException, DynamicParamException, AgentException;

    /**
     * Resets policy.
     *
     */
    public void resetPolicy() {
        policy.reset(true);
    }

}
