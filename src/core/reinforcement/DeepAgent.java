/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement;

import core.NeuralNetworkException;
import core.reinforcement.policy.ActionableBasicPolicy;
import core.reinforcement.policy.ActionablePolicy;
import core.reinforcement.policy.GreedyPolicy;
import core.reinforcement.value.ValueFunction;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.Serializable;
import java.util.HashMap;
import java.util.TreeMap;

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
    private boolean episodic = true;

    /**
     * Number of episodes.
     *
     */
    private int episodeCount = 0;

    /**
     * Time step of episode.
     *
     */
    private int timeStep = 0;

    /**
     * Reference to current sample.
     *
     */
    private transient RLSample sample;

    /**
     * Reference to current policy.
     *
     */
    protected final ActionablePolicy policy;

    /**
     * Reference to greedy policy.
     *
     */
    private final ActionablePolicy greedyPolicy;

    /**
     * Reference to buffer.
     *
     */
    private final Buffer buffer;

    /**
     * If true agent is learning.
     *
     */
    private boolean isLearning = true;

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
     * Current update count.
     *
     */
    private transient int updateCount = 0;

    /**
     * Constructor for deep agent.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param buffer reference to buffer.
     * @param valueFunction reference to value function.
     */
    public DeepAgent(Environment environment, ActionablePolicy policy, Buffer buffer, ValueFunction valueFunction) {
        this.environment = environment;
        this.policy = policy;
        policy.setEnvironment(environment);
        this.buffer = buffer;
        this.valueFunction = valueFunction;
        greedyPolicy = new ActionableBasicPolicy(new GreedyPolicy(), policy.getFunctionEstimator());
        greedyPolicy.setEnvironment(environment);
    }

    /**
     * Constructor for deep agent.
     *
     * @param environment reference to environment.
     * @param policy reference to policy.
     * @param buffer reference to buffer.
     * @param valueFunction reference to value function.
     * @param params parameters for deep agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public DeepAgent(Environment environment, ActionablePolicy policy, Buffer buffer, ValueFunction valueFunction, String params) throws DynamicParamException {
        this(environment, policy, buffer, valueFunction);
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for DeepAgent.
     *
     * @return parameters used for DeepAgent.
     */
    protected HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("updateCycle", DynamicParam.ParamType.INT);
        paramDefs.put("episodic", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for DeepAgent.<br>
     * <br>
     * Supported parameters are:<br>
     *     - updateCycle: estimator update cycle. Default value 5.<br>
     *     - episodic: If true agent assumes episodic environment. Default value true.<br>
     *
     * @param params parameters used for DeepAgent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("updateCycle")) updateCycle = params.getValueAsInteger("updateCycle");
        if (params.hasParam("episodic")) episodic = params.getValueAsBoolean("episodic");
    }

    /**
     * Starts agent.
     *
     * @throws NeuralNetworkException throws exception if start of neural network estimator(s) fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     */
    public void start() throws NeuralNetworkException, MatrixException {
        policy.start();
        valueFunction.start();
    }

    /**
     * Stops agent.
     *
     */
    public void stop() {
        policy.stop();
        valueFunction.stop();
    }

    /**
     * Starts new episode.
     *
     */
    public void newEpisode() {
        if (episodic) {
            sample = null;
            episodeCount++;
            updateCount++;
            policy.setEpisode(episodeCount);
            valueFunction.setEpisode(episodeCount);
            timeStep = 0;
        }
    }

    /**
     * Begins new episode step for agent.
     *
     */
    public void newStep() {
        if (episodic) timeStep++;
        else {
            episodeCount++;
            updateCount++;
            policy.setEpisode(episodeCount);
            valueFunction.setEpisode(episodeCount);
            timeStep = 1;
            sample = null;
        }
        sample = sample == null ? new RLSample(new State(environment.getState())) : new RLSample(sample.state.getNextState(environment.getState()));
        sample.timeStep = timeStep;
        if (isLearning) bufferSample(sample);
    }

    /**
     * Disables learning.
     *
     */
    public void disableLearning() {
        isLearning = false;
    }

    /**
     * Enables learning.
     *
     */
    public void enableLearning() {
        isLearning = true;
    }

    /**
     * Takes action as defined by agent's policy.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void act() throws NeuralNetworkException, MatrixException, DynamicParamException {
        act(false);
    }

    /**
     * Takes action as defined by agent's policy.
     *
     * @param alwaysGreedy if true greedy action is always taken.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void act(boolean alwaysGreedy) throws NeuralNetworkException, MatrixException, DynamicParamException {
        if (!alwaysGreedy) policy.act(sample);
        else greedyPolicy.act(sample);
        environment.commitAction(this, sample.state.action);
    }

    /**
     * Response from environment to the action taken.<br>
     * If state is final then buffer is used to update policy and value functions of agent.
     *
     * @param reward immediate reward.
     * @param finalState if true state is final otherwise false.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void respond(double reward, boolean finalState) throws NeuralNetworkException, MatrixException, DynamicParamException {
        sample.state.reward = reward;
        sample.state.finalState = !episodic || finalState;

        if (sample.state.isFinalState() && isLearning) {
            if (updateCount >= updateCycle) {
                TreeMap<Integer, RLSample> samples = buffer.getSamples();
                updateAgent(samples, buffer.hasImportanceSamplingWeights());
                buffer.update(samples);
                updateCount = 0;
                buffer.clear();
            }
        }
    }

    /**
     * Buffers current sample.
     *
     * @param currentSample current sample.
     */
    private void bufferSample(RLSample currentSample) {
        if (buffer != null && currentSample != null) buffer.add(currentSample);
    }

    /**
     * Updates policy and value functions of agent.
     *
     * @param samples samples used to update function estimator.
     * @param hasImportanceSamplingWeights if true samples contain importance sampling weights otherwise false.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected abstract void updateAgent(TreeMap<Integer, RLSample> samples, boolean hasImportanceSamplingWeights) throws MatrixException, NeuralNetworkException, DynamicParamException;

}
