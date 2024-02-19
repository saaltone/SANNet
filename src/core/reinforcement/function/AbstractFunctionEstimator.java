/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.function;

import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;
import utils.configurable.Configurable;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashSet;
import java.util.Random;

/**
 * Implements abstract function estimator containing memory management operations and agent handling.<br>
 *
 */
public abstract class AbstractFunctionEstimator implements Configurable, FunctionEstimator, Serializable {

    @Serial
    private static final long serialVersionUID = -557430597852291426L;

    /**
     * Parameters for function estimator.
     *
     */
    private final String params;

    /**
     * Agents registered for function estimator.
     *
     */
    private final HashSet<Agent> registeredAgents = new HashSet<>();

    /**
     * Agents ready for function estimator update.
     *
     */
    private final HashSet<Agent> completedAgents = new HashSet<>();

    /**
     * If true function is combined state action value function.
     *
     */
    protected final boolean isStateActionValueFunction;

    /**
     * Number of states for function estimator.
     *
     */
    protected final int numberOfStates;

    /**
     * Number of actions for function estimator.
     *
     */
    protected final int numberOfActions;

    /**
     * Random function for abstract function estimator policy.
     *
     */
    private final Random random = new Random();

    /**
     * Constructor for abstract function estimator.
     *
     * @param numberOfStates             number of states.
     * @param numberOfActions            number of actions.
     * @param isStateActionValueFunction if true function is combined state action value function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public AbstractFunctionEstimator(int numberOfStates, int numberOfActions, boolean isStateActionValueFunction) throws DynamicParamException, MatrixException {
        this(numberOfStates, numberOfActions, isStateActionValueFunction, null);
    }

    /**
     * Constructor for abstract function estimator.
     *
     * @param numberOfStates             number of states.
     * @param numberOfActions            number of actions.
     * @param isStateActionValueFunction if true function is combined state action value function.
     * @param params                     parameters for function
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public AbstractFunctionEstimator(int numberOfStates, int numberOfActions, boolean isStateActionValueFunction, String params) throws DynamicParamException, MatrixException {
        initializeDefaultParams();
        this.numberOfStates = numberOfStates;
        this.numberOfActions = numberOfActions;
        this.isStateActionValueFunction = isStateActionValueFunction;
        this.params = params;
    }

    /**
     * Returns parameters of function estimator.
     *
     * @return parameters for function estimator.
     */
    protected String getParams() {
        return params;
    }

    /**
     * Returns number of states for abstract function estimator.
     *
     * @return number of states for abstract function estimator.
     */
    public int getNumberOfStates() {
        return numberOfStates;
    }

    /**
     * Returns number of actions for abstract function estimator.
     *
     * @return number of actions for abstract function estimator.
     */
    public int getNumberOfActions() {
        return numberOfActions;
    }

    /**
     * Sets if function estimator can use importance sampling weights.
     *
     * @param canUseImportanceSamplingWeights if true can use importance sampling weights otherwise not.
     */
    public void setCanUseImportanceSamplingWeights(boolean canUseImportanceSamplingWeights) {
    }

    /**
     * Registers agent for abstract function estimator.
     *
     * @param agent agent.
     */
    public void registerAgent(Agent agent) {
        registeredAgents.add(agent);
    }

    /**
     * Notifies that agent is ready to update.
     *
     * @param agent current agent.
     * @return true if all registered agents are ready to update.
     * @throws AgentException throws exception if agent is not registered for function estimator.
     */
    public boolean readyToUpdate(Agent agent) throws AgentException {
        if (!registeredAgents.contains(agent)) throw new AgentException("Agent is not registered for function estimator.");
        completedAgents.add(agent);
        return completedAgents.containsAll(registeredAgents);
    }

    /**
     * Completes abstract function estimator update.
     */
    public void updateComplete() {
        completedAgents.clear();
        reset();
    }

    /**
     * Resets function estimator.
     *
     */
    protected abstract void reset();

    /**
     * If true value function is combined state action value function.
     *
     * @return true if value function is combined state action value function.
     */
    public boolean isStateActionValueFunction() {
        return isStateActionValueFunction;
    }

    /**
     * Returns action with maximum state value given available actions.
     *
     * @param stateValues state values.
     * @param availableActions actions available in state.
     * @return action with maximum state value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public int argmax(Matrix stateValues, HashSet<Integer> availableActions) throws MatrixException {
        if (availableActions == null) return stateValues.argmax()[0];

        int maxAction = -1;
        double maxValue = Double.NEGATIVE_INFINITY;
        for (int action : availableActions) {
            double actionValue = stateValues.getValue(action, 0, 0);
            if (maxValue == Double.NEGATIVE_INFINITY || maxValue < actionValue) {
                maxValue = actionValue;
                maxAction = action;
            }
        }
        return maxAction;
    }

    /**
     * Samples action weighted random choice.
     *
     * @param stateValues state values.
     * @param availableActions actions available in state.
     * @return sampled action.
     * @throws MatrixException        throws exception if matrix operation fails.
     */
    public int sample(Matrix stateValues, HashSet<Integer> availableActions) throws MatrixException {
        if (availableActions == null) return stateValues.sample()[0];

        double valueSum = 0;
        for (Integer action : availableActions) valueSum += stateValues.getValue(action, 0, 0);

        double threshold = valueSum * random.nextDouble();

        valueSum = 0;
        for (Integer action : availableActions) {
            valueSum += stateValues.getValue(action, 0, 0);
            if (valueSum >= threshold) return action;
        }

        return -1;
    }

}
