/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import core.reinforcement.agent.StateTransition;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;

import java.io.Serial;
import java.io.Serializable;
import java.util.*;

/**
 * Implements MCTS policy.<br>
 * <br>
 * Reference: <a href="https://medium.com/@jonathan_hui/monte-carlo-tree-search-mcts-in-alphago-zero-8a403588276a">...</a> <br>
 * Reference: <a href="https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5">...</a> <br>
 * Reference: <a href="https://lczero.org/blog/2018/12/alphazero-paper-and-lc0-v0191/">...</a> <br>
 * Reference: <a href="https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py">...</a> <br>
 *
 */
public class MCTSPolicy implements ExecutablePolicy, Serializable {

    @Serial
    private static final long serialVersionUID = -6567362723286339425L;

    /**
     * Parameter name types for MCTS policy.
     *     - cPUCT: Constant value for controlling amount of exploration i.e. C value of polynomial upper confidence tree. Default value 2.5.<br>
     *     - alpha: Shape value for Dirichlet sampling. Default value 0.6.<br>
     *     - epsilon: Weighting for Dirichlet distribution at action selection. Default value 0.8.<br>
     *     - tau: Temperature value for node visit count. Default value 1.1.<br>
     *     - resetCycle: Reset cycle counted as number of increments if cycle is 0 then reset is never applied. Default value 0.<br>
     *
     */
    private final static String paramNameTypes = "(cPUCT:DOUBLE), " +
            "(alpha:DOUBLE), " +
            "(epsilon:DOUBLE), " +
            "(tau:DOUBLE), " +
            "(resetCycle:INT)";

    /**
     * Executable policy type.
     *
     */
    private final ExecutablePolicyType executablePolicyType = ExecutablePolicyType.MCTS;

    /**
     * Implements action for state.
     *
     */
    private class Action {

        /**
         * Action ID.
         *
         */
        private final int actionID;

        /**
         * Action value.
         *
         */
        private double actionValue;

        /**
         * Action probability.
         *
         */
        private double actionProbability = 0;

        /**
         * Reference to parent state.
         *
         */
        private final State parentState;

        /**
         * Reference to child state.
         *
         */
        private final State childState;

        /**
         * Constructor for action.
         *
         * @param parentState parent state
         * @param actionID action ID
         */
        Action(State parentState, int actionID) {
            this.parentState = parentState;
            this.actionID = actionID;
            childState = new State(parentState);
        }

        /**
         * Constructor for action.
         *
         * @param parentState parent state
         * @param actionID action ID
         * @param actionProbability action probability.
         */
        Action(State parentState, int actionID, double actionProbability) {
            this(parentState, actionID);
            this.actionProbability = actionProbability;
        }

        /**
         * Returns parent state.
         *
         * @return parent state.
         */
        State getParentState() {
            return parentState;
        }

        /**
         * Returns child state.
         *
         * @return child state.
         */
        State getChildState() {
            return childState;
        }

        /**
         * Returns action ID.
         *
         * @return action ID.
         */
        int getActionID() {
            return actionID;
        }

        /**
         * Sets action probability.
         *
         * @param actionProbability action probability.
         */
        void setActionProbability(double actionProbability) {
            this.actionProbability = actionProbability;
        }

        /**
         * Returns action probability.
         *
         * @return action probability.
         */
        double getActionProbability() {
            return actionProbability;
        }

        /**
         * Updates action value
         *
         */
        void updateActionValue(double stateValue) {
            actionValue = getChildState().hasMaxAction() ? actionValue + (getActionProbability() + stateValue - actionValue) / (1 + (double)getChildState().getVisitCount()) : stateValue;
        }

        /**
         * Returns action value.
         *
         * @return action value.
         */
        double getActionValue() {
            return actionValue;
        }

        /**
         * Returns policy value.
         *
         * @return policy value.
         */
        double getPolicyValue() {
            return getChildState().hasMaxAction() ? Math.pow(getChildState().getVisitCount(), 1 / tau) / Math.pow(getParentState().getVisitCount(), 1 / tau) : 0;
        }

        /**
         * Returns parent visit count.
         *
         * @return parent visit count.
         */
        int getParentStateVisitCount() {
            return getParentState().getVisitCount();
        }

        /**
         * Returns child state visit count.
         *
         * @return child state visit count.
         */
        int getChildStateVisitCount() {
            return getChildState().getVisitCount();
        }

        /**
         * Returns polynomial upper confidence tree loss for action selection.
         *
         * @return polynomial upper confidence tree loss for action selection.
         */
        double getPUCT(double dirichletDistributionValue, double epsilon) {
            return getActionValue() + cPUCT * (epsilon * getActionProbability() + (1 - epsilon) * dirichletDistributionValue) * Math.sqrt(getParentStateVisitCount()) / (1 + getChildStateVisitCount());
        }

    }

    /**
     * Implements state for MCTS.
     *
     */
    private class State {

        /**
         * Parent state for state;
         *
         */
        private final State parentState;

        /**
         * Visit count for state.
         *
         */
        private int visitCount = 0;

        /**
         * Actions for state.
         *
         */
        private final HashMap<Integer, Action> actions = new HashMap<>();

        /**
         * Reference to action with maximum value.
         *
         */
        private Action maxAction = null;

        /**
         * Constructor for state.
         *
         */
        State() {
            parentState = null;
        }

        /**
         * Constructor for state.
         *
         * @param parentState parent state.
         */
        State(State parentState) {
            this.parentState = parentState;
        }

        /**
         * Returns child state based on max action.
         *
         * @return child state based on max action.
         */
        State getNextState() {
            return maxAction.getChildState();
        }

        /**
         * Returns true if state has parent state otherwise false.
         *
         * @return true if state has parent state otherwise false.
         */
        boolean hasParentState() {
            return getParentState() != null;
        }

        /**
         * Returns parent state.
         *
         * @return parent state.
         */
        State getParentState() {
            return parentState;
        }

        /**
         * Returns true if state has max action otherwise false.
         *
         * @return true if state has max action otherwise false.
         */
        boolean hasMaxAction() {
            return maxAction != null;
        }

        /**
         * Returns max action.
         *
         * @return max action.
         */
        Action getMaxAction() {
            return maxAction;
        }

        /**
         * Returns visit count.
         *
         * @return visit count.
         */
        int getVisitCount() {
            return visitCount;
        }

        /**
         * Increments visit count by one.
         *
         */
        void incrementVisitCount() {
            visitCount++;
        }

        /**
         * Takes action decided by external agent.
         *
         * @param stateValueMatrix current state value matrix.
         * @param availableActions available actions in current state
         * @param action action.
         */
        void act(Matrix stateValueMatrix, HashSet<Integer> availableActions, int action) {
            updateActionProbabilities(stateValueMatrix, availableActions);
            maxAction = actions.get(action);
        }

        /**
         * Takes action based on heuristic value. Adds noise to UCB value from Dirichlet distribution.
         *
         * @param policyValueMatrix current policy value matrix.
         * @param availableActions available actions in current state
         * @param alwaysGreedy if true greedy action is always taken.
         * @return action taken.
         */
        public int act(Matrix policyValueMatrix, HashSet<Integer> availableActions, boolean alwaysGreedy) {
            updateActionProbabilities(policyValueMatrix, availableActions);
            maxAction = null;
            double maxValue = Double.NEGATIVE_INFINITY;
            if (alwaysGreedy) {
                for (Integer actionID : availableActions) {
                    Action action = actions.get(actionID);
                    double currentValue = action.getPolicyValue();
                    if (maxValue == Double.NEGATIVE_INFINITY || maxValue < currentValue) {
                        maxValue = currentValue;
                        maxAction = action;
                    }
                }
            }
            else {
                incrementVisitCount();
                HashMap<Integer, Double> dirichletDistribution = getDirichletDistribution(alpha, availableActions);
                for (Integer actionID : availableActions) {
                    Action action = actions.get(actionID);
                    double currentValue = action.getPUCT(dirichletDistribution.get(action.actionID), epsilon);
                    if (maxValue == Double.NEGATIVE_INFINITY || maxValue < currentValue) {
                        maxValue = currentValue;
                        maxAction = action;
                    }
                }
            }
            return (maxAction != null) ? maxAction.getActionID() : -1;
        }

        /**
         * Returns Dirichlet distribution.<br>
         * Reference: <a href="https://stats.stackexchange.com/questions/69210/drawing-from-dirichlet-distribution">...</a><br>
         *
         * @param shape shape parameter.
         * @param availableActions available actions.
         * @return Dirichlet distribution.
         */
        private HashMap<Integer, Double> getDirichletDistribution(double shape, HashSet<Integer> availableActions) {
            // Computes log(sum(exp(elements across dimensions of a tensor))).
            HashMap<Integer, Double> dirichletDistribution = new HashMap<>();
            double cumulativeValue = 0;
            for (Integer action : availableActions) {
                double gammaValue = sampleGamma(shape, 1);
                cumulativeValue += gammaValue;
                dirichletDistribution.put(action, gammaValue);
            }
            for (Map.Entry<Integer, Double> entry : dirichletDistribution.entrySet()) {
                int action = entry.getKey();
                double value = entry.getValue();
                dirichletDistribution.put(action, value / cumulativeValue);
            }
            return dirichletDistribution;
        }

        /**
         * Samples random variable from gamma distribution.<br>
         * Reference: <a href="https://www.hongliangjie.com/2012/12/19/how-to-generate-gamma-random-variables/">...</a>
         *
         * @param shape shape (alpha) parameter
         * @param scale scale (beta) parameter
         * @return random variable from gamma distribution
         */
        private double sampleGamma(double shape, double scale) {
            if (shape > 1) {
                double d = shape - 1 / (double)3;
                double c = 1 / Math.sqrt(9 * d);
                while (true) {
                    double gaussian = random.nextGaussian();
                    if (gaussian > - 1 / c) {
                        double uniform = random.nextDouble();
                        double V = Math.pow(1 + c * gaussian, 3);
                        if (Math.log(uniform) < 0.5 * Math.pow(gaussian, 2) + d - d * V + d * Math.log(V)) return d * V / scale;
                    }
                }
            }
            else return sampleGamma(shape + 1, scale) * Math.pow(random.nextDouble(), 1 / shape);
        }

        /**
         * Updates action probabilities for state.
         *
         * @param policyValueMatrix current state value matrix.
         * @param availableActions available actions in current state
         */
        private void updateActionProbabilities(Matrix policyValueMatrix, HashSet<Integer> availableActions) {
            for (Integer action : availableActions) {
                double actionValue = policyValueMatrix.getValue(action, 0, 0);
                MCTSPolicy.Action mctsAction = actions.get(action);
                if (mctsAction != null) mctsAction.setActionProbability(actionValue);
                else actions.put(action, new Action(this, action, actionValue));
            }
            double cumulativeValue = 0;
            for (Action action : actions.values()) cumulativeValue += action.getActionProbability();
            if (cumulativeValue > 0) for (Action action : actions.values()) action.setActionProbability(action.getActionProbability() / cumulativeValue);
        }

        /**
         * Updates state action value chain from leaf state towards root state.
         *
         * @param stateTransitionStack state transition stack.
         */
        public void update(Stack<StateTransition> stateTransitionStack) {
            StateTransition stateTransition = stateTransitionStack.pop();
            getMaxAction().updateActionValue(stateTransition.tdTarget);
            stateTransition.value = getMaxAction().getPolicyValue();
            if (hasParentState()) getParentState().update(stateTransitionStack);
        }

    }

    /**
     * Constant value for controlling amount of exploration i.e. C value of polynomial upper confidence tree.
     *
     */
    private double cPUCT;

    /**
     * Shape value for Dirichlet sampling.
     *
     */
    private double alpha;

    /**
     * Weighting for Dirichlet distribution at action selection.
     *
     */
    private double epsilon;

    /**
     * Temperature value for node visit count.
     *
     */
    private double tau;

    /**
     * Reference to root state.
     *
     */
    public State rootState;

    /**
     * Reference to current state.
     *
     */
    public State currentState;

    /**
     * Stack to store state transitions for actions taken.
     *
     */
    private Stack<StateTransition> stateTransitionStack = new Stack<>();

    /**
     * Random number generator.
     *
     */
    private final Random random = new Random();

    /**
     * Reset count.
     *
     */
    private int resetCount = 0;

    /**
     * Reset cycle. If value is zero there are no resets applied.
     *
     */
    private int resetCycle;

    /**
     * If true agent is in learning mode.
     *
     */
    private boolean isLearning = true;

    /**
     * Constructor for MCTS policy
     *
     */
    public MCTSPolicy() {
        initializeDefaultParams();
    }

    /**
     * Constructor for MCTS policy
     *
     * @param params parameters for MCTS policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public MCTSPolicy(String params) throws DynamicParamException {
        this();
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        cPUCT = 2.5;
        alpha = 0.6;
        epsilon = 0.8;
        tau = 1.1;
        resetCycle = 0;
    }

    /**
     * Returns parameters used for MCTS policy.
     *
     * @return parameters used for MCTS policy.
     */
    public String getParamDefs() {
        return paramNameTypes;
    }

    /**
     * Sets parameters used for MCTS policy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - cPUCT: Constant value for controlling amount of exploration i.e. C value of polynomial upper confidence tree. Default value 2.5.<br>
     *     - alpha: Shape value for Dirichlet sampling. Default value 0.6.<br>
     *     - epsilon: Weighting for Dirichlet distribution at action selection. Default value 0.8.<br>
     *     - tau: Temperature value for node visit count. Default value 1.1.<br>
     *     - resetCycle: Reset cycle counted as number of increments if cycle is 0 then reset is never applied. Default value 0.<br>
     *
     * @param params parameters used for MCTS policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("cPUCT")) cPUCT = params.getValueAsDouble("cPUCT");
        if (params.hasParam("alpha")) alpha = params.getValueAsDouble("alpha");
        if (params.hasParam("epsilon")) epsilon = params.getValueAsDouble("epsilon");
        if (params.hasParam("tau")) tau = params.getValueAsDouble("tau");
        if (params.hasParam("resetCycle")) resetCycle = params.getValueAsInteger("resetCycle");
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
     * Return flag is agent is in learning mode.
     *
     * @return if true agent is in learning mode.
     */
    private boolean isLearning() {
        return isLearning;
    }

    /**
     * Increments policy.
     *
     */
    public void increment() {
        if (++resetCount < resetCycle || resetCycle < 1) return;
        resetCount = 0;
        rootState = currentState = null;
    }

    /**
     * Updates state.
     *
     */
    private void updateState() {
        if (rootState == null) rootState = new State();
        currentState = currentState == null ? rootState : currentState.getNextState();
    }

    /**
     * Takes action decided by external agent.
     *
     * @param policyValueMatrix current state value matrix.
     * @param availableActions available actions in current state
     * @param action action.
     */
   public void action(Matrix policyValueMatrix, HashSet<Integer> availableActions, int action) {
        updateState();
        currentState.act(policyValueMatrix, availableActions, action);
    }

    /**
     * Takes action based on policy.
     *
     * @param policyValueMatrix current state value matrix.
     * @param availableActions available actions in current state
     * @param alwaysGreedy if true greedy action is always taken.
     * @return action taken.
     */
    public int action(Matrix policyValueMatrix, HashSet<Integer> availableActions, boolean alwaysGreedy) {
        updateState();
        return currentState.act(policyValueMatrix, availableActions, alwaysGreedy);
    }

    /**
     * Adds state transition for action execution.
     *
     * @param stateTransition state transition.
     */
    public void add(StateTransition stateTransition) {
        stateTransitionStack.push(stateTransition);
    }

    /**
     * Ends episode.
     *
     */
    public void endEpisode() {
        if (isLearning()) {
            if (currentState == null || stateTransitionStack.isEmpty()) return;
            currentState.update(stateTransitionStack);
        }
        stateTransitionStack = new Stack<>();
        currentState = null;
    }

    /**
     * Resets executable policy.
     *
     */
    public void reset() {
        rootState = currentState = null;
        stateTransitionStack = new Stack<>();
        resetCount = 0;
    }

    /**
     * Returns executable policy type.
     *
     * @return executable policy type.
     */
    public ExecutablePolicyType getExecutablePolicyType() {
        return executablePolicyType;
    }

}
