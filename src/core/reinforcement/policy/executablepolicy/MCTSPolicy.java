/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.policy.executablepolicy;

import core.reinforcement.agent.State;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.AbstractMatrix;
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
    private class MCTSAction {

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
        private final MCTSState parentMCTSState;

        /**
         * Reference to child state.
         *
         */
        private final MCTSState childMCTSState;

        /**
         * Constructor for action.
         *
         * @param parentMCTSState parent state
         * @param actionID action ID
         */
        MCTSAction(MCTSState parentMCTSState, int actionID) {
            this.parentMCTSState = parentMCTSState;
            this.actionID = actionID;
            childMCTSState = new MCTSState(parentMCTSState);
        }

        /**
         * Constructor for action.
         *
         * @param parentMCTSState parent state
         * @param actionID action ID
         * @param actionProbability action probability.
         */
        MCTSAction(MCTSState parentMCTSState, int actionID, double actionProbability) {
            this(parentMCTSState, actionID);
            this.actionProbability = actionProbability;
        }

        /**
         * Returns parent state.
         *
         * @return parent state.
         */
        MCTSState getParentState() {
            return parentMCTSState;
        }

        /**
         * Returns child state.
         *
         * @return child state.
         */
        MCTSState getChildState() {
            return childMCTSState;
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
    private class MCTSState {

        /**
         * Parent state for state;
         *
         */
        private final MCTSState parentMCTSState;

        /**
         * Visit count for state.
         *
         */
        private int visitCount = 0;

        /**
         * Actions for state.
         *
         */
        private final HashMap<Integer, MCTSAction> stateMCTSActions = new HashMap<>();

        /**
         * Reference to action with maximum value.
         *
         */
        private MCTSAction maxMCTSAction = null;

        /**
         * Constructor for state.
         *
         */
        MCTSState() {
            parentMCTSState = null;
        }

        /**
         * Constructor for state.
         *
         * @param parentMCTSState parent state.
         */
        MCTSState(MCTSState parentMCTSState) {
            this.parentMCTSState = parentMCTSState;
        }

        /**
         * Returns child state based on max action.
         *
         * @return child state based on max action.
         */
        MCTSState getNextState() {
            return maxMCTSAction.getChildState();
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
        MCTSState getParentState() {
            return parentMCTSState;
        }

        /**
         * Returns true if state has max action otherwise false.
         *
         * @return true if state has max action otherwise false.
         */
        boolean hasMaxAction() {
            return maxMCTSAction != null;
        }

        /**
         * Returns max action.
         *
         * @return max action.
         */
        MCTSAction getMaxAction() {
            return maxMCTSAction;
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
            maxMCTSAction = stateMCTSActions.get(action);
        }

        /**
         * Takes action based on heuristic value. Adds noise to upper confidence boundary (UCB) value from Dirichlet distribution.
         *
         * @param policyValueMatrix current policy value matrix.
         * @param availableActions available actions in current state
         * @param alwaysGreedy if true greedy action is always taken.
         * @return action taken.
         */
        public int act(Matrix policyValueMatrix, HashSet<Integer> availableActions, boolean alwaysGreedy) {
            updateActionProbabilities(policyValueMatrix, availableActions);
            maxMCTSAction = null;
            double maxValue = Double.MIN_VALUE;
            if (alwaysGreedy) {
                for (Integer actionID : availableActions) {
                    MCTSAction MCTSAction = stateMCTSActions.get(actionID);
                    double currentValue = MCTSAction.getPolicyValue();
                    if (maxValue == Double.MIN_VALUE || maxValue < currentValue) {
                        maxValue = currentValue;
                        maxMCTSAction = MCTSAction;
                    }
                }
            }
            else {
                incrementVisitCount();
                HashMap<Integer, Double> dirichletDistribution = getDirichletDistribution(alpha, availableActions);
                for (Integer actionID : availableActions) {
                    MCTSAction MCTSAction = stateMCTSActions.get(actionID);
                    double currentValue = MCTSAction.getPUCT(dirichletDistribution.get(MCTSAction.actionID), epsilon);
                    if (maxValue == Double.MIN_VALUE || maxValue < currentValue) {
                        maxValue = currentValue;
                        maxMCTSAction = MCTSAction;
                    }
                }
            }
            return (maxMCTSAction != null) ? maxMCTSAction.getActionID() : -1;
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
            HashMap<Integer, Double> dirichletDistribution = new HashMap<>();
            double cumulativeValue = 0;
            for (Integer action : availableActions) {
                double gammaValue = AbstractMatrix.sampleGamma(shape, 1, random);
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
         * Updates action probabilities for state.
         *
         * @param policyValueMatrix current state value matrix.
         * @param availableActions available actions in current state
         */
        private void updateActionProbabilities(Matrix policyValueMatrix, HashSet<Integer> availableActions) {
            for (Integer actionID : availableActions) {
                double actionValue = policyValueMatrix.getValue(actionID, 0, 0);
                MCTSAction MCTSAction = stateMCTSActions.get(actionID);
                if (MCTSAction != null) MCTSAction.setActionProbability(actionValue);
                else stateMCTSActions.put(actionID, new MCTSAction(this, actionID, actionValue));
            }
            double cumulativeValue = 0;
            for (MCTSAction MCTSAction : stateMCTSActions.values()) cumulativeValue += MCTSAction.getActionProbability();
            if (cumulativeValue > 0) for (MCTSAction MCTSAction : stateMCTSActions.values()) MCTSAction.setActionProbability(MCTSAction.getActionProbability() / cumulativeValue);
        }

        /**
         * Updates state action value chain from leaf state towards root state.
         *
         * @param stateStack state stack.
         */
        public void update(Stack<core.reinforcement.agent.State> stateStack) {
            State state = stateStack.pop();
            getMaxAction().updateActionValue(state.tdTarget);
            state.policyValue = getMaxAction().getPolicyValue();
            if (hasParentState()) getParentState().update(stateStack);
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
    public MCTSState rootMCTSState;

    /**
     * Reference to current state.
     *
     */
    public MCTSState currentMCTSState;

    /**
     * Stack to store states for actions taken.
     *
     */
    private Stack<State> stateStack = new Stack<>();

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
        rootMCTSState = currentMCTSState = null;
    }

    /**
     * Updates state.
     *
     */
    private void updateState() {
        if (rootMCTSState == null) rootMCTSState = new MCTSState();
        currentMCTSState = currentMCTSState == null ? rootMCTSState : currentMCTSState.getNextState();
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
        currentMCTSState.act(policyValueMatrix, availableActions, action);
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
        return currentMCTSState.act(policyValueMatrix, availableActions, !isLearning() || alwaysGreedy);
    }

    /**
     * Adds state for action execution.
     *
     * @param state state.
     */
    public void add(core.reinforcement.agent.State state) {
        stateStack.push(state);
    }

    /**
     * Ends episode.
     *
     */
    public void endEpisode() {
        if (isLearning()) {
            if (currentMCTSState == null || stateStack.isEmpty()) return;
            currentMCTSState.update(stateStack);
        }
        stateStack = new Stack<>();
        currentMCTSState = null;
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
