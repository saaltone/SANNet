package core.reinforcement.policy.executablepolicy;

import core.reinforcement.memory.StateTransition;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.Matrix;

import java.io.Serializable;
import java.util.*;

/**
 * Class that defines MCTSPolicy.<br>
 * <br>
 * Reference: https://medium.com/@jonathan_hui/monte-carlo-tree-search-mcts-in-alphago-zero-8a403588276a <br>
 * Reference: https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5 <br>
 * Reference: https://lczero.org/blog/2018/12/alphazero-paper-and-lc0-v0191/ <br>
 * Reference: https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py <br>
 *
 */
public class MCTSPolicy implements ExecutablePolicy, Serializable {

    private static final long serialVersionUID = -6567362723286339425L;

    /**
     * Class that defines action for state.
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
     * Class that defines state for MCTS.
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
         * @param stateValueOffset state value offset
         * @param action action.
         */
        void act(Matrix stateValueMatrix, HashSet<Integer> availableActions, int stateValueOffset, int action) {
            updateActionProbabilities(stateValueMatrix, availableActions, stateValueOffset);
            maxAction = actions.get(action);
        }

        /**
         * Takes action based on heuristic value. Adds noise to UCB value from Dirichlet distribution.
         *
         * @param stateValueMatrix current state value matrix.
         * @param availableActions available actions in current state
         * @param stateValueOffset state value offset
         * @param alwaysGreedy if true greedy action is always taken.
         * @return action taken.
         */
        public int act(Matrix stateValueMatrix, HashSet<Integer> availableActions, int stateValueOffset, boolean alwaysGreedy) {
            updateActionProbabilities(stateValueMatrix, availableActions, stateValueOffset);
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
         * Reference: https://stats.stackexchange.com/questions/69210/drawing-from-dirichlet-distribution<br>
         *
         * @param alpha shape of Dirichlet distribution.
         * @param availableActions available actions.
         * @return Dirichlet distribution.
         */
        private HashMap<Integer, Double> getDirichletDistribution(double alpha, HashSet<Integer> availableActions) {
            HashMap<Integer, Double> dirichletDistribution = new HashMap<>();
            double cumulativeValue = 0;
            for (Integer action : availableActions) {
                double gammaValue = generateGamma(alpha, 1);
                cumulativeValue += gammaValue;
                dirichletDistribution.put(action, gammaValue);
            }
            for (Integer action : dirichletDistribution.keySet()) dirichletDistribution.put(action, dirichletDistribution.get(action) / cumulativeValue);
            return dirichletDistribution;
        }

        /**
         * Generates random variable from gamma distribution.<br>
         * Reference: https://www.hongliangjie.com/2012/12/19/how-to-generate-gamma-random-variables/
         *
         * @param alpha alpha parameter
         * @param beta beta parameter
         * @return random variable from gamma distribution
         */
        private double generateGamma (double alpha, double beta) {
            if (alpha > 1) {
                double d = alpha - 1 / (double)3;
                double c = 1 / Math.sqrt(9 * d);
                while (true) {
                    double Z = random.nextGaussian();
                    if (Z > - 1 / c) {
                        double U = random.nextDouble();
                        double V = Math.pow(1 + c * Z, 3);
                        if (Math.log(U) < 0.5 * Math.pow(Z, 2) + d - d * V + d * Math.log(V)) return d * V / beta;
                    }
                }
            }
            else return generateGamma(alpha + 1, beta) * Math.pow(random.nextDouble(), 1 / alpha);
        }

        /**
         * Updates action probabilities for state.
         *
         * @param stateValueMatrix current state value matrix.
         * @param availableActions available actions in current state
         * @param stateValueOffset state value offset
         */
        private void updateActionProbabilities(Matrix stateValueMatrix, HashSet<Integer> availableActions, int stateValueOffset) {
            for (Integer action : availableActions) {
                double actionValue = stateValueMatrix.getValue(action + stateValueOffset, 0);
                if (actions.containsKey(action)) actions.get(action).setActionProbability(actionValue);
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
    private double cPUCT = 2.75;

    /**
     * Shape value for Dirichlet sampling.
     *
     */
    private double alpha = 0.6;

    /**
     * Weighting for Dirichlet distribution at action selection.
     *
     */
    private double epsilon = 0.8;

    /**
     * Temperature value for node visit count.
     *
     */
    private double tau = 1.1;

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
    private int resetCycle = 0;

    /**
     * Constructor for MCTSPolicy
     *
     */
    public MCTSPolicy() {
    }

    /**
     * Constructor for MCTSPolicy
     *
     * @param params parameters for MCTSPolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public MCTSPolicy(String params) throws DynamicParamException {
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for MCTSPolicy.
     *
     * @return parameters used for MCTSPolicy.
     */
    public HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("cPUCT", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("alpha", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("epsilon", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("tau", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("resetCycle", DynamicParam.ParamType.INT);
        return paramDefs;
    }

    /**
     * Sets parameters used for MCTSPolicy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - cPUCT: Constant value for controlling amount of exploration i.e. C value of polynomial upper confidence tree. Default value 2.75.<br>
     *     - alpha: Shape value for Dirichlet sampling. Default value 0.6.<br>
     *     - epsilon: Weighting for Dirichlet distribution at action selection. Default value 0.8.<br>
     *     - tau: Temperature value for node visit count. Default value 1.1.<br>
     *     - resetCycle: Reset cycle counted as number of increments if cycle is 0 then reset is never applied. Default value 0.<br>
     *
     * @param params parameters used for MCTSPolicy.
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
     * Resets policy.
     *
     * @param forceReset forces to trigger reset.
     */
    public void reset(boolean forceReset) {
        forceReset = false;
        if (!forceReset) if (++resetCount < resetCycle || resetCycle < 1) return;
        resetCount = 0;
        rootState = currentState = null;
    }

    /**
     * Increments policy.
     *
     */
    public void increment() {
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
     * @param stateValueMatrix current state value matrix.
     * @param availableActions available actions in current state
     * @param stateValueOffset state value offset
     * @param action action.
     */
   public void action(Matrix stateValueMatrix, HashSet<Integer> availableActions, int stateValueOffset, int action) {
        updateState();
        currentState.act(stateValueMatrix, availableActions, stateValueOffset, action);
    }

    /**
     * Takes action based on policy.
     *
     * @param stateValueMatrix current state value matrix.
     * @param availableActions available actions in current state
     * @param stateValueOffset state value offset
     * @param alwaysGreedy if true greedy action is always taken.
     * @return action taken.
     */
    public int action(Matrix stateValueMatrix, HashSet<Integer> availableActions, int stateValueOffset, boolean alwaysGreedy) {
        updateState();
        return currentState.act(stateValueMatrix, availableActions, stateValueOffset, alwaysGreedy);
    }

    /**
     * Records state transition for action execution.
     *
     * @param stateTransition state transition.
     */
    public void record(StateTransition stateTransition) {
        stateTransitionStack.push(stateTransition);
    }

    /**
     * Finishes episode.
     *
     * @param update if true update is executed.
     */
    public void finish(boolean update) {
        if (update) {
            if (currentState == null || stateTransitionStack.isEmpty()) return;
            currentState.update(stateTransitionStack);
        }
        stateTransitionStack = new Stack<>();
        currentState = null;
    }

}
