package core.reinforcement.policy.updateablepolicy;

import core.NeuralNetworkException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.policy.AbstractUpdateablePolicy;
import core.reinforcement.policy.executablepolicy.MCTSPolicy;
import utils.matrix.MatrixException;

/**
 * Class that defines updateable MCTS policy.<br>
 *
 */
public class UpdateableMCTSPolicy extends AbstractUpdateablePolicy {

    /**
     * Constructor for UpdateableMCTSPolicy.
     *
     * @param functionEstimator reference to FunctionEstimator.
     */
    public UpdateableMCTSPolicy(FunctionEstimator functionEstimator) {
        super (new MCTSPolicy(), functionEstimator);
    }

    /**
     * Constructor for UpdateableMCTSPolicy.
     *
     * @param functionEstimator reference to FunctionEstimator.
     * @param mctsPolicy reference to MCTS policy.
     */
    public UpdateableMCTSPolicy(FunctionEstimator functionEstimator, MCTSPolicy mctsPolicy) {
        super (mctsPolicy, functionEstimator);
    }

    /**
     * Returns policy value for update.
     *
     * @param stateTransition state transition.
     * @return policy value.
     */
    protected double getPolicyValue(StateTransition stateTransition) throws MatrixException, NeuralNetworkException {
        return -stateTransition.value * Math.log(functionEstimator.predict(stateTransition.environmentState.state).getValue(getAction(stateTransition.action), 0) + 10E-6);
    }

}
