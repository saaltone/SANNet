package core.reinforcement.policy;

import core.NeuralNetworkException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.policy.executablepolicy.MCTSPolicy;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

/**
 * Class that defines updateable MCTS policy.
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
     * @param params parameters for MCTS policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public UpdateableMCTSPolicy(FunctionEstimator functionEstimator, String params) throws DynamicParamException {
        super (new MCTSPolicy(params), functionEstimator);
    }

    /**
     * Returns policy value for update.
     *
     * @param stateTransition state transition.
     * @return policy value.
     */
    protected double getPolicyValue(StateTransition stateTransition) throws MatrixException, NeuralNetworkException {
        return -stateTransition.actionValue * Math.log(getFunctionEstimator().predict(stateTransition.environmentState.state).getValue(getAction(stateTransition.action), 0) + 10E-6);
    }

}
