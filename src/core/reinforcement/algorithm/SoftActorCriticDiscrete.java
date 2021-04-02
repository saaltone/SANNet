package core.reinforcement.algorithm;

import core.NeuralNetworkException;
import core.reinforcement.AgentException;
import core.reinforcement.Environment;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import core.reinforcement.policy.updateablepolicy.UpdateableSoftQPolicy;
import core.reinforcement.value.SoftQValueFunctionEstimator;
import utils.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Class that defines discrete Soft Actor Critic algorithm.
 *
 */
public class SoftActorCriticDiscrete extends AbstractPolicyGradient {

    /**
     * Constructor for SoftActorCriticDiscrete
     *
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param valueFunctionEstimator reference to value function estimator.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws AgentException throws exception if creation of executable policy fails.
     */
    public SoftActorCriticDiscrete(Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator policyFunctionEstimator, FunctionEstimator valueFunctionEstimator) throws ClassNotFoundException, MatrixException, DynamicParamException, IOException, AgentException, NeuralNetworkException {
        Matrix softQAlphaMatrix = new DMatrix(1, 1);
        initialize(environment, new UpdateableSoftQPolicy(executablePolicyType, policyFunctionEstimator, softQAlphaMatrix), new SoftQValueFunctionEstimator(policyFunctionEstimator, valueFunctionEstimator, softQAlphaMatrix));
    }

    /**
     * Constructor for SoftActorCriticDiscrete
     *
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param valueFunctionEstimator reference to value function estimator.
     * @param params parameters for agent.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws AgentException throws exception if creation of executable policy fails.
     */
    public SoftActorCriticDiscrete(Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator policyFunctionEstimator, FunctionEstimator valueFunctionEstimator, String params) throws ClassNotFoundException, MatrixException, DynamicParamException, IOException, AgentException, NeuralNetworkException {
        Matrix softQAlphaMatrix = new DMatrix(1, 1);
        initialize(environment, new UpdateableSoftQPolicy(executablePolicyType, policyFunctionEstimator, softQAlphaMatrix), new SoftQValueFunctionEstimator(policyFunctionEstimator, valueFunctionEstimator, softQAlphaMatrix), params);
    }

}
