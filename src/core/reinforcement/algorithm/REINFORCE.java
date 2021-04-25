package core.reinforcement.algorithm;

import core.reinforcement.Environment;
import core.reinforcement.function.DirectFunctionEstimator;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.OnlineMemory;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import core.reinforcement.policy.updateablepolicy.UpdateableBasicPolicy;
import core.reinforcement.policy.updateablepolicy.UpdateableProximalPolicy;
import core.reinforcement.value.PlainValueFunction;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Class that defines REINFORCE algorithm.<br>
 *
 */
public class REINFORCE extends AbstractPolicyGradient {

    /**
     * Constructor for REINFORCE
     *
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public REINFORCE(Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator policyFunctionEstimator) throws DynamicParamException {
        super(environment, new UpdateableBasicPolicy(executablePolicyType, policyFunctionEstimator), new PlainValueFunction(policyFunctionEstimator.getNumberOfActions(), new DirectFunctionEstimator(new OnlineMemory(), policyFunctionEstimator.getNumberOfActions())));
    }

    /**
     * Constructor for REINFORCE
     *
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param params parameters for agent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public REINFORCE(Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator policyFunctionEstimator, String params) throws DynamicParamException {
        super(environment, new UpdateableBasicPolicy(executablePolicyType, policyFunctionEstimator), new PlainValueFunction(policyFunctionEstimator.getNumberOfActions(), new DirectFunctionEstimator(new OnlineMemory(), policyFunctionEstimator.getNumberOfActions())), params);
    }

    /**
     * Constructor for REINFORCE
     *
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param asProximalPolicy if true proximal policy as applied otherwise basic policy gradient is applied.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public REINFORCE(Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator policyFunctionEstimator, boolean asProximalPolicy) throws ClassNotFoundException, MatrixException, DynamicParamException, IOException {
        super(environment, asProximalPolicy ? new UpdateableProximalPolicy(executablePolicyType, policyFunctionEstimator) : new UpdateableBasicPolicy(executablePolicyType, policyFunctionEstimator), new PlainValueFunction(policyFunctionEstimator.getNumberOfActions(), new DirectFunctionEstimator(new OnlineMemory(), policyFunctionEstimator.getNumberOfActions())));
    }

    /**
     * Constructor for REINFORCE
     *
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param policyFunctionEstimator reference to policy function estimator.
     * @param params parameters for agent.
     * @param asProximalPolicy if true proximal policy as applied otherwise basic policy gradient is applied.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public REINFORCE(Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator policyFunctionEstimator, String params, boolean asProximalPolicy) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException {
        super(environment, asProximalPolicy ? new UpdateableProximalPolicy(executablePolicyType, policyFunctionEstimator) : new UpdateableBasicPolicy(executablePolicyType, policyFunctionEstimator), new PlainValueFunction(policyFunctionEstimator.getNumberOfActions(), new DirectFunctionEstimator(new OnlineMemory(), policyFunctionEstimator.getNumberOfActions())), params);
    }

}
