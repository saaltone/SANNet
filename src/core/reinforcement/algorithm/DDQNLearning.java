/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.algorithm;

import core.reinforcement.AgentException;
import core.reinforcement.Environment;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.ActionablePolicy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import core.reinforcement.value.QTargetValueFunctionEstimator;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Class that defines Double Q Learning.
 *
 */
public class DDQNLearning extends AbstractQLearning {

    /**
     * Constructor for DDQNLearning.
     *
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param valueFunctionEstimator reference to value function estimator.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if creation of executable policy fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public DDQNLearning(Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator valueFunctionEstimator) throws ClassNotFoundException, MatrixException, DynamicParamException, IOException, AgentException {
        super(environment, new ActionablePolicy(executablePolicyType, valueFunctionEstimator), new QTargetValueFunctionEstimator(valueFunctionEstimator));
    }

    /**
     * Constructor for DDQNLearning.
     *
     * @param environment reference to environment.
     * @param executablePolicyType executable policy type.
     * @param valueFunctionEstimator reference to value function estimator.
     * @param params parameters for agent.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if creation of executable policy fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public DDQNLearning(Environment environment, ExecutablePolicyType executablePolicyType, FunctionEstimator valueFunctionEstimator, String params) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException, AgentException {
        super(environment, new ActionablePolicy(executablePolicyType, valueFunctionEstimator), new QTargetValueFunctionEstimator(valueFunctionEstimator), params);
    }

}
