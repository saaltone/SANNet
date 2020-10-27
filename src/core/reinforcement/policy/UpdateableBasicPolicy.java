/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.policy;

import core.NeuralNetworkException;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.executablepolicy.ExecutablePolicy;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashMap;
import java.util.HashSet;

/**
 * Class that defines vanilla UpdateableBasicPolicy.
 *
 */
public class UpdateableBasicPolicy extends AbstractUpdateablePolicy {

    /**
     * If true entropy is applied when policy is updated.
     *
     */
    protected boolean applyEntropy = true;

    /**
     * Entropy coefficient for policy gradient.
     *
     */
    protected double entropyCoefficient = 0.01;

    /**
     * Constructor for UpdateableBasicPolicy.
     *
     * @param executablePolicy reference to UpdateableBasicPolicy.
     * @param functionEstimator reference to FunctionEstimator.
     */
    public UpdateableBasicPolicy(ExecutablePolicy executablePolicy, FunctionEstimator functionEstimator) {
        super(executablePolicy, functionEstimator);
    }

    /**
     * Constructor for UpdateableBasicPolicy.
     *
     * @param executablePolicy reference to executable policy.
     * @param functionEstimator reference to FunctionEstimator.
     * @param params parameters for UpdateableBasicPolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public UpdateableBasicPolicy(ExecutablePolicy executablePolicy, FunctionEstimator functionEstimator, String params) throws DynamicParamException {
        this(executablePolicy, functionEstimator);
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for UpdateableBasicPolicy.
     *
     * @return parameters used for UpdateableBasicPolicy.
     */
    protected HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("applyEntropy", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("entropyCoefficient", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for UpdateableBasicPolicy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - applyEntropy: if true entropy is applied when policy is updated. Default value true.<br>
     *     - entropyCoefficient: co-efficient factor for entropy. Default value 0.01.<br>
     *
     * @param params parameters used for UpdateableBasicPolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("applyEntropy")) applyEntropy = params.getValueAsBoolean("applyEntropy");
        if (params.hasParam("entropyCoefficient")) entropyCoefficient = params.getValueAsDouble("entropyCoefficient");
    }

    /**
     * Returns policy gradient value for update.
     *
     * @param stateTransition state transition.
     * @return policy gradient value.
     */
    protected double getPolicyValue(StateTransition stateTransition) throws MatrixException, NeuralNetworkException {
        Matrix currentPolicyValues = functionEstimator.predict(stateTransition.environmentState.state);
        double currentPolicyValue = currentPolicyValues.getValue(getAction(stateTransition.action), 0) + 10E-15;
        return -(currentPolicyValue * Math.log(currentPolicyValue) * (stateTransition.advantage + (applyEntropy ? entropyCoefficient * getSampleEntropy(currentPolicyValues, stateTransition.environmentState.availableActions) : 0)));
    }

    /**
     * Returns entropy.
     *
     * @param policyValues policy values.
     * @param availableActions available actions in state.
     * @return entropy.
     */
    private double getSampleEntropy(Matrix policyValues, HashSet<Integer> availableActions) {
        double entropy = 0;
        for (Integer action: availableActions) {
            double actionValue = policyValues.getValue(getAction(action), 0);
            if (actionValue != 0) entropy += -actionValue * Math.log(actionValue);
        }
        return entropy;
    }

}
