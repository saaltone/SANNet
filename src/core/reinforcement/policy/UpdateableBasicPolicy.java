/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.policy;

import core.NeuralNetworkException;
import core.reinforcement.RLSample;
import core.reinforcement.function.FunctionEstimator;
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
     * If true entropy is calculated when policy is executed.
     *
     */
    protected boolean updateEntropy = true;

    /**
     * Entropy coefficient for policy gradient.
     *
     */
    protected double entropyCoefficient = 0.01;

    /**
     * Constructor for UpdateableBasicPolicy.
     *
     * @param policy reference to UpdateableBasicPolicy.
     * @param functionEstimator reference to FunctionEstimator.
     */
    public UpdateableBasicPolicy(Policy policy, FunctionEstimator functionEstimator) {
        super(policy, functionEstimator);
    }

    /**
     * Constructor for UpdateableBasicPolicy.
     *
     * @param policy reference to policy.
     * @param functionEstimator reference to FunctionEstimator.
     * @param params parameters for policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public UpdateableBasicPolicy(Policy policy, FunctionEstimator functionEstimator, String params) throws DynamicParamException {
        this(policy, functionEstimator);
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for UpdateableBasicPolicy.
     *
     * @return parameters used for UpdateableBasicPolicy.
     */
    protected HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("updateEntropy", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("entropyCoefficient", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for UpdateableBasicPolicy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - updateEntropy: If true entropy is calculated for sample. Default value true.<br>
     *     - entropyCoefficient: co-efficient factor for entropy. Default value 0.01.<br>
     *
     * @param params parameters used for UpdateableBasicPolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("updateEntropy")) updateEntropy = params.getValueAsBoolean("updateEntropy");
        if (params.hasParam("entropyCoefficient")) entropyCoefficient = params.getValueAsDouble("entropyCoefficient");
    }

    /**
     * Takes action by applying defined policy,
     *
     * @param sample sample.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void act(RLSample sample) throws NeuralNetworkException, MatrixException {
        super.act(sample);
        if (updateEntropy) sample.entropy = getSampleEntropy(currentPolicyValues, sample.state.availableActions);
    }

    /**
     * Returns sample entropy.
     *
     * @param policyValues policy values.
     * @param availableActions available actions in state.
     * @return sample entropy.
     */
    private double getSampleEntropy(Matrix policyValues, HashSet<Integer> availableActions) {
        double entropy = 0;
        for (Integer action: availableActions) {
            double actionValue = policyValues.getValue(action, 0);
            entropy += actionValue * Math.log(actionValue);
        }
        return -entropy;
    }

    /**
     * Returns policy FunctionEstimator.
     *
     * @return policy FunctionEstimator.
     */
    public FunctionEstimator getFunctionEstimator() {
        return functionEstimator;
    }

    /**
     * Preprocesses policy gradient setting.
     *
     */
    protected void preProcess() {
    }

    /**
     * Returns policy gradient value for sample.
     *
     * @param sample sample
     * @param hasImportanceSamplingWeight if true sample has importance sample weight set.
     * @return policy gradient value for sample.
     */
    protected double getPolicyGradientValue(RLSample sample, boolean hasImportanceSamplingWeight) {
        double advantage = sample.tdTarget - sample.baseline;
        return Math.log(sample.policyValue) * (advantage + entropyCoefficient * sample.entropy) + 0.5 * Math.pow(advantage, 2);
    }

    /**
     * Postprocesses policy gradient setting.
     *
     */
    protected void postProcess() {
    }

}
