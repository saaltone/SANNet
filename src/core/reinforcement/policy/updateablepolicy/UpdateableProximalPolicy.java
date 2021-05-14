/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.policy.updateablepolicy;

import core.NeuralNetworkException;
import core.reinforcement.AgentException;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.policy.AbstractUpdateablePolicy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.util.HashMap;

/**
 * Class that defines UpdateableProximalPolicy. Implements Proximal Policy Optimization (PPO).<br>
 *
 */
public class UpdateableProximalPolicy extends AbstractUpdateablePolicy {

    /**
     * Reference to previous policy function estimator.
     *
     */
    private final FunctionEstimator previousFunctionEstimator;

    /**
     * Epsilon value for proximal policy value clipping.
     *
     */
    private double epsilon = 0.2;

    /**
     * Update cycle for previous function estimator.
     *
     */
    private int updateCycle = 1;

    /**
     * Update count for previous function estimator updates.
     *
     */
    private int updateCount = 0;

    /**
     * Constructor for UpdateableProximalPolicy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to FunctionEstimator.
     * @throws IOException throws exception if creation of previousFunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of previousFunctionEstimator fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public UpdateableProximalPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator) throws IOException, ClassNotFoundException, DynamicParamException, MatrixException {
        super(executablePolicyType, functionEstimator);
        previousFunctionEstimator = functionEstimator.copy();
    }

    /**
     * Returns parameters used for UpdateableProximalPolicy.
     *
     * @return parameters used for UpdateableProximalPolicy.
     */
    public HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>(super.getParamDefs());
        paramDefs.put("epsilon", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("updateCycle", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for UpdateableProximalPolicy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - epsilon: epsilon value for proximal policy value clipping. Default value 0.2.<br>
     *     - updateCycle: update cycle for previous estimator function update. Default value 1.<br>
     *
     * @param params parameters used for UpdateableProximalPolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
        if (params.hasParam("epsilon")) epsilon = params.getValueAsDouble("epsilon");
        if (params.hasParam("updateCycle")) updateCycle = params.getValueAsInteger("updateCycle");
    }

    /**
     * Starts FunctionEstimator
     *
     * @throws NeuralNetworkException throws exception if start of neural network estimator(s) fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void start() throws NeuralNetworkException, MatrixException, DynamicParamException {
        super.start();
        previousFunctionEstimator.start();
    }

    /**
     * Stops FunctionEstimator
     *
     */
    public void stop() {
        super.stop();
        previousFunctionEstimator.stop();
    }

    /**
     * Returns policy gradient value for update.
     *
     * @param stateTransition state transition.
     * @return policy gradient value.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected double getPolicyValue(StateTransition stateTransition) throws NeuralNetworkException, MatrixException {
        int action = getAction(stateTransition.action);
        double currentActionValue = functionEstimator.predict(stateTransition.environmentState.state()).getValue(action, 0);
        double previousActionValue = previousFunctionEstimator.predict(stateTransition.environmentState.state()).getValue(action, 0);
        double rValue = previousActionValue == 0 ? 1 : currentActionValue / previousActionValue;
        double clippedRValue = Math.min(Math.max(rValue, 1 - epsilon), 1 + epsilon);
        return -Math.min(rValue * stateTransition.advantage, clippedRValue * stateTransition.advantage);
    }

    /**
     * Postprocesses policy gradient update.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws AgentException throws exception if update cycle is ongoing.
     */
    protected void postProcess() throws MatrixException, AgentException {
        if (++updateCount >= updateCycle) {
            previousFunctionEstimator.append(functionEstimator, true);
            updateCount = 0;
        }
    }

}
