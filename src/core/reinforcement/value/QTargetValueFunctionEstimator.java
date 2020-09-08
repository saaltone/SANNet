/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.value;

import core.NeuralNetworkException;
import core.reinforcement.Agent;
import core.reinforcement.AgentException;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.function.FunctionEstimator;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.util.HashMap;
import java.util.TreeSet;

/**
 * Class that defines QTargetValueFunctionEstimator (Q value function with target function estimator).
 *
 */
public class QTargetValueFunctionEstimator extends AbstractValueFunctionEstimator {

    /**
     * Target value FunctionEstimator.
     *
     */
    private final FunctionEstimator targetValueFunctionEstimator;

    /**
     * Update cycle (in episodes) for target FunctionEstimator. If update cycle is zero then smooth parameter updates are applied with update rate tau.
     *
     */
    private int updateCycle = 0;

    /**
     * Update count for update cycle.
     *
     */
    private transient int updateCount = 0;

    /**
     * Constructor for QTargetValueFunctionEstimator.
     *
     * @param functionEstimator reference to FunctionEstimator.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public QTargetValueFunctionEstimator(FunctionEstimator functionEstimator) throws IOException, ClassNotFoundException, DynamicParamException, MatrixException {
        super(functionEstimator.getNumberOfActions(), functionEstimator);
        targetValueFunctionEstimator = functionEstimator.copy();
    }

    /**
     * Constructor for QTargetValueFunctionEstimator.
     *
     * @param functionEstimator reference to value FunctionEstimator.
     * @param params parameters for QTargetValueFunctionEstimator.
     * @throws IOException throws exception if creation of target value FunctionEstimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value FunctionEstimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     */
    public QTargetValueFunctionEstimator(FunctionEstimator functionEstimator, String params) throws IOException, ClassNotFoundException, DynamicParamException, MatrixException {
        super(functionEstimator.getNumberOfActions(), functionEstimator, params);
        setParams(new DynamicParam(params, getParamDefs()));
        targetValueFunctionEstimator = functionEstimator.copy();
    }

    /**
     * Returns parameters used for QTargetValueFunctionEstimator.
     *
     * @return parameters used for QTargetValueFunctionEstimator.
     */
    protected HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>(super.getParamDefs());
        paramDefs.put("updateCycle", DynamicParam.ParamType.INT);
        return paramDefs;
    }

    /**
     * Sets parameters used for QTargetValueFunctionEstimator.<br>
     * <br>
     * Supported parameters are:<br>
     *     - updateCycle; update cycle for target function (assumes full update). Default value 0 i.e. applies smooth update.<br>
     *
     * @param params parameters used for QTargetValueFunctionEstimator.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
        if (params.hasParam("updateCycle")) updateCycle = params.getValueAsInteger("updateCycle");
    }

    /**
     * Starts FunctionEstimator
     *
     * @throws NeuralNetworkException throws exception if start of neural network estimator(s) fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     */
    public void start() throws NeuralNetworkException, MatrixException {
        super.start();
        targetValueFunctionEstimator.start();
    }

    /**
     * Stops FunctionEstimator
     *
     */
    public void stop() {
        super.stop();
        targetValueFunctionEstimator.stop();
    }

    /**
     * Returns target value based on next state. Uses target action with maximal value as defined by target FunctionEstimator.
     *
     * @param nextStateTransition next state transition.
     * @return target value based on next state
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double getTargetValue(StateTransition nextStateTransition) throws NeuralNetworkException, MatrixException {
        return functionEstimator.predict(nextStateTransition.environmentState.state).getValue(argmax(targetValueFunctionEstimator.predict(nextStateTransition.environmentState.state), nextStateTransition.environmentState.availableActions), 0);
    }

    /**
     * Updates FunctionEstimator.
     *
     * @param agent agent.
     * @param stateTransitions state transitions used to update FunctionEstimator.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if function estimator update fails.
     */
    public void updateFunctionEstimator(Agent agent, TreeSet<StateTransition> stateTransitions) throws NeuralNetworkException, MatrixException, DynamicParamException, AgentException {
        super.updateFunctionEstimator(agent, stateTransitions);
        if (updateCycle == 0) targetValueFunctionEstimator.append(functionEstimator, false);
        else {
            if (++updateCount >= updateCycle) {
                targetValueFunctionEstimator.append(functionEstimator, true);
                updateCount = 0;
            }
        }
    }

}
