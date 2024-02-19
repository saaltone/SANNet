/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.value;

import core.reinforcement.function.FunctionEstimator;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Implements target Q value function including option for dual estimator.<br>
 *
 */
public class QTargetValueFunction extends QValueFunction {

    /**
     * Constructor for target Q value function.
     *
     * @param functionEstimator reference to value FunctionEstimator.
     * @param params parameters for target Q value function estimator.
     */
    public QTargetValueFunction(FunctionEstimator functionEstimator, String params) {
        this(functionEstimator, false, params);
    }

    /**
     * Constructor for target Q value function.
     *
     * @param functionEstimator  reference to value function estimator.
     * @param dualFunctionEstimation if true soft Q value function estimator has dual function estimator.
     * @param params             parameters for target Q value function estimator.
     */
    public QTargetValueFunction(FunctionEstimator functionEstimator, boolean dualFunctionEstimation, String params) {
        this(functionEstimator, null, dualFunctionEstimation, true, params);
    }

    /**
     * Constructor for target Q value function.
     *
     * @param functionEstimator  reference to value function estimator.
     * @param functionEstimator2 reference to second value function estimator.
     * @param dualFunctionEstimation if true soft Q value function estimator has dual function estimator.
     * @param usesTargetValueFunctionEstimator if true uses target value function estimator.
     * @param params             parameters for target Q value function estimator.
     */
    protected QTargetValueFunction(FunctionEstimator functionEstimator, FunctionEstimator functionEstimator2, boolean dualFunctionEstimation, boolean usesTargetValueFunctionEstimator, String params) {
        super(functionEstimator, functionEstimator2, dualFunctionEstimation, usesTargetValueFunctionEstimator, params);
    }

    /**
     * Returns reference to value function.
     *
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @return reference to value function.
     * @throws IOException            throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     * @throws MatrixException        throws exception if neural network has less output than actions.
     */
    public ValueFunction reference(boolean sharedValueFunctionEstimator) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException {
        return new QTargetValueFunction(sharedValueFunctionEstimator ? getFunctionEstimator() : getFunctionEstimator().reference(), sharedValueFunctionEstimator ? getFunctionEstimator2() : null, dualFunctionEstimation, isUsingTargetValueFunctionEstimator(), getParams());
    }

}
