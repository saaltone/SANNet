/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.loss;

import utils.configurable.DynamicParamException;
import utils.matrix.BinaryFunction;
import utils.matrix.BinaryFunctionType;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements loss function for neural network used at output layer.
 *
 */
public class LossFunction extends BinaryFunction {

    /**
     * Constructor for loss function.<br>
     * Supported parameters are:<br>
     *     - hingeMargin: default value for hinge margin 1.<br>
     *     - huberDelta: default value for Huber loss 1.<br>
     *
     * @param lossFunctionType loss function type.
     * @param params parameters used for loss function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public LossFunction(LossFunctionType lossFunctionType, String params) throws MatrixException, DynamicParamException {
        super(switch(lossFunctionType) {
            case MEAN_SQUARED_ERROR -> BinaryFunctionType.MEAN_SQUARED_ERROR;
            case MEAN_SQUARED_LOGARITHMIC_ERROR -> BinaryFunctionType.MEAN_SQUARED_LOGARITHMIC_ERROR;
            case MEAN_ABSOLUTE_ERROR -> BinaryFunctionType.MEAN_ABSOLUTE_ERROR;
            case MEAN_ABSOLUTE_PERCENTAGE_ERROR -> BinaryFunctionType.MEAN_ABSOLUTE_PERCENTAGE_ERROR;
            case CROSS_ENTROPY -> BinaryFunctionType.CROSS_ENTROPY;
            case BINARY_CROSS_ENTROPY -> BinaryFunctionType.BINARY_CROSS_ENTROPY;
            case KULLBACK_LEIBLER -> BinaryFunctionType.KULLBACK_LEIBLER;
            case NEGATIVE_LOG_LIKELIHOOD -> BinaryFunctionType.NEGATIVE_LOG_LIKELIHOOD;
            case POISSON -> BinaryFunctionType.POISSON;
            case HINGE -> BinaryFunctionType.HINGE;
            case SQUARED_HINGE -> BinaryFunctionType.SQUARED_HINGE;
            case HUBER -> BinaryFunctionType.HUBER;
            case DIRECT_GRADIENT -> BinaryFunctionType.DIRECT_GRADIENT;
            case POLICY_GRADIENT -> BinaryFunctionType.POLICY_GRADIENT;
            case POLICY_VALUE -> BinaryFunctionType.POLICY_VALUE;
            case DQN_REG_LOSS -> BinaryFunctionType.DQN_REG_LOSS;
            case COS_SIM -> BinaryFunctionType.COS_SIM;
            case CUSTOM -> BinaryFunctionType.CUSTOM;
        }, params);
    }

    /**
     * Returns mean error
     *
     * @param totalError total error
     * @param numberOfErrors number of error;
     * @return mean error
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public static Matrix getMeanError(Matrix totalError, int numberOfErrors) throws MatrixException {
        return totalError.divide(numberOfErrors);
    }

    /**
     * Returns absolute error.
     *
     * @param error error.
     * @return absolute error.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double getAbsoluteError(Matrix error) throws MatrixException {
        return getType() == BinaryFunctionType.COS_SIM ? 1 - error.mean() * (double)error.size() : error.mean();
    }

}