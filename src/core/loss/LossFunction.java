/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.loss;

import core.network.NeuralNetworkException;
import utils.*;
import utils.matrix.*;

import java.io.Serial;

/**
 * Defines loss function class for neural network.<br>
 * Provides calculation for both function and it's derivative.<br>
 * <br>
 * Following functions are supported:
 *     MEAN_SQUARED_ERROR,
 *     MEAN_SQUARED_LOGARITHMIC_ERROR,
 *     MEAN_ABSOLUTE_ERROR,
 *     MEAN_ABSOLUTE_PERCENTAGE_ERROR,
 *     CROSS_ENTROPY,
 *     KULLBACK_LEIBLER,
 *     NEGATIVE_LOG_LIKELIHOOD,
 *     POISSON,
 *     HINGE,
 *     SQUARED_HINGE,
 *     HUBER,
 *     DIRECT_GRADIENT
 *     POLICY_GRADIENT,
 *     POLICY_VALUE,
 *     DQN_REG_LOSS,
 *     CUSTOM
 */
public class LossFunction extends BinaryFunction {

    @Serial
    private static final long serialVersionUID = 6218297482907539129L;

    /**
     * List of supported loss functions.
     *
     */
    private final BinaryFunctionType[] lossFunctions = new BinaryFunctionType[] {
            BinaryFunctionType.MEAN_SQUARED_ERROR,
            BinaryFunctionType.MEAN_SQUARED_LOGARITHMIC_ERROR,
            BinaryFunctionType.MEAN_ABSOLUTE_ERROR,
            BinaryFunctionType.MEAN_ABSOLUTE_PERCENTAGE_ERROR,
            BinaryFunctionType.CROSS_ENTROPY,
            BinaryFunctionType.KULLBACK_LEIBLER,
            BinaryFunctionType.NEGATIVE_LOG_LIKELIHOOD,
            BinaryFunctionType.POISSON,
            BinaryFunctionType.HINGE,
            BinaryFunctionType.SQUARED_HINGE,
            BinaryFunctionType.HUBER,
            BinaryFunctionType.DIRECT_GRADIENT,
            BinaryFunctionType.POLICY_GRADIENT,
            BinaryFunctionType.POLICY_VALUE,
            BinaryFunctionType.DQN_REG_LOSS,
            BinaryFunctionType.CUSTOM
    };

    /**
     * Constructor for loss function.
     *
     * @param binaryFunctionType type of loss function to be used.
     * @throws NeuralNetworkException throws exception if function is not available as loss function.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public LossFunction(BinaryFunctionType binaryFunctionType) throws NeuralNetworkException, MatrixException, DynamicParamException {
        super(binaryFunctionType);
        for (BinaryFunctionType lossFunctionType : lossFunctions) {
            if (lossFunctionType == binaryFunctionType) return;
        }
        throw new NeuralNetworkException("No such loss function available.");
    }

    /**
     * Constructor for loss function.<br>
     * Supported parameters are:<br>
     *     - alpha: default value for Huber loss 1.<br>
     *     - hinge: default value for hinge margin 1.<br>
     *
     * @param binaryFunctionType type of loss function to be used.
     * @param params parameters used for loss function.
     * @throws NeuralNetworkException throws exception if function is not available as loss function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public LossFunction(BinaryFunctionType binaryFunctionType, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super(binaryFunctionType, params);
        for (BinaryFunctionType lossFunctionType : lossFunctions) {
            if (lossFunctionType == binaryFunctionType) return;
        }
        throw new NeuralNetworkException("No such loss function available.");
    }

    /**
     * Returns error of loss function
     *
     * @param output predicted output
     * @param target actual target
     * @return error of loss function
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix getError(Matrix output, Matrix target) throws MatrixException {
        switch (getType()) {
            case DIRECT_GRADIENT -> {
                return target;
            }
            case POLICY_VALUE -> {
                Matrix error = new DMatrix(target.getRows(), 1);
                for (int row = 0; row < target.getRows(); row++) {
                    error.setValue(row, 0 , row == 0 ? (0.5 * Math.pow(target.getValue(0, 0) - output.getValue(0, 0), 2)) : target.getValue(row, 0));
                }
                return error;
            }
            default -> {
                return output.applyBi(target, getFunction());
            }
        }
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
     * Returns gradient of loss function
     *
     * @param output predicted output
     * @param target actual target
     * @return gradient of loss function
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix getGradient(Matrix output, Matrix target) throws MatrixException {
        switch (getType()) {
            case DIRECT_GRADIENT -> {
                return target;
            }
            case POLICY_VALUE -> {
                Matrix gradient = new DMatrix(target.getRows(), 1);
                for (int row = 0; row < target.getRows(); row++) {
                    gradient.setValue(row, 0 , row == 0 ? (output.getValue(0, 0) - target.getValue(0, 0)) : target.getValue(row, 0));
                }
                return gradient;
            }
            default -> {
                return output.applyBi(target, getDerivative());
            }
        }
    }

}