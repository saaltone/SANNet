/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.loss;

import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.BinaryFunction;
import utils.matrix.BinaryFunctionType;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements loss function for neural network used at output layer.<br>
 * Provides calculation for both function and it's derivative.<br>
 * <br>
 * Following functions are supported:
 *     MEAN_SQUARED_ERROR,
 *     MEAN_SQUARED_LOGARITHMIC_ERROR,
 *     MEAN_ABSOLUTE_ERROR,
 *     MEAN_ABSOLUTE_PERCENTAGE_ERROR,
 *     CROSS_ENTROPY,
 *     BINARY_CROSS_ENTROPY,
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
 *     COS_SIM,
 *     CUSTOM
 */
public class LossFunction extends BinaryFunction {

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
            BinaryFunctionType.BINARY_CROSS_ENTROPY,
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
            BinaryFunctionType.COS_SIM,
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
                Matrix error = target.getNewMatrix();
                int targetRows = target.getRows();
                for (int row = 0; row < targetRows; row++) {
                    error.setValue(row, 0 , row == 0 ? (0.5 * Math.pow(target.getValue(0, 0) - output.getValue(0, 0), 2)) : target.getValue(row, 0));
                }
                return error;
            }
            // https://math.stackexchange.com/questions/1923613/partial-derivative-of-cosine-similarity
            case COS_SIM -> {
                double norm_output = output.norm(2);
                double norm_target = target.norm(2);
                return output.multiply(target).divide(norm_output * norm_target);
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
     * Returns absolute error.
     *
     * @param error error.
     * @return absolute error.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double getAbsoluteError(Matrix error) throws MatrixException {
        return getType() == BinaryFunctionType.COS_SIM ? 1 - error.mean() * (double)error.size() : error.mean();
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
                Matrix gradient = target.getNewMatrix();
                int targetRows = target.getRows();
                for (int row = 0; row < targetRows; row++) {
                    gradient.setValue(row, 0 , row == 0 ? (output.getValue(0, 0) - target.getValue(0, 0)) : target.getValue(row, 0));
                }
                return gradient;
            }
            // https://math.stackexchange.com/questions/1923613/partial-derivative-of-cosine-similarity
            case COS_SIM -> {
                double norm_output = output.norm(2);
                double norm_target = target.norm(2);
                double norm_multiply = norm_output * norm_target;
                Matrix cos_sim = output.multiply(target).divide(norm_multiply);
                return output.divide(norm_multiply).subtract(target.divide(Math.pow(norm_output, 2)).multiply(cos_sim));
            }
            default -> {
                return output.applyBi(target, getDerivative());
            }
        }
    }

}