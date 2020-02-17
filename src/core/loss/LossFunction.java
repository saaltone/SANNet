/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.loss;

import core.NeuralNetworkException;
import utils.*;
import utils.matrix.BinaryFunction;
import utils.matrix.BinaryFunctionType;

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
 */
public class LossFunction extends BinaryFunction {

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
            BinaryFunctionType.DIRECT_GRADIENT
    };

    /**
     * Constructor for loss function.
     *
     * @param binaryFunctionType type of loss function to be used.
     * @throws NeuralNetworkException throws exception if function is not available as loss function.
     */
    public LossFunction(BinaryFunctionType binaryFunctionType) throws NeuralNetworkException {
        super(binaryFunctionType);
        boolean found = false;
        for (BinaryFunctionType lossFunction : lossFunctions) {
            if (lossFunction == binaryFunctionType) {
                found = true;
                break;
            }
        }
        if (!found) throw new NeuralNetworkException("No such loss function available.");
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
     */
    public LossFunction(BinaryFunctionType binaryFunctionType, String params) throws NeuralNetworkException, DynamicParamException {
        super(binaryFunctionType, params);
        boolean found = false;
        for (BinaryFunctionType lossFunction : lossFunctions) {
            if (lossFunction == binaryFunctionType) {
                found = true;
                break;
            }
        }
        if (!found) throw new NeuralNetworkException("No such loss function available.");
    }

}
