/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.loss;

import core.NeuralNetworkException;
import utils.*;

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
 *     HUBER
 */
public class LossFunction extends BiFunction {

    private static final long serialVersionUID = 6218297482907539129L;

    /**
     * List of supported loss functions.
     *
     */
    private final BiFunctionType[] lossFunctions = new BiFunctionType[] {
            BiFunctionType.MEAN_SQUARED_ERROR,
            BiFunctionType.MEAN_SQUARED_LOGARITHMIC_ERROR,
            BiFunctionType.MEAN_ABSOLUTE_ERROR,
            BiFunctionType.MEAN_ABSOLUTE_PERCENTAGE_ERROR,
            BiFunctionType.CROSS_ENTROPY,
            BiFunctionType.KULLBACK_LEIBLER,
            BiFunctionType.NEGATIVE_LOG_LIKELIHOOD,
            BiFunctionType.POISSON,
            BiFunctionType.HINGE,
            BiFunctionType.SQUARED_HINGE,
            BiFunctionType.HUBER
    };

    /**
     * Constructor for loss function.
     *
     * @param biFunctionType type of loss function to be used.
     * @throws NeuralNetworkException throws exception if function is not available as loss function.
     */
    public LossFunction(BiFunctionType biFunctionType) throws NeuralNetworkException {
        super(biFunctionType);
        boolean found = false;
        for (BiFunctionType lossFunction : lossFunctions) {
            if (lossFunction == biFunctionType) {
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
     * @param biFunctionType type of loss function to be used.
     * @param params parameters used for loss function.
     * @throws NeuralNetworkException throws exception if function is not available as loss function.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public LossFunction(BiFunctionType biFunctionType, String params) throws NeuralNetworkException, DynamicParamException {
        super(biFunctionType, params);
        boolean found = false;
        for (BiFunctionType lossFunction : lossFunctions) {
            if (lossFunction == biFunctionType) {
                found = true;
                break;
            }
        }
        if (!found) throw new NeuralNetworkException("No such loss function available.");
    }

}
