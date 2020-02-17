/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package utils.matrix;

/**
 * Following functions are supported:
 *     POW,
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
 *
 */
public enum BinaryFunctionType {

    POW,
    MEAN_SQUARED_ERROR,
    MEAN_SQUARED_LOGARITHMIC_ERROR,
    MEAN_ABSOLUTE_ERROR,
    MEAN_ABSOLUTE_PERCENTAGE_ERROR,
    CROSS_ENTROPY,
    KULLBACK_LEIBLER,
    NEGATIVE_LOG_LIKELIHOOD,
    POISSON,
    HINGE,
    SQUARED_HINGE,
    HUBER,
    DIRECT_GRADIENT

}
