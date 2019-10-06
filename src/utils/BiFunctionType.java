/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package utils;

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
 *     HUBER
 *
 */
public enum BiFunctionType {
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
    HUBER

}
