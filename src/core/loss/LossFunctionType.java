/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.loss;

/**
 * Loss function types.<br>
 * Currently supported types are:
 *     MEAN_SQUARED_ERROR,
 *     MEAN_ABSOLUTE_ERROR,
 *     CROSS_ENTROPY,
 *     KULLBACK_LEIBLER,
 *     NEGATIVE_LOG_LIKELIHOOD,
 *     POISSON
 *
 */
public enum LossFunctionType {
    MEAN_SQUARED_ERROR,
    MEAN_ABSOLUTE_ERROR,
    CROSS_ENTROPY,
    KULLBACK_LEIBLER,
    NEGATIVE_LOG_LIKELIHOOD,
    POISSON

}

