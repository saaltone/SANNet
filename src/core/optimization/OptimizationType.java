/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.optimization;

/**
 * Enum for supported optimization types.<br>
 * Currently supported types are:
 *    GRADIENT_DESCENT,
 *    MOMENTUM_GRADIENT_DESCENT,
 *    NESTEROV_ACCELERATED_GRADIENT,
 *    ADAGRAD,
 *    ADADELTA,
 *    RMSPROP,
 *    ADAM,
 *    ADAMAX,
 *    NADAM,
 *    RADAM,
 *    AMSGRAD,
 *    RESILIENT_PROPAGATION
 *
 */
public enum OptimizationType {

    GRADIENT_DESCENT,
    MOMENTUM_GRADIENT_DESCENT,
    NESTEROV_ACCELERATED_GRADIENT,
    ADAGRAD,
    ADADELTA,
    RMSPROP,
    ADAM,
    ADAMAX,
    NADAM,
    RADAM,
    AMSGRAD,
    RESILIENT_PROPAGATION

}
