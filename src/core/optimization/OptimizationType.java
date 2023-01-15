/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.optimization;

/**
 * Defines supported optimization types.
 *
 */
public enum OptimizationType {

    /**
     * Gradient descent
     *
     */
    GRADIENT_DESCENT,

    /**
     * Momentum gradient descent
     *
     */
    MOMENTUM_GRADIENT_DESCENT,

    /**
     * Nesterov accelerated gradient
     *
     */
    NESTEROV_ACCELERATED_GRADIENT,

    /**
     * Adagrad
     *
     */
    ADAGRAD,

    /**
     * Adadelta
     *
     */
    ADADELTA,

    /**
     * RMPProp
     *
     */
    RMSPROP,

    /**
     * Adam
     *
     */
    ADAM,

    /**
     * Adamax
     *
     */
    ADAMAX,

    /**
     * NAdam
     *
     */
    NADAM,

    /**
     * RAdam
     *
     */
    RADAM,

    /**
     * AMSGrad
     *
     */
    AMSGRAD,

    /**
     * Resilient propagation
     *
     */
    RESILIENT_PROPAGATION

}
