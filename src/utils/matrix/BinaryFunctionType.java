/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package utils.matrix;

/**
 * Defines supported binary functions.
 *
 */
public enum BinaryFunctionType {

    /**
     * Mean squared error
     *
     */
    MEAN_SQUARED_ERROR,

    /**
     * Mean squared logarithmic error
     *
     */
    MEAN_SQUARED_LOGARITHMIC_ERROR,

    /**
     * Mean absolute error
     *
     */
    MEAN_ABSOLUTE_ERROR,

    /**
     * Mean absolute percentage error
     *
     */
    MEAN_ABSOLUTE_PERCENTAGE_ERROR,

    /**
     * Cross-entropy (log) error
     *
     */
    CROSS_ENTROPY,

    /**
     * Kullback-Leibler error
     *
     */
    KULLBACK_LEIBLER,

    /**
     * Negative log likelihood error
     *
     */
    NEGATIVE_LOG_LIKELIHOOD,

    /**
     * Poisson error
     *
     */
    POISSON,

    /**
     * Hinge error
     *
     */
    HINGE,

    /**
     * Squared hinge error
     *
     */
    SQUARED_HINGE,

    /**
     * Huber error
     *
     */
    HUBER,

    /**
     * Direct (user definable) gradient.
     *
     */
    DIRECT_GRADIENT,

    /**
     * Policy gradient
     *
     */
    POLICY_GRADIENT,

    /**
     * Policy value i.e. first entry contains value function error and rest policy gradient values for policy function.
     *
     */
    POLICY_VALUE,

    /**
     * Power function with definable power value
     *
     */
    POW,

    /**
     * Maximum between two entries
     *
     */
    MAX,

    /**
     * Minimum between two entries
     *
     */
    MIN,

    /**
     * Customer (user definable) function
     *
     */
    CUSTOM

}
