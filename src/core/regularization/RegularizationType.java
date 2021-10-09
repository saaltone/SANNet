/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.regularization;

/**
 * Defines supported regularization types.
 *
 */
public enum RegularizationType {

    /**
     * Inverted drop out
     *
     */
    DROPOUT,

    /**
     * Gradient clipping
     *
     */
    GRADIENT_CLIPPING,

    /**
     * L1 (lasso) regularization
     *
     */
    L1_REGULARIZATION,

    /**
     * L2 (ridge) regularization
     *
     */
    L2_REGULARIZATION,

    /**
     * Lp regularization that applies regularization of order p
     *
     */
    LP_REGULARIZATION,

    /**
     * Weight noising
     *
     */
    WEIGHT_NOISING

}
