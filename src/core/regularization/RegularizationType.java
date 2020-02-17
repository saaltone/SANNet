/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.regularization;

/**
 * Enum for regularization type.<br>
 * Currenty supported types are:
 *     DROPOUT,
 *     GRADIENT_CLIPPING,
 *     L1_REGULARIZATION,
 *     L2_REGULARIZATION,
 *     LP_REGULARIZATION
 *
 */
public enum RegularizationType {

    DROPOUT,
    GRADIENT_CLIPPING,
    L1_REGULARIZATION,
    L2_REGULARIZATION,
    LP_REGULARIZATION

}
