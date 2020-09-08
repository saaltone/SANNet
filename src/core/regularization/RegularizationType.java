/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.regularization;

/**
 * Enum for regularization type.<br>
 * Currently supported types are:
 *     DROPOUT,
 *     GRADIENT_CLIPPING,
 *     L1_REGULARIZATION,
 *     L2_REGULARIZATION,
 *     LP_REGULARIZATION,
 *     WEIGHT_NOISING
 *
 */
public enum RegularizationType {

    DROPOUT,
    GRADIENT_CLIPPING,
    L1_REGULARIZATION,
    L2_REGULARIZATION,
    LP_REGULARIZATION,
    WEIGHT_NOISING

}
