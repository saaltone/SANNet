/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.normalization;

/**
 * Defines supported normalization types.<br>
 * <br>
 * Currently supported types are:
 *     BATCH_NORMALIZATION,
 *     LAYER_NORMALIZATION,
 *     WEIGHT_NORMALIZATION,
 *
 */
public enum NormalizationType {

    BATCH_NORMALIZATION,
    LAYER_NORMALIZATION,
    WEIGHT_NORMALIZATION,

}
