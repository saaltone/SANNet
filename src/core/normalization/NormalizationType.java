/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.normalization;

/**
 * Defines supported normalization types.
 *
 */
public enum NormalizationType {

    /**
     * Batch normalization that normalizes over batch per feature.
     *
     */
    BATCH_NORMALIZATION,

    /**
     * Layer normalization that normalizes per sample over all features.
     *
     */
    LAYER_NORMALIZATION,

    /**
     * Weight normalization that normalizes each weight using norm of weight.
     *
     */
    WEIGHT_NORMALIZATION,

}
