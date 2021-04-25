/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.normalization;

import core.NeuralNetworkException;
import utils.DynamicParamException;

import java.io.Serializable;

/**
 * Factory class that creates normalizer instances.<br>
 *
 */
public class NormalizationFactory implements Serializable {

    /**
     * Creates normalizer instance with given type with defined parameters.
     *
     * @param normalizationType type of normalizer.
     * @param params parameters for normalizer.
     * @return constructed normalizer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public static Normalization create(NormalizationType normalizationType, String params) throws DynamicParamException {
        return switch (normalizationType) {
            case BATCH_NORMALIZATION -> (params == null) ? new BatchNormalization() : new BatchNormalization(params);
            case LAYER_NORMALIZATION -> (params == null) ? new LayerNormalization() : new LayerNormalization(params);
            case WEIGHT_NORMALIZATION -> (params == null) ? new WeightNormalization() : new WeightNormalization(params);
        };
    }

    /**
     * Creates normalizer instance with given type with defined parameters.
     *
     * @param normalizationType type of normalizer.
     * @return constructed normalizer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public static Normalization create(NormalizationType normalizationType) throws DynamicParamException {
        return create(normalizationType, null);
    }

    /**
     * Returns type of normalizer.
     *
     * @param normalization normalizer.
     * @return type of normalizer.
     * @throws NeuralNetworkException throws exception is normalizer is of unknown type.
     */
    public static NormalizationType getNormalizationType(Normalization normalization) throws NeuralNetworkException {
        if (normalization instanceof BatchNormalization) return NormalizationType.BATCH_NORMALIZATION;
        if (normalization instanceof LayerNormalization) return NormalizationType.LAYER_NORMALIZATION;
        if (normalization instanceof WeightNormalization) return NormalizationType.WEIGHT_NORMALIZATION;
        throw new NeuralNetworkException("Unknown normalization type");
    }

}
