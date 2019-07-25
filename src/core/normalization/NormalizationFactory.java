/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.normalization;

import core.NeuralNetworkException;
import core.layer.Connector;
import utils.DynamicParamException;

import java.io.Serializable;

/**
 * Factory class that creates normalizer instances.
 *
 */
public class NormalizationFactory implements Serializable {


    /**
     * Creates normalizer instance with given type with defined parameters.
     *
     * @param normalizationType type of normalizer.
     * @param connector connector between previous and next layer.
     * @param params parameters for normalizer.
     * @return constructed normalizer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if creation of normalizer fails.
     */
    public static Normalization create(NormalizationType normalizationType, Connector connector, String params) throws DynamicParamException, NeuralNetworkException {
        switch (normalizationType) {
            case BATCH_NORMALIZATION:
                return (params == null) ? new BatchNormalization(connector) : new BatchNormalization(connector, params);
            case LAYER_NORMALIZATION:
                return (params == null) ? new LayerNormalization(connector) : new LayerNormalization(connector, params);
            case WEIGHT_NORMALIZATION:
                return (params == null) ? new WeightNormalization(connector) : new WeightNormalization(connector, params);
        }
        throw new NeuralNetworkException("Creation of normalizer failed.");
    }

    /**
     * Creates normalizer instance with given type with defined parameters.
     *
     * @param normalizationType type of normalizer.
     * @param connector connector between previous and next layer.
     * @return constructed normalizer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if creation of normalizer fails.
     */
    public static Normalization create(NormalizationType normalizationType, Connector connector) throws DynamicParamException, NeuralNetworkException {
        return create(normalizationType, connector, null);
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
