/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer.normalization;

import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.Initialization;
import utils.matrix.MatrixException;

/**
 * Implements layer for instance normalization.
 *
 */
public class InstanceNormalization extends AbstractNormalization {

    /**
     * Constructor for instance normalization layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for instance normalization layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     */
    public InstanceNormalization(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, params, 1);
    }

}
