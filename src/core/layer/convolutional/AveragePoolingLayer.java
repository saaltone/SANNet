/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.layer.convolutional;

import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.Initialization;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Defines average pooling layer.
 *
 */
public class AveragePoolingLayer extends AbstractPoolingLayer {

    /**
     * Constructor for AveragePoolingLayer.
     *
     * @param layerIndex layer Index.
     * @param initialization initialization function for weight maps (not relevant for pooling layer).
     * @param params parameters for pooling layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception setting of activation function fails.
     */
    public AveragePoolingLayer(int layerIndex, Initialization initialization, String params) throws DynamicParamException, NeuralNetworkException {
        super (layerIndex, initialization, params);
    }

    /**
     * Executes pooling operation.
     *
     * @param input input matrix.
     * @param output output matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void executePoolingOperation(Matrix input, Matrix output) throws MatrixException {
        input.averagePool(output);
    }

    /**
     * Returns pooling type.
     *
     * @return pooling type.
     */
    protected String getPoolingType() {
        return "Average";
    }

}
