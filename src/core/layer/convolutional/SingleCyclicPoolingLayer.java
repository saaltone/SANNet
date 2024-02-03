/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.layer.convolutional;

import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.Initialization;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashMap;

/**
 * Implements single cyclic pooling layer.<br>
 * Traverses cyclically each filter row and column through step by step and propagates selected row and column.<br>
 *
 */
public class SingleCyclicPoolingLayer extends AbstractSinglePoolingLayer {

    /**
     * Constructor for single cyclic pooling layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight maps (not relevant for pooling layer).
     * @param params parameters for single cyclic pooling layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception setting of activation function fails.
     */
    public SingleCyclicPoolingLayer(int layerIndex, Initialization initialization, String params) throws DynamicParamException, NeuralNetworkException {
        super (layerIndex, initialization, params);
    }

    /**
     * Executes pooling operation.
     *
     * @param input input matrix.
     * @return output matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix executePoolingOperation(Matrix input) throws MatrixException {
        return input.cyclicPool(new HashMap<>());
    }

    /**
     * Returns pooling type.
     *
     * @return pooling type.
     */
    protected String getPoolingType() {
        return "Cyclic";
    }

}
