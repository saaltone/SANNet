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

/**
 * Implements depth-wise separable convolution layer.
 *
 */
public class DSConvolutionLayer extends AbstractDSConvolutionalLayer {

    /**
     * Constructor for depth-wise separable convolution layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight maps.
     * @param params parameters for depth-wise separable convolution layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     */
    public DSConvolutionLayer(int layerIndex, Initialization initialization, String params) throws DynamicParamException, NeuralNetworkException {
        super (layerIndex, initialization, params);
    }

    /**
     * Executes convolutional operation.
     *
     * @param input  input matrix.
     * @param filter filter matrix.
     * @return result of convolutional operation.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix executeConvolutionalOperation(Matrix input, Matrix filter) throws MatrixException {
        return input.convolve(filter);
    }

    /**
     * Returns convolution type.
     *
     * @return convolution type.
     */
    protected String getConvolutionType() {
        return "Convolution";
    }

}
