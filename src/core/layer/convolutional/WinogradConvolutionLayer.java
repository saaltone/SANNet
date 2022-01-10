/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.layer.convolutional;

import core.activation.ActivationFunction;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.Initialization;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Class that defines Winograd convolution layer with filter size 3x3, stride 1 and dilation 1.
 *
 */
public class WinogradConvolutionLayer extends AbstractConvolutionalLayer {

    /**
     * Constructor for WinogradConvolutionLayer.
     *
     * @param layerIndex layer Index.
     * @param activationFunction activation function used.
     * @param initialization initialization function for weight maps.
     * @param params parameters for convolutional layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     */
    public WinogradConvolutionLayer(int layerIndex, ActivationFunction activationFunction, Initialization initialization, String params) throws DynamicParamException, NeuralNetworkException {
        super (layerIndex, activationFunction, initialization, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        setFilterRowSize(3);
        setFilterColumnSize(3);
        setStride(2);
        setDilation(1);
    }

    /**
     * Returns current layer width based on layer dimensions.
     *
     * @return current layer width
     */
    protected int getCurrentLayerWidth() {
        return getPreviousLayerWidth() - 2;
    }

    /**
     * Returns current layer height based on layer dimensions.
     *
     * @return current layer height
     */
    protected int getCurrentLayerHeight() {
        return getPreviousLayerHeight() - 2;
    }

    /**
     * Executes convolutional operation.
     *
     * @param input input matrix.
     * @param filter filter matrix.
     * @param output output matrix.
     * @return result of convolutional operation.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix executeConvolutionalOperation(Matrix input, Matrix filter, Matrix output) throws MatrixException {
        return output.add(input.winogradConvolve(filter));
    }

    /**
     * Returns convolution type.
     *
     * @return convolution type.
     */
    protected String getConvolutionType() {
        return "Winograd convolution";
    }

}
