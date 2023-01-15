/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer.convolutional;

import core.activation.ActivationFunction;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Initialization;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements convolution layer.
 *
 */
public class ConvolutionLayer extends AbstractConvolutionalLayer {

    /**
     * Parameter name types for convolution layer.
     *     - filterSize size of filter. Default value 3.<br>
     *     - filterRowSize size of filter in terms of rows. Overrides filterSize parameter. Default value 3.<br>
     *     - filterColumnSize size of filter in terms of columns. Overrides filterSize parameter. Default value 3.<br>
     *     - stride: size of stride. Default size 1.<br>
     *     - dilation: dilation step for filter. Default step 1.<br>
     *
     */
    private final static String paramNameTypes = "(filterSize:INT), " +
            "(filterRowSize:INT), " +
            "(filterColumnSize:INT), " +
            "(stride:INT), " +
            "(dilation:INT), ";


    /**
     * Constructor for convolution layer.
     *
     * @param layerIndex layer index
     * @param activationFunction activation function used.
     * @param initialization initialization function for weight maps.
     * @param params parameters for convolution layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     */
    public ConvolutionLayer(int layerIndex, ActivationFunction activationFunction, Initialization initialization, String params) throws DynamicParamException, NeuralNetworkException {
        super (layerIndex, activationFunction, initialization, params);
    }

    /**
     * Returns parameters used for convolution layer.
     *
     * @return parameters used for convolution layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + ConvolutionLayer.paramNameTypes;
    }

    /**
     * Sets parameters used for convolution layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - filterSize size of filter. Default value 3.<br>
     *     - filterRowSize size of filter in terms of rows. Overrides filterSize parameter. Default value 3.<br>
     *     - filterColumnSize size of filter in terms of columns. Overrides filterSize parameter. Default value 3.<br>
     *     - stride: size of stride. Default size 1.<br>
     *     - dilation: dilation step for filter. Default step 1.<br>
     *
     * @param params parameters used for convolution layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("filterSize")) {
            int filterSize = params.getValueAsInteger("filterSize");
            if (filterSize < 1) throw new NeuralNetworkException("Filter size must be at least 1.");
            setFilterRowSize(filterSize);
            setFilterColumnSize(filterSize);
        }
        if (params.hasParam("filterRowSize")) {
            int filterRowSize = params.getValueAsInteger("filterRowSize");
            if (filterRowSize < 1) throw new NeuralNetworkException("Filter row size must be at least 1.");
            setFilterRowSize(filterRowSize);
        }
        if (params.hasParam("filterColumnSize")) {
            int filterColumnSize = params.getValueAsInteger("filterColumnSize");
            if (filterColumnSize < 1) throw new NeuralNetworkException("Filter column size must be at least 1.");
            setFilterColumnSize(filterColumnSize);
        }
        if (params.hasParam("stride")) {
            int stride = params.getValueAsInteger("stride");
            if (stride < 1) throw new NeuralNetworkException("Stride must be at least 1.");
            setStride(stride);
        }
        if (params.hasParam("dilation")) {
            int dilation = params.getValueAsInteger("dilation");
            if (dilation < 1) throw new NeuralNetworkException("Dilation must be at least 1.");
            setDilation(dilation);
        }
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
        return output.add(input.convolve(filter));
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
