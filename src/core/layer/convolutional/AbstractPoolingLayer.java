/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.layer.convolutional;

import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.Initialization;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.TreeMap;

/**
 * Implements abstract pooling layer which implements common functionality for pooling layer.
 *
 */
public abstract class AbstractPoolingLayer extends AbstractConvolutionLayer {

    /**
     * Parameter name types for abstract pooling layer.
     *     - filterSize size of filter. Default size 2.<br>
     *     - filterRowSize row size of filter. Default size 2.<br>
     *     - filterColumnSize column size of filter. Default size 2.<br>
     *     - stride: size of stride. Default size 1.<br>
     *     - dilation: dilation step for filter. Default step 1.<br>
     *
     */
    private final static String paramNameTypes = "(filters:INT), " +
            "(filterSize:INT), " +
            "(filterRowSize:INT), " +
            "(filterColumnSize:INT), " +
            "(stride:INT), " +
            "(dilation:INT), ";

    /**
     * Input matrices for procedure construction.
     *
     */
    private TreeMap<Integer, Matrix> inputs;

    /**
     * Constructor for abstract pooling layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight maps (not relevant for pooling layer).
     * @param params parameters for abstract pooling layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception setting of activation function fails.
     */
    public AbstractPoolingLayer(int layerIndex, Initialization initialization, String params) throws DynamicParamException, NeuralNetworkException {
        super (layerIndex, initialization, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        filterRowSize = 2;
        filterColumnSize = 2;
        stride = 1;
        dilation = 1;
    }

    /**
     * Returns parameters used for abstract pooling layer.
     *
     * @return parameters used for abstract pooling layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + AbstractPoolingLayer.paramNameTypes;
    }

    /**
     * Sets parameters used for abstract pooling layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - filterSize size of filter. Default size 2.<br>
     *     - filterRowSize row size of filter. Default size 2.<br>
     *     - filterColumnSize column size of filter. Default size 2.<br>
     *     - stride: size of stride. Default size 1.<br>
     *     - dilation: dilation step for filter. Default step 1.<br>
     *
     * @param params parameters used for abstract pooling layer.
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
     * Initializes neural network layer dimensions.
     *
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     */
    public void initializeDimensions() throws NeuralNetworkException {
        super.initializeDimensions();

        setLayerDepth(previousLayerDepth);
    }

    /**
     * Returns weight set.
     *
     * @return weight set.
     */
    protected WeightSet getWeightSet() {
        return null;
    }

    /**
     * Initializes neural network layer weights.
     *
     */
    public void initializeWeights() {
    }

    /**
     * Returns input matrix for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     */
    public TreeMap<Integer, Matrix> getInputMatrices(boolean resetPreviousInput) {
        inputs = new TreeMap<>();
        Matrix input = new DMatrix(previousLayerWidth, previousLayerHeight, previousLayerDepth);
        input.setName("Input" + getDefaultPreviousLayer().getLayerIndex());
        inputs.put(0, input);
        return inputs;
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix getForwardProcedure() throws MatrixException {
        Matrix input = inputs.get(0);
        input.setStride(stride);
        input.setFilterRowSize(filterRowSize);
        input.setFilterColumnSize(filterColumnSize);

        Matrix output = executePoolingOperation(input);
        output.setName("Output");

        return output;
    }

    /**
     * Executes pooling operation.
     *
     * @param input input matrix.
     * @return output matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract Matrix executePoolingOperation(Matrix input) throws MatrixException;

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        String layerDetailsByName = super.getLayerDetailsByName() + ", ";
        layerDetailsByName += "Pooling type: " + getPoolingType();
        return layerDetailsByName;
    }

    /**
     * Returns pooling type.
     *
     * @return pooling type.
     */
    protected abstract String getPoolingType();

}
