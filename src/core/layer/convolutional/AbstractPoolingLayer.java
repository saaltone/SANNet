/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.layer.convolutional;

import core.layer.AbstractExecutionLayer;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.io.Serial;
import java.util.HashSet;

/**
 * Implements AbstractPoolingLayer which implements common functionality for pooling layer.
 *
 */
public abstract class AbstractPoolingLayer extends AbstractExecutionLayer {

    @Serial
    private static final long serialVersionUID = -8711824454978626258L;

    /**
     * Parameter name types for pooling layer.
     *     - filterSize size of filter. Default size 2.<br>
     *     - filterRowSize row size of filter. Default size 2.<br>
     *     - filterColumnSize column size of filter. Default size 2.<br>
     *     - stride: size of stride. Default size 1.<br>
     *     - avgPool: if true does average pooling otherwise does max pooling.<br>
     *
     */
    private final static String paramNameTypes = "(filters:INT), " +
            "(filterSize:INT), " +
            "(filterRowSize:INT), " +
            "(filterColumnSize:INT), " +
            "(stride:INT)";

    /**
     * Defines width of incoming image.
     *
     */
    private int previousLayerWidth;

    /**
     * Defines height of incoming image.
     *
     */
    private int previousLayerHeight;

    /**
     * Defines height of incoming image.
     *
     */
    private int previousLayerDepth;

    /**
     * Defines width of outgoing image.
     *
     */
    private int layerWidth;

    /**
     * Defines height of outgoing image.
     *
     */
    private int layerHeight;

    /**
     * Row size for filter.
     *
     */
    private int filterRowSize;

    /**
     * Column size for filter.
     *
     */
    private int filterColumnSize;

    /**
     * Defines stride i.e. size of step when moving over image.
     *
     */
    private int stride;

    /**
     * Input matrices for procedure construction.
     *
     */
    private MMatrix inputs;

    /**
     * Constructor for AbstractExecutionLayer.
     *
     * @param layerIndex layer Index.
     * @param initialization initialization function for weight maps (not relevant for pooling layer).
     * @param params parameters for pooling layer.
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
    }

    /**
     * Returns parameters used for pooling layer.
     *
     * @return parameters used for pooling layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + AbstractPoolingLayer.paramNameTypes;
    }

    /**
     * Sets parameters used for max pooling layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - filterSize size of filter. Default size 2.<br>
     *     - filterRowSize row size of filter. Default size 2.<br>
     *     - filterColumnSize column size of filter. Default size 2.<br>
     *     - stride: size of stride. Default size 1.<br>
     *
     * @param params parameters used for pooling layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("filterSize")) {
            filterRowSize = filterColumnSize = params.getValueAsInteger("filterSize");
            if (filterRowSize < 2) throw new NeuralNetworkException("Filter size must be at least 2.");
        }
        if (params.hasParam("filterRowSize")) {
            filterRowSize = params.getValueAsInteger("filterRowSize");
            if (filterRowSize < 2) throw new NeuralNetworkException("Filter row size must be at least 2.");
        }
        if (params.hasParam("filterColumnSize")) {
            filterColumnSize = params.getValueAsInteger("filterColumnSize");
            if (filterColumnSize < 2) throw new NeuralNetworkException("Filter column size must be at least 2.");
        }
        if (params.hasParam("stride")) {
            stride = params.getValueAsInteger("stride");
            if (stride < 1) throw new NeuralNetworkException("Stride must be at least 1.");
        }
    }

    /**
     * Checks if layer is recurrent layer type.
     *
     * @return always false.
     */
    public boolean isRecurrentLayer() { return false; }

    /**
     * Checks if layer is convolutional layer type.
     *
     * @return always true.
     */
    public boolean isConvolutionalLayer() { return true; }

    /**
     * Initializes pooling layer.<br>
     * Sets input and output dimensions.<br>
     *
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     */
    public void initialize() throws NeuralNetworkException {
        previousLayerWidth = getPreviousLayerWidth();
        previousLayerHeight = getPreviousLayerHeight();
        previousLayerDepth = getPreviousLayerDepth();

        if ((previousLayerWidth - filterRowSize) % stride != 0)  throw new NeuralNetworkException("Pooling layer widthIn: " + previousLayerWidth + " - filterRowSize: " + filterRowSize + " must be divisible by stride: " + stride);
        if ((previousLayerHeight - filterColumnSize) % stride != 0)  throw new NeuralNetworkException("Pooling layer heightIn: " + previousLayerHeight + " - filterColumnSize: " + filterColumnSize + " must be divisible by stride: " + stride);

        layerWidth = ((previousLayerWidth - filterRowSize) / stride) + 1;
        layerHeight = ((previousLayerHeight - filterColumnSize) / stride) + 1;

        if (layerWidth < 1) throw new NeuralNetworkException("Pooling layer width cannot be less than 1: " + layerWidth);
        if (layerHeight < 1) throw new NeuralNetworkException("Pooling layer height cannot be less than 1: " + layerHeight);
        if (previousLayerDepth < 1) throw new NeuralNetworkException("Pooling layer depth cannot be less than 1: " + previousLayerDepth);

        setLayerWidth(layerWidth);
        setLayerHeight(layerHeight);
        setLayerDepth(previousLayerDepth);
    }

    /**
     * Returns input matrix for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public MMatrix getInputMatrices(boolean resetPreviousInput) throws MatrixException {
        inputs = new MMatrix(previousLayerDepth, "Inputs");
        for (int index = 0; index < previousLayerDepth; index++) inputs.put(index, new DMatrix(previousLayerWidth, previousLayerHeight, "Input" + index));
        return inputs;
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public MMatrix getForwardProcedure() throws MatrixException {
        MMatrix outputs = new MMatrix(previousLayerDepth, "Outputs");

        int size = inputs.size();
        for (int channelIndex = 0; channelIndex < size; channelIndex++) {
            Matrix input = inputs.get(channelIndex);
            input.setStride(stride);
            input.setFilterRowSize(filterRowSize);
            input.setFilterColumnSize(filterColumnSize);

            Matrix output;
            outputs.put(channelIndex, output = new DMatrix(layerWidth, layerHeight, "Output" + channelIndex));

            executePoolingOperation(input, output);
        }

        return outputs;
    }

    /**
     * Executes pooling operation.
     *
     * @param input input matrix.
     * @param output output matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract void executePoolingOperation(Matrix input, Matrix output) throws MatrixException;

    /**
     * Returns matrices for which gradient is not calculated.
     *
     * @return matrices for which gradient is not calculated.
     */
    protected HashSet<Matrix> getStopGradients() {
        return new HashSet<>();
    }

    /**
     * Returns constant matrices.
     *
     * @return constant matrices.
     */
    protected HashSet<Matrix> getConstantMatrices() {
        return new HashSet<>();
    }

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        String layerDetailsByName = "";
        layerDetailsByName += "Pooling type: " + getPoolingType() + ", ";
        layerDetailsByName += "Filter row size: " + filterRowSize + ", ";
        layerDetailsByName += "Filter column size: " + filterColumnSize + ", ";
        layerDetailsByName += "Stride: " + stride;
        return layerDetailsByName;
    }

    /**
     * Returns pooling type.
     *
     * @return pooling type.
     */
    protected abstract String getPoolingType();

}