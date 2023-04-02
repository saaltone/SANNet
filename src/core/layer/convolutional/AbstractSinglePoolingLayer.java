/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer.convolutional;

import core.layer.AbstractExecutionLayer;
import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.util.HashSet;
import java.util.TreeMap;

/**
 * Implements abstract single pooling layer which implements common functionality for pooling layer.
 *
 */
public abstract class AbstractSinglePoolingLayer extends AbstractExecutionLayer {

    /**
     * Parameter name types for abstract single pooling layer.
     *     - filterSize size of filter. Default size 2.<br>
     *     - filterRowSize row size of filter. Default size 2.<br>
     *     - filterColumnSize column size of filter. Default size 2.<br>
     *     - stride: size of stride. Default size 1.<br>
     *
     */
    private final static String paramNameTypes = "(filterSize:INT), " +
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
    private TreeMap<Integer, Matrix> inputs;

    /**
     * Constructor for abstract single pooling layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight maps (not relevant for pooling layer).
     * @param params parameters for abstract single pooling layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception setting of activation function fails.
     */
    public AbstractSinglePoolingLayer(int layerIndex, Initialization initialization, String params) throws DynamicParamException, NeuralNetworkException {
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
     * Returns parameters used for abstract pooling layer.
     *
     * @return parameters used for abstract pooling layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + AbstractSinglePoolingLayer.paramNameTypes;
    }

    /**
     * Sets parameters used for abstract single pooling layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - filterSize size of filter. Default size 2.<br>
     *     - filterRowSize row size of filter. Default size 2.<br>
     *     - filterColumnSize column size of filter. Default size 2.<br>
     *     - stride: size of stride. Default size 1.<br>
     *
     * @param params parameters used for abstract single pooling layer.
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
     * Checks if layer works with recurrent layers.
     *
     * @return if true layer works with recurrent layers otherwise false.
     */
    public boolean worksWithRecurrentLayer() {
        return true;
    }

    /**
     * Initializes neural network layer dimensions.
     *
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     */
    public void initializeDimensions() throws NeuralNetworkException {
        previousLayerWidth = getDefaultPreviousLayer().getLayerWidth();
        previousLayerHeight = getDefaultPreviousLayer().getLayerHeight();

        if (previousLayerWidth < 1) throw new NeuralNetworkException("Default previous layer width must be positive. Invalid value: " + previousLayerWidth);
        if (previousLayerHeight < 1) throw new NeuralNetworkException("Default previous height width must be positive. Invalid value: " + previousLayerHeight);

        if ((previousLayerWidth - filterRowSize) % stride != 0)  throw new NeuralNetworkException("Pooling layer widthIn: " + previousLayerWidth + " - filterRowSize: " + filterRowSize + " must be divisible by stride: " + stride);
        if ((previousLayerHeight - filterColumnSize) % stride != 0)  throw new NeuralNetworkException("Pooling layer heightIn: " + previousLayerHeight + " - filterColumnSize: " + filterColumnSize + " must be divisible by stride: " + stride);

        int layerWidth = ((previousLayerWidth - filterRowSize) / stride) + 1;
        int layerHeight = ((previousLayerHeight - filterColumnSize) / stride) + 1;

        if (layerWidth < 1) throw new NeuralNetworkException("Pooling layer width cannot be less than 1: " + layerWidth);
        if (layerHeight < 1) throw new NeuralNetworkException("Pooling layer height cannot be less than 1: " + layerHeight);

        setLayerWidth(layerWidth);
        setLayerHeight(layerHeight);
        setLayerDepth(1);
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
        Matrix input = new DMatrix(previousLayerWidth, previousLayerHeight, 1);
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
        input.setFilterRowSize(filterRowSize);
        input.setFilterColumnSize(filterColumnSize);
        input.setStride(stride);

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
     * Returns matrices for which gradient is not calculated.
     *
     * @return matrices for which gradient is not calculated.
     */
    public HashSet<Matrix> getStopGradients() {
        return new HashSet<>();
    }

    /**
     * Returns constant matrices.
     *
     * @return constant matrices.
     */
    public HashSet<Matrix> getConstantMatrices() {
        return new HashSet<>();
    }

    /**
     * Returns number of truncated steps for gradient calculation. -1 means no truncation.
     *
     * @return number of truncated steps.
     */
    protected int getTruncateSteps() {
        return -1;
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
