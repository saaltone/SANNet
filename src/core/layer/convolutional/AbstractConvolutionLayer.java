/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer.convolutional;

import core.layer.AbstractExecutionLayer;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.Initialization;

/**
 * Implements abstract convolutional layer which implements common functionality for convolutional layer.
 *
 */
public abstract class AbstractConvolutionLayer extends AbstractExecutionLayer {

    /**
     * Defines width of incoming image.
     *
     */
    protected int previousLayerWidth;

    /**
     * Defines height of incoming image.
     *
     */
    protected int previousLayerHeight;

    /**
     * Defines number of channels (depth) of incoming image.
     *
     */
    protected int previousLayerDepth;

    /**
     * Defines filter dimension in terms of rows.
     *
     */
    protected int filterRowSize;

    /**
     * Defines filter dimension in terms of columns.
     *
     */
    protected int filterColumnSize;

    /**
     * Defines stride i.e. size of step when moving filter over image.
     *
     */
    protected int stride;

    /**
     * Defines dilation step for filter.
     *
     */
    protected int dilation;

    /**
     * Constructor for abstract convolutional layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight maps.
     * @param params parameters for abstract convolutional layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     */
    public AbstractConvolutionLayer(int layerIndex, Initialization initialization, String params) throws DynamicParamException, NeuralNetworkException {
        super (layerIndex, initialization, params);
    }

    /**
     * Sets filter row size.
     *
     * @param filterRowSize filter row size.
     */
    protected void setFilterRowSize(int filterRowSize) {
        this.filterRowSize = filterRowSize;
    }

    /**
     * Sets filter column size.
     *
     * @param filterColumnSize filter column size.
     */
    protected void setFilterColumnSize(int filterColumnSize) {
        this.filterColumnSize = filterColumnSize;
    }

    /**
     * Sets stride.
     *
     * @param stride stride.
     */
    protected void setStride(int stride) {
        this.stride = stride;
    }

    /**
     * Sets dilation.
     *
     * @param dilation dilation.
     */
    protected void setDilation(int dilation) {
        this.dilation = dilation;
    }

    /**
     * Initializes neural network layer dimensions.
     *
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     */
    public void initializeDimensions() throws NeuralNetworkException {
        previousLayerWidth = getDefaultPreviousLayer().getLayerWidth();
        previousLayerHeight = getDefaultPreviousLayer().getLayerHeight();
        previousLayerDepth = getDefaultPreviousLayer().getLayerDepth();

        if (previousLayerWidth < 1) throw new NeuralNetworkException("Default previous layer width must be positive. Invalid value: " + previousLayerWidth);
        if (previousLayerHeight < 1) throw new NeuralNetworkException("Default previous height width must be positive. Invalid value: " + previousLayerHeight);
        if (previousLayerDepth < 1) throw new NeuralNetworkException("Default previous depth width must be positive. Invalid value: " + previousLayerDepth);

        if (getCurrentLayerWidth() < 1) throw new NeuralNetworkException("Convolutional layer width cannot be less than 1: " + getCurrentLayerWidth());
        if (getCurrentLayerHeight() < 1) throw new NeuralNetworkException("Convolutional layer height cannot be less than 1: " + getCurrentLayerHeight());

        setLayerWidth(getCurrentLayerWidth());
        setLayerHeight(getCurrentLayerHeight());
    }

    /**
     * Returns current layer width based on layer dimensions.
     *
     * @return current layer width
     * @throws NeuralNetworkException throws exception if layer dimensions do not match.
     */
    protected int getCurrentLayerWidth() throws NeuralNetworkException {
        if ((previousLayerWidth - filterRowSize) % stride != 0)  throw new NeuralNetworkException("Convolutional layer widthIn: " + previousLayerWidth + " - filterRowSize: " + filterRowSize + " must be divisible by stride: " + stride + " using dilation: " + dilation);

        return ((previousLayerWidth - filterRowSize) / stride) + 1;
    }

    /**
     * Returns current layer height based on layer dimensions.
     *
     * @return current layer height
     * @throws NeuralNetworkException throws exception if layer dimensions do not match.
     */
    protected int getCurrentLayerHeight() throws NeuralNetworkException {
        if ((previousLayerHeight - filterColumnSize) % stride != 0)  throw new NeuralNetworkException("Convolutional layer heightIn: " + previousLayerHeight + " - filterColumnSize: " + filterColumnSize + " must be divisible by stride: " + stride + " using dilation: " + dilation);

        return ((previousLayerHeight - filterColumnSize) / stride) + 1;
    }

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        String layerDetailsByName = "";
        layerDetailsByName += "Filter row size: " + filterRowSize + ", ";
        layerDetailsByName += "Filter column size: " + filterColumnSize + ", ";
        layerDetailsByName += "Stride: " + stride + ", ";
        layerDetailsByName += "Dilation: " + dilation;
        return layerDetailsByName;
    }

}
