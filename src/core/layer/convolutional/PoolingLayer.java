/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.layer.convolutional;

import core.NeuralNetworkException;
import core.layer.AbstractExecutionLayer;
import utils.*;
import utils.matrix.*;

import java.util.HashMap;

/**
 * Defines class for pooling layer that executes either average or max pooling.<br>
 * <br>
 * Reference: https://pdfs.semanticscholar.org/5d79/11c93ddcb34cac088d99bd0cae9124e5dcd1.pdf<br>
 *
 */
public class PoolingLayer extends AbstractExecutionLayer {

    private static final long serialVersionUID = 4806254935177730053L;

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
     * Step size of pooling operation.
     *
     */
    private int poolSize;

    /**
     * Defines stride i.e. size of step when moving over image.
     *
     */
    private int stride = 1;

    /**
     * If true executes average pooling otherwise executes max pooling.
     *
     */
    private boolean averagePool = false;

    /**
     * Input matrices for procedure construction.
     *
     */
    private MMatrix inputs;

    /**
     * Constructor for pooling layer.
     *
     * @param layerIndex layer Index.
     * @param initialization initialization function for weight maps (not relevant for pooling layer).
     * @param params parameters for pooling layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception setting of activation function fails.
     */
    public PoolingLayer(int layerIndex, Initialization initialization, String params) throws DynamicParamException, NeuralNetworkException {
        super (layerIndex, initialization, params);
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for pooling layer.
     *
     * @return parameters used for pooling layer.
     */
    public HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>(super.getParamDefs());
        paramDefs.put("poolSize", DynamicParam.ParamType.INT);
        paramDefs.put("stride", DynamicParam.ParamType.INT);
        paramDefs.put("avgPool", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for max pooling layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - poolSize size of pool.<br>
     *     - stride: size of stride. Default size 1.<br>
     *     - avgPool: if true does average pooling otherwise does max pooling.<br>
     *
     * @param params parameters used for pooling layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("poolSize")) poolSize = params.getValueAsInteger("poolSize");
        if (params.hasParam("stride")) stride = params.getValueAsInteger("stride");
        if (params.hasParam("avgPool")) averagePool = params.getValueAsBoolean("avgPool");
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

        if ((previousLayerWidth - poolSize) % stride != 0)  throw new NeuralNetworkException("Pooling layer widthIn: " + previousLayerWidth + " - poolSize: " + poolSize + " must be divisible by stride: " + stride);
        if ((previousLayerHeight - poolSize) % stride != 0)  throw new NeuralNetworkException("Pooling layer heightIn: " + previousLayerHeight + " - poolSize: " + poolSize + " must be divisible by stride: " + stride);

        layerWidth = ((previousLayerWidth - poolSize) / stride) + 1;
        layerHeight = ((previousLayerHeight - poolSize) / stride) + 1;

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
        Matrix output;

        for (int channelIndex = 0; channelIndex < inputs.size(); channelIndex++) {
            Matrix input = inputs.get(channelIndex);

            input.setStride(stride);
            input.setPoolSize(poolSize);

            outputs.put(channelIndex, output = new DMatrix(layerWidth, layerHeight, "Output" + channelIndex));

            if (!averagePool) input.maxPool(output, new int[output.getRows()][output.getColumns()][2]);
            else input.averagePool(output);
        }

        return outputs;
    }

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        String layerDetailsByName = "";
        layerDetailsByName += "Pooling type: " + (averagePool ? "Average" : "Max") + ", ";
        layerDetailsByName += "Pool size: " + poolSize + ", ";
        layerDetailsByName += "Stride: " + stride;
        return layerDetailsByName;
    }

}
