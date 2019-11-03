/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.layer.convolutional;

import core.NeuralNetworkException;
import core.activation.ActivationFunction;
import core.layer.AbstractExecutionLayer;
import core.layer.AbstractLayer;
import core.normalization.Normalization;
import utils.*;
import utils.matrix.DMatrix;
import utils.matrix.Init;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashMap;
import java.util.HashSet;

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
    private int widthIn;

    /**
     * Defines height of incoming image.
     *
     */
    private int heightIn;

    /**
     * Defines height of incoming image.
     *
     */
    private int depthIn;

    /**
     * Defines width of outgoing image.
     *
     */
    private int widthOut;

    /**
     * Defines height of outgoing image.
     *
     */
    private int heightOut;

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
    private boolean avgPool = false;

    /**
     * Input matrices for procedure construction.
     *
     */
    private Sample inputs;

    /**
     * Constructor for pooling layer.
     *
     * @param parent reference to parent layer.
     * @param activation activation function used (not relevant for pooling layer).
     * @param initialization intialization function for weight maps (not relevant for pooling layer).
     * @param params parameters for pooling layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception setting of activation function fails.
     */
    public PoolingLayer(AbstractLayer parent, ActivationFunction activation, Init initialization, String params) throws DynamicParamException, NeuralNetworkException {
        super (parent, activation, initialization, params);
    }

    /**
     * Returns parameters used for pooling layer.
     *
     * @return parameters used for pooling layer.
     */
    public HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
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
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("poolSize")) poolSize = params.getValueAsInteger("poolSize");
        if (params.hasParam("stride")) stride = params.getValueAsInteger("stride");
        if (params.hasParam("avgPool")) avgPool = params.getValueAsBoolean("avgPool");
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
        widthIn = parent.getBackward().getPLayerWidth();
        heightIn = parent.getBackward().getPLayerHeight();
        depthIn = parent.getBackward().getPLayerDepth();

        if ((widthIn - poolSize) % stride != 0)  throw new NeuralNetworkException("Pooling layer widthIn: " + widthIn + " - poolSize: " + poolSize + " must be divisible by stride: " + stride);
        if ((heightIn - poolSize) % stride != 0)  throw new NeuralNetworkException("Pooling layer heigthIn: " + heightIn + " - poolSize: " + poolSize + " must be divisible by stride: " + stride);

        widthOut = ((widthIn - poolSize) / stride) + 1;
        heightOut = ((heightIn - poolSize) / stride) + 1;

        if (widthOut < 1) throw new NeuralNetworkException("Pooling layer width cannot be less than 1: " + widthOut);
        if (heightOut < 1) throw new NeuralNetworkException("Pooling layer heigth cannot be less than 1: " + heightOut);
        if (depthIn < 1) throw new NeuralNetworkException("Pooling layer depth cannot be less than 1: " + depthIn);

        parent.setWidth(widthOut);
        parent.setHeight(heightOut);
        parent.setDepth(depthIn);
    }

    /**
     * Resets input.
     *
     * @param resetPreviousInput if true resets also previous input.
     */
    protected void resetInput(boolean resetPreviousInput) throws MatrixException {
        inputs = new Sample(depthIn);
        for (int index = 0; index < depthIn; index++) inputs.put(index, new DMatrix(widthIn, heightIn));
    }

    /**
     * Returns input matrix for procedure construction.
     *
     * @return input matrix for procedure construction.
     */
    protected Sample getInputMatrices() {
        return inputs;
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @param normalizers normalizers for layer normalization.
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Sample getForwardProcedure(HashSet<Normalization> normalizers) throws MatrixException {
        Sample outputs = new Sample(depthIn);
        Matrix output;

        for (int channelIndex = 0; channelIndex < inputs.size(); channelIndex++) {
            Matrix input = inputs.get(channelIndex);

            input.setStride(stride);
            input.setPoolSize(poolSize);

            outputs.put(channelIndex, output = new DMatrix(widthOut, heightOut));

            if (!avgPool) input.maxPool(output, new int[output.getRows()][output.getCols()][2]);
            else input.avgPool(output);
        }

        return outputs;
    }

}
