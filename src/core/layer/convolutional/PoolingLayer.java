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
import utils.*;

import java.util.HashMap;
import java.util.TreeMap;

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
     * Value is true is next layer is convolutional layer.
     *
     */
    private boolean toNonConvolutionalLayer;

    /**
     * Intermediate tree map structure to store pooling values and their locations in forward step.<br>
     * This cache saves execution time and simplifies backward processing step for max pooling.<br>
     *
     */
    private transient TreeMap<Integer, int[][][]> argsMax;

    /**
     * Tree map for flattened outputs after forward processing.<br>
     * This is relevant if next layer is non-convolutional layer.<br>
     *
     */
    private transient TreeMap<Integer, Matrix> fouts;

    /**
     * True if layer allows to return flattened outputs. In practise this happens when layer is not processing internally.
     *
     */
    private transient boolean allowFlattening = false;

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
     * Gets parameters used for pooling layer.
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
     * Initializes max pooling layer.<br>
     * Sets input and output dimensions.<br>
     *
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     */
    public void initialize() throws NeuralNetworkException {
        toNonConvolutionalLayer = forward.hasNLayer() && forward.getNLayer().isConvolutionalLayer();

        widthIn = backward.getPLayer().getWidth();
        heightIn = backward.getPLayer().getHeight();
        int depthIn = backward.getPLayer().getDepth();

        if ((widthIn - poolSize) % stride != 0)  throw new NeuralNetworkException("Pooling layer widthIn: " + widthIn + " - poolSize: " + poolSize + " must be divisible by stride: " + stride);
        if ((heightIn - poolSize) % stride != 0)  throw new NeuralNetworkException("Pooling layer heigthIn: " + heightIn + " - poolSize: " + poolSize + " must be divisible by stride: " + stride);

        widthOut = ((widthIn - poolSize) / stride) + 1;
        heightOut = ((heightIn - poolSize) / stride) + 1;

        if (widthOut < 1) throw new NeuralNetworkException("Pooling layer width cannot be less than 1: " + widthOut);
        if (heightOut < 1) throw new NeuralNetworkException("Pooling layer heigth cannot be less than 1: " + heightOut);
        if (depthIn < 1) throw new NeuralNetworkException("Pooling layer depth cannot be less than 1: " + depthIn);

        setWidth(widthOut);
        setHeight(heightOut);
        setDepth(depthIn);

    }

    /**
     * Takes single forward processing step for pooling layer to process input(s).<br>
     * Applies max or average pooling and produces outputs.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    public void forwardProcess() throws MatrixException, NeuralNetworkException {
        allowFlattening = false;
        parent.resetOuts();
        argsMax = new TreeMap<>();

        Matrix output;
        for (Integer index : getOutsP().keySet()) {
            Matrix input = getOutsP().get(index);
            parent.getOuts().put(index, output = new DMatrix(widthOut, heightOut));
            int [][][] argsAt = null;
            if (!avgPool) {
                argsAt = new int[output.getRows()][output.getCols()][2];
                argsMax.put(index, argsAt);
            }
            input.setSliceSize(poolSize, poolSize);
            for (int row = 0; row < output.getRows(); row = row + stride) {
                for (int col = 0; col < output.getCols(); col = col + stride) {
                    if (!avgPool) input.sliceAt(row, col).maxPool(output, argsAt);
                    else input.sliceAt(row, col).avgPool(output);
                }
            }
        }

        fouts = null;
        if (forward == null) parent.updateOutputError();
        else if (toNonConvolutionalLayer) fouts = flattenOutput(parent.getOuts());
        allowFlattening = true;

    }

    /**
     * Returns outputs of pooling layer.
     *
     * @return flattened outputs if next layer is non-convolutional layer otherwise normal outputs.
     */
    public TreeMap<Integer, Matrix> getOuts(TreeMap<Integer, Matrix> outs) {
        return toNonConvolutionalLayer && allowFlattening ? fouts : outs;
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @param input input of forward procedure.
     * @param reset reset recurring inputs of procedure.
     * @return output of forward procedure.
     */
    protected Matrix getForwardProcedure(Matrix input, boolean reset) {
        return null;
    }

    /**
     * Takes single backward processing step for pooling layer to process input(s).<br>
     * Calculates gradient (error signal) towards previous layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backwardProcess() throws MatrixException {
        allowFlattening = false;
        parent.resetOutGrads();

        backward.resetGrad();

        TreeMap<Integer, Matrix> dEosN = toNonConvolutionalLayer ? unflattenOutput(parent.getdEosN()) : parent.getdEosN();
        Matrix dEo;
        for (Integer index : parent.getOuts().keySet()) {
            Matrix dEi = dEosN.get(index);

            parent.getdEos().put(index, dEo = new DMatrix(widthIn, heightIn));
            dEi.setSliceSize(1, 1);
            for (int row = 0; row < dEi.getRows(); row = row + stride) {
                for (int col = 0; col < dEi.getCols(); col = col + stride) {
                    if (!avgPool) dEi.sliceAt(row, col).maxPoolGrad(dEo, argsMax.get(index)[row][col]);
                    else dEi.sliceAt(row, col).avgPoolGrad(dEo);
                }
            }
        }

        backward.sumGrad();
        allowFlattening = true;

    }

}
