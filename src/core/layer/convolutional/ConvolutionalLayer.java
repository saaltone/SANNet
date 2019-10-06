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
 * Defines class for convolutional layer.<br>
 * <br>
 * Reference: https://pdfs.semanticscholar.org/5d79/11c93ddcb34cac088d99bd0cae9124e5dcd1.pdf<br>
 *
 */
public class ConvolutionalLayer extends AbstractExecutionLayer {

    private static final long serialVersionUID = -7210767738512077627L;

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
     * Defines number of channels (depthIn) for incoming image.
     *
     */
    private int channels;

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
     * Defines number of filters.
     *
     */
    private int filters;

    /**
     * Defines filter size (filter size x filter size).
     *
     */
    private int filterSize;

    /**
     * Defines stride i.e. size of step when moving filter over image.
     *
     */
    private int stride;

    /**
     * True is filter weights are regulated otherwise weights are not regulated.
     *
     */
    private boolean regulateWeights;

    /**
     * Value is true is next layer is convolutional layer.
     *
     */
    private boolean toNonConvolutionalLayer;

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
     * Tree map for filter maps (weights).
     *
     */
    private final HashMap<Integer, Matrix> Wfs = new HashMap<>();

    /**
     * Tree map for biases.
     *
     */
    private final HashMap<Integer, Matrix> Bfs = new HashMap<>();

    /**
     * If true convolution operation and gradient is calculated as convolution (flipped filter) otherwise as cross-correlation.
     *
     */
    private final boolean asConvolution = true;

    /**
     * Constructor for convolutional layer.
     *
     * @param parent reference to parent layer.
     * @param activation activation function used.
     * @param initialization intialization function for weight maps.
     * @param params parameters for convolutional layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception setting of activation function fails.
     */
    public ConvolutionalLayer(AbstractLayer parent, ActivationFunction activation, Init initialization, String params) throws DynamicParamException, NeuralNetworkException {
        super (parent, activation, initialization, params);
    }

    /**
     * Gets parameters used for convolutional layer.
     *
     * @return parameters used for convolutional layer.
     */
    public HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("filters", DynamicParam.ParamType.INT);
        paramDefs.put("filterSize", DynamicParam.ParamType.INT);
        paramDefs.put("stride", DynamicParam.ParamType.INT);
        paramDefs.put("regulateWeights", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for convolutional layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - filters: number of filters.<br>
     *     - filterSize size of filter.<br>
     *     - stride: size of stride.<br>
     *     - regulateWeights: true is filter weights are regulated otherwise false (default false).<br>
     *
     * @param params parameters used for convolutional layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("filters")) filters = params.getValueAsInteger("filters");
        if (params.hasParam("filterSize")) filterSize = params.getValueAsInteger("filterSize");
        if (params.hasParam("stride")) stride = params.getValueAsInteger("stride");
        if (params.hasParam("regulateWeights")) regulateWeights = params.getValueAsBoolean("regulateWeights");
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
     * Initializes convolutional layer.<br>
     * Sets input and output dimensions.<br>
     * Initializes weight and biases and their gradients.<br>
     *
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     */
    public void initialize() throws NeuralNetworkException {
        toNonConvolutionalLayer = forward.hasNLayer() && forward.getNLayer().isConvolutionalLayer();

        widthIn = backward.getPLayer().getWidth();
        heightIn = backward.getPLayer().getHeight();
        channels = backward.getPLayer().getDepth();

        if ((widthIn - filterSize) % stride != 0)  throw new NeuralNetworkException("Convolutional layer widthIn: " + widthIn + " - filterSize: " + filterSize + " must be divisible by stride: " + stride);
        if ((heightIn - filterSize) % stride != 0)  throw new NeuralNetworkException("Convolutional layer heigthIn: " + heightIn + " - filterSize: " + filterSize + " must be divisible by stride: " + stride);

        widthOut = ((widthIn - filterSize) / stride) + 1;
        heightOut = ((heightIn - filterSize) / stride) + 1;

        if (widthOut < 1) throw new NeuralNetworkException("Convolutional layer width cannot be less than 1: " + widthOut);
        if (heightOut < 1) throw new NeuralNetworkException("Convolutional layer heigth cannot be less than 1: " + heightOut);
        if (filters < 1) throw new NeuralNetworkException("At least one filter must be defined");

        setWidth(widthOut);
        setHeight(heightOut);
        setDepth(filters);

        int inputAmount = channels * filterSize * filterSize;
        int outputAmount = filters * filterSize * filterSize;
        for (int filterIndex = 0; filterIndex < filters; filterIndex++) {
            for (int channelIndex = 0; channelIndex < channels; channelIndex++) {
                int filter = getFilterIndex(filterIndex, channelIndex, channels);
                Matrix Wf = new DMatrix(filterSize, filterSize, this.initialization, inputAmount, outputAmount);
                Wfs.put(filter, Wf);

                backward.registerWeight(Wf, true, regulateWeights, true);
            }

            Matrix Bf = new DMatrix(1, 1, Init.ZERO);
            Bfs.put(filterIndex, Bf);

            backward.registerWeight(Bf, true, false, false);
        }
    }

    /**
     * Takes single forward processing step for convolutional layer to process input(s).<br>
     * Applies filters and produces outputs.<br>
     * Additionally applies any regularization defined for the layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    public void forwardProcess() throws MatrixException, NeuralNetworkException {
        allowFlattening = false;

        backward.regulateForwardPre(getOutsP(), -1);
        backward.normalizeForwardPre(getOutsP(), channels);

        parent.resetOuts();

        int sampleIndex = 0;
        int channelIndex = 0;

        for (Integer inIndex : getOutsP().keySet()) {
            backward.regulateForwardPre(getOutsP(), inIndex);

            Matrix input = getOutsP().get(inIndex);

            for (int filterIndex = 0; filterIndex < filters; filterIndex++) {
                int outIndex = getOutIndex(filterIndex, sampleIndex, filters);

                Matrix output = getElement(outIndex, parent.getOuts(), widthOut, heightOut);
                Matrix Wf = Wfs.get(getFilterIndex(filterIndex, channelIndex, channels));
                Matrix Bf = Bfs.get(filterIndex);

                if (asConvolution) input.convolve(Wf, output, stride);
                else input.crosscorrelate(Wf, output, stride);

                if (channelIndex == 0) output.add(Bf.getValue(0, 0), output);
            }

            if (++channelIndex == channels) {
                channelIndex = 0;
                sampleIndex++;
            }

        }


        for (Matrix out : parent.getOuts().values()) activation.applyFunction(out, true);

        backward.regulateForwardPost(parent.getOuts());
        backward.normalizeForwardPost(parent.getOuts());

        fouts = null;
        if (forward == null) parent.updateOutputError();
        else if (toNonConvolutionalLayer) fouts = flattenOutput(parent.getOuts());
        allowFlattening = true;

    }

    /**
     * Returns outputs of convolutional layer.
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
     * Takes single backward processing step for convolutional layer to process input(s).<br>
     * Calculates gradients for weights and biases and gradient (error signal) towards previous layer.<br>
     * Additionally applies any regularization defined for the layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backwardProcess() throws MatrixException {
        allowFlattening = false;
        parent.resetOutGrads();

        backward.resetGrad();

        backward.regulateBackward(-1);

        TreeMap<Integer, Matrix> dEosN = toNonConvolutionalLayer ? unflattenOutput(parent.getdEosN()) : parent.getdEosN();

        int sampleIndex = 0;
        int filterIndex = 0;

        for (Integer outIndex : parent.getOuts().keySet()) {
            backward.regulateBackward(outIndex);

            Matrix dEo = dEosN.get(outIndex);
            Matrix dEi = activation.applyGradient(parent.getOuts().get(outIndex), dEo);

            for (int channelIndex = 0; channelIndex < channels; channelIndex++) {
                int inIndex = getInIndex(channelIndex, sampleIndex, channels);

                Matrix dEoP = getElement(inIndex, parent.getdEos(), widthIn, heightIn);
                Matrix outP = getOutsP().get(inIndex);

                int flatFilterIndex = getFilterIndex(filterIndex, channelIndex, channels);
                Matrix Wf = Wfs.get(flatFilterIndex);

                Matrix dW = getElement(sampleIndex, backward.getdWs(Wf), filterSize, filterSize);

                if (asConvolution) dEi.convolveGrad(Wf, outP, dW, dEoP, stride);
                else dEi.crosscorrelateGrad(Wf, outP, dW, dEoP, stride);
            }

            Matrix dB = getElement(sampleIndex, backward.getdWs(Bfs.get(filterIndex)), 1, 1);
            dB.add(dEi.sum(), dB);

            if (++filterIndex == filters) {
                filterIndex = 0;
                sampleIndex++;
            }

        }

        backward.normalizeBackward();

        backward.sumGrad();

        allowFlattening = true;

    }

    /**
     * Returns element with certain index.<br>
     * If element does not exists it creates of with given width and height and stores it into outs tree map.<br>
     *
     * @param index index of element to be returned.
     * @param elements tree map of existing elements.
     * @param width widght of element.
     * @param height height of element.
     * @return requested element.
     */
    private Matrix getElement(int index, TreeMap<Integer, Matrix> elements, int width, int height) {
        Matrix element;
        if (elements.containsKey(index)) element = elements.get(index);
        else elements.put(index, element = new DMatrix(width, height));
        return element;
    }

}
