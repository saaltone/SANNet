/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
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
    private int stride = 1;

    /**
     * True is filter weights are regulated otherwise weights are not regulated.
     *
     */
    private boolean regulateWeights;

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
    private boolean asConvolution = true;

    /**
     * Input matrices for procedure construction.
     *
     */
    private Sample inputs;

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
     * Returns parameters used for convolutional layer.
     *
     * @return parameters used for convolutional layer.
     */
    public HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("filters", DynamicParam.ParamType.INT);
        paramDefs.put("filterSize", DynamicParam.ParamType.INT);
        paramDefs.put("stride", DynamicParam.ParamType.INT);
        paramDefs.put("regulateWeights", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("asConvolution", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for convolutional layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - filters: number of filters.<br>
     *     - filterSize size of filter.<br>
     *     - stride: size of stride. Default size 1.<br>
     *     - regulateWeights: true if filter weights are regulated otherwise false (default false).<br>
     *     - asConvolution: true if convolutional layer applies convolution operation otherwise applies crosscorrelation (default true).<br>
     *
     * @param params parameters used for convolutional layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("filters")) filters = params.getValueAsInteger("filters");
        if (params.hasParam("filterSize")) filterSize = params.getValueAsInteger("filterSize");
        if (params.hasParam("stride")) stride = params.getValueAsInteger("stride");
        if (params.hasParam("regulateWeights")) regulateWeights = params.getValueAsBoolean("regulateWeights");
        if (params.hasParam("asConvolution")) asConvolution = params.getValueAsBoolean("asConvolution");
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
    public void initialize() throws MatrixException, NeuralNetworkException {
        widthIn = parent.getBackward().getPLayerWidth();
        heightIn = parent.getBackward().getPLayerHeight();
        channels = parent.getBackward().getPLayerDepth();

        if ((widthIn - filterSize) % stride != 0)  throw new NeuralNetworkException("Convolutional layer widthIn: " + widthIn + " - filterSize: " + filterSize + " must be divisible by stride: " + stride);
        if ((heightIn - filterSize) % stride != 0)  throw new NeuralNetworkException("Convolutional layer heigthIn: " + heightIn + " - filterSize: " + filterSize + " must be divisible by stride: " + stride);

        int widthOut = ((widthIn - filterSize) / stride) + 1;
        int heightOut = ((heightIn - filterSize) / stride) + 1;

        if (widthOut < 1) throw new NeuralNetworkException("Convolutional layer width cannot be less than 1: " + widthOut);
        if (heightOut < 1) throw new NeuralNetworkException("Convolutional layer heigth cannot be less than 1: " + heightOut);
        if (filters < 1) throw new NeuralNetworkException("At least one filter must be defined");

        parent.setWidth(widthOut);
        parent.setHeight(heightOut);
        parent.setDepth(filters);

        int inputAmount = channels * filterSize * filterSize;
        int outputAmount = filters * filterSize * filterSize;
        for (int filterIndex = 0; filterIndex < filters; filterIndex++) {

            for (int channelIndex = 0; channelIndex < channels; channelIndex++) {
                int filter = getFilterIndex(filterIndex, channelIndex, channels);
                Matrix Wf = new DMatrix(filterSize, filterSize, this.initialization, inputAmount, outputAmount);
                Wfs.put(filter, Wf);
                parent.getBackward().registerWeight(Wf, true, regulateWeights, true);
            }

            Matrix Bf = new DMatrix(1, 1, Init.CONSTANT);
            Bfs.put(filterIndex, Bf);

            parent.getBackward().registerWeight(Bf, true, false, false);
        }
    }

    /**
     * Resets input.
     *
     * @param resetPreviousInput if true resets also previous input.
     */
    protected void resetInput(boolean resetPreviousInput) throws MatrixException {
        inputs = new Sample(channels);
        for (int index = 0; index < channels; index++) inputs.put(index, new DMatrix(widthIn, heightIn));
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
        Sample outputs = new Sample(filters);
        for (int sampleIndex = 0; sampleIndex < inputs.size() / channels; sampleIndex++) {
            for (int filterIndex = 0; filterIndex < filters; filterIndex++) {
                Matrix output = Bfs.get(filterIndex);
                for (int channelIndex = 0; channelIndex < channels; channelIndex++) {
                    Matrix Wf = Wfs.get(getFilterIndex(filterIndex, channelIndex, channels));
                    Matrix input = inputs.get(getInIndex(channelIndex, sampleIndex, channels));
                    input.setStride(stride);
                    if (asConvolution) output = output.add(input.convolve(Wf));
                    else output = output.add(input.crosscorrelate(Wf));
                }
                if (normalizers.size() > 0) output.setNormalization(normalizers);
                output = activation.applyFunction(output);
                outputs.put(getOutIndex(filterIndex, sampleIndex, filters), output);
            }
        }
        return outputs;
    }

    /**
     * Returns flat filter index by filterIndex, depthIndex and number of channels for a convolutional layer.
     *
     * @param filterIndex index for filter.
     * @param channelIndex index for input channel.
     * @param channels number of input channels.
     * @return flat filter index.
     */
    private int getFilterIndex(int filterIndex, int channelIndex, int channels) {
        return channelIndex + filterIndex * channels;
    }

    /**
     * Returns output index by filterIndex, sampleIndex and number of filters for a convolutional layer.
     *
     * @param filterIndex index for filter.
     * @param sampleIndex index for current sample.
     * @param filters number of filters.
     * @return output index.
     */
    private int getOutIndex(int filterIndex, int sampleIndex, int filters) {
        return filterIndex + sampleIndex * filters;
    }

    /**
     * Returns input index by filterIndex, sampleIndex and number of channels for a convolutional layer.
     *
     * @param channelIndex index for channel.
     * @param sampleIndex index for current sample.
     * @param channels number of channels.
     * @return input index.
     */
    private int getInIndex(int channelIndex, int sampleIndex, int channels) {
        return channelIndex + sampleIndex * channels;
    }

}
