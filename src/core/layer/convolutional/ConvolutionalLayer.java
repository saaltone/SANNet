/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.layer.convolutional;

import core.NeuralNetworkException;
import core.activation.ActivationFunction;
import core.layer.AbstractExecutionLayer;
import utils.*;
import utils.matrix.*;

import java.util.HashMap;

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
    private int previousLayerWidth;

    /**
     * Defines height of incoming image.
     *
     */
    private int previousLayerHeight;

    /**
     * Defines number of channels (depth) of incoming image.
     *
     */
    private int previousLayerDepth;

    /**
     * Defines number of filters.
     *
     */
    private int numberOfFilters;

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
     * Defines dilation step for filter.
     *
     */
    private int dilation = 1;

    /**
     * True is filter weights are regulated otherwise weights are not regulated.
     *
     */
    private boolean regulateWeights = false;

    /**
     * Tree map for filter maps (weights).
     *
     */
    private final HashMap<Integer, Matrix> filterWeights = new HashMap<>();

    /**
     * Tree map for biases.
     *
     */
    private final HashMap<Integer, Matrix> filterBiases = new HashMap<>();

    /**
     * Activation function for convolutional layer.
     *
     */
    protected ActivationFunction activationFunction;

    /**
     * If true convolution operation and gradient is calculated as convolution (flipped filter) otherwise as cross-correlation.
     *
     */
    private boolean asConvolution = true;

    /**
     * Input matrices for procedure construction.
     *
     */
    private MMatrix inputs;

    /**
     * Constructor for convolutional layer.
     *
     * @param layerIndex layer Index.
     * @param activationFunction activation function used.
     * @param initialization initialization function for weight maps.
     * @param params parameters for convolutional layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public ConvolutionalLayer(int layerIndex, ActivationFunction activationFunction, Initialization initialization, String params) throws DynamicParamException, NeuralNetworkException, MatrixException {
        super (layerIndex, initialization, params);
        setParams(new DynamicParam(params, getParamDefs()));
        if (activationFunction != null) this.activationFunction = activationFunction;
        else this.activationFunction = new ActivationFunction(UnaryFunctionType.ELU);
    }

    /**
     * Returns parameters used for convolutional layer.
     *
     * @return parameters used for convolutional layer.
     */
    public HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>(super.getParamDefs());
        paramDefs.put("filters", DynamicParam.ParamType.INT);
        paramDefs.put("filterSize", DynamicParam.ParamType.INT);
        paramDefs.put("stride", DynamicParam.ParamType.INT);
        paramDefs.put("dilation", DynamicParam.ParamType.INT);
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
     *     - dilation: dilation step for filter. Default step 1.<br>
     *     - regulateWeights: true if filter weights are regulated otherwise false (default false).<br>
     *     - asConvolution: true if convolutional layer applies convolution operation otherwise applies crosscorrelation (default true).<br>
     *
     * @param params parameters used for convolutional layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("filters")) numberOfFilters = params.getValueAsInteger("filters");
        if (params.hasParam("filterSize")) filterSize = params.getValueAsInteger("filterSize");
        if (params.hasParam("stride")) stride = params.getValueAsInteger("stride");
        if (params.hasParam("dilation")) dilation = params.getValueAsInteger("dilation");
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
     * Initializes weights and biases.<br>
     *
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     */
    public void initialize() throws NeuralNetworkException {
        int dilatedSize = filterSize + (filterSize - 1) * (dilation - 1);

        previousLayerWidth = getPreviousLayerWidth();
        previousLayerHeight = getPreviousLayerHeight();
        previousLayerDepth = getPreviousLayerDepth();

        if ((previousLayerWidth - dilatedSize) % stride != 0)  throw new NeuralNetworkException("Convolutional layer widthIn: " + previousLayerWidth + " - filterSize: " + filterSize + " must be divisible by stride: " + stride);
        if ((previousLayerHeight - dilatedSize) % stride != 0)  throw new NeuralNetworkException("Convolutional layer heightIn: " + previousLayerHeight + " - filterSize: " + filterSize + " must be divisible by stride: " + stride);

        int layerWidth = ((previousLayerWidth - dilatedSize) / stride) + 1;
        int layerHeight = ((previousLayerHeight - dilatedSize) / stride) + 1;

        if (layerWidth < 1) throw new NeuralNetworkException("Convolutional layer width cannot be less than 1: " + layerWidth);
        if (layerHeight < 1) throw new NeuralNetworkException("Convolutional layer height cannot be less than 1: " + layerHeight);
        if (numberOfFilters < 1) throw new NeuralNetworkException("At least one filter must be defined");

        setLayerWidth(layerWidth);
        setLayerHeight(layerHeight);
        setLayerDepth(numberOfFilters);

        int inputSize = previousLayerDepth * dilatedSize * dilatedSize;
        int outputSize = numberOfFilters * dilatedSize * dilatedSize;
        for (int filterIndex = 0; filterIndex < numberOfFilters; filterIndex++) {

            Matrix filterWeight = new DMatrix(dilatedSize, dilatedSize, this.initialization, inputSize, outputSize, "Wf" + filterIndex);
            filterWeights.put(filterIndex, filterWeight);
            registerWeight(filterWeight, regulateWeights, true);

            Matrix filterBias = new DMatrix(0, "Bf" + filterIndex);
            filterBiases.put(filterIndex, filterBias);
            registerWeight(filterBias, false, false);

        }
    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets previous input.
     * @return input matrices for procedure construction.
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
        MMatrix outputs = new MMatrix(numberOfFilters, "Outputs");
        for (int sampleIndex = 0; sampleIndex < inputs.size() / previousLayerDepth; sampleIndex++) {
            for (int filterIndex = 0; filterIndex < numberOfFilters; filterIndex++) {
                Matrix output = filterBiases.get(filterIndex);
                for (int channelIndex = 0; channelIndex < previousLayerDepth; channelIndex++) {
                    Matrix Wf = filterWeights.get(filterIndex);
                    Matrix input = inputs.get(getInputIndex(channelIndex, sampleIndex, previousLayerDepth));
                    input.setStride(stride);
                    input.setDilation(dilation);
                    input.setFilterSize(filterSize + (filterSize - 1) * (dilation - 1));
                    input.setRegularize(true);
                    if (asConvolution) output = output.add(input.convolve(Wf));
                    else output = output.add(input.crosscorrelate(Wf));
                }
                output.setNormalize(true);
                output = activationFunction.applyFunction(output);
                output.setName("Output" + filterIndex);
                outputs.put(getOutputIndex(filterIndex, sampleIndex, numberOfFilters), output);
            }
        }
        return outputs;
    }

    /**
     * Returns flat filter index by filterIndex, channelIndex and number of channels for a convolutional layer.
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
    private int getOutputIndex(int filterIndex, int sampleIndex, int filters) {
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
    private int getInputIndex(int channelIndex, int sampleIndex, int channels) {
        return channelIndex + sampleIndex * channels;
    }

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        String layerDetailsByName = "";
        layerDetailsByName += "Pooling type: " + (asConvolution ? "Convolution" : "Cross-correlation") + ", ";
        layerDetailsByName += "Number of filters: " + numberOfFilters + ", ";
        layerDetailsByName += "Filter size: " + filterSize + ", ";
        layerDetailsByName += "Stride: " + stride + ", ";
        layerDetailsByName += "Dilation: " + dilation + ", ";
        layerDetailsByName += "Activation function: " + activationFunction.getName();
        return layerDetailsByName;
    }

}
