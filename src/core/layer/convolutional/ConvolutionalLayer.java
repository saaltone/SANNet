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

import java.io.Serial;
import java.util.HashMap;

/**
 * Implements convolutional layer.<br>
 * <br>
 * Reference: https://pdfs.semanticscholar.org/5d79/11c93ddcb34cac088d99bd0cae9124e5dcd1.pdf<br>
 *
 */
public class ConvolutionalLayer extends AbstractExecutionLayer {

    @Serial
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
     * Defines filter dimension in terms of rows.
     *
     */
    private int filterRowSize = 3;

    /**
     * Defines filter dimension in terms of columns.
     *
     */
    private int filterColumnSize = 3;

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
     * If true convolution operation and gradient is calculated as convolution (flipped filter) otherwise as crosscorrelation.
     *
     */
    private boolean asConvolution = false;

    /**
     * If true applies Winograd convolution operation otherwise applies normal convolution or crosscorrelation (default false).
     *
     */
    private boolean asWinogradConvolution = false;

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
        paramDefs.put("filterRowSize", DynamicParam.ParamType.INT);
        paramDefs.put("filterColumnSize", DynamicParam.ParamType.INT);
        paramDefs.put("stride", DynamicParam.ParamType.INT);
        paramDefs.put("dilation", DynamicParam.ParamType.INT);
        paramDefs.put("regulateWeights", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("asConvolution", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("asWinogradConvolution", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for convolutional layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - filters: number of filters.<br>
     *     - filterSize size of filter. Default value 3.<br>
     *     - filterRowSize size of filter in terms of rows. Overrides filterSize parameter. Default value 3.<br>
     *     - filterColumnSize size of filter in terms of columns. Overrides filterSize parameter. Default value 3.<br>
     *     - stride: size of stride. Default size 1.<br>
     *     - dilation: dilation step for filter. Default step 1.<br>
     *     - regulateWeights: true if filter weights are regulated otherwise false (default false).<br>
     *     - asConvolution: true if convolutional layer applies convolution operation otherwise applies crosscorrelation (default false).<br>
     *     - asWinogradConvolution: true if convolutional layer applies Winograd convolution operation otherwise applies normal convolution or crosscorrelation (default false).<br>
     *
     * @param params parameters used for convolutional layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("filters")) {
            numberOfFilters = params.getValueAsInteger("filters");
            if (numberOfFilters < 1) throw new NeuralNetworkException("At least one filter must be defined. Number of filters cannot be: " + numberOfFilters);
        }
        if (params.hasParam("filterSize")) {
            filterRowSize = filterColumnSize = params.getValueAsInteger("filterSize");
            if (filterRowSize < 1) throw new NeuralNetworkException("Filter size must be at least 1.");
        }
        if (params.hasParam("filterRowSize")) {
            filterRowSize = params.getValueAsInteger("filterRowSize");
            if (filterRowSize < 1) throw new NeuralNetworkException("Filter row size must be at least 1.");
        }
        if (params.hasParam("filterColumnSize")) {
            filterColumnSize = params.getValueAsInteger("filterColumnSize");
            if (filterColumnSize < 1) throw new NeuralNetworkException("Filter column size must be at least 1.");
        }
        if (params.hasParam("stride")) {
            stride = params.getValueAsInteger("stride");
            if (stride < 1) throw new NeuralNetworkException("Stride must be at least 1.");
        }
        if (params.hasParam("dilation")) {
            dilation = params.getValueAsInteger("dilation");
            if (dilation < 1) throw new NeuralNetworkException("Dilation must be at least 1.");
        }
        if (params.hasParam("regulateWeights")) regulateWeights = params.getValueAsBoolean("regulateWeights");
        if (params.hasParam("asConvolution")) asConvolution = params.getValueAsBoolean("asConvolution");
        if (params.hasParam("asWinogradConvolution")) asWinogradConvolution = params.getValueAsBoolean("asWinogradConvolution");
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

        previousLayerWidth = getPreviousLayerWidth();
        previousLayerHeight = getPreviousLayerHeight();
        previousLayerDepth = getPreviousLayerDepth();

        int layerWidth;
        int layerHeight;

        if (asWinogradConvolution) {
            asConvolution = false;
            filterRowSize = filterColumnSize = 3;
            stride = 2;
            dilation = 1;

            layerWidth = previousLayerWidth - 2;
            layerHeight = previousLayerHeight - 2;
        }
        else {
            // Enlarge filter size with dilation factor.
            filterRowSize = filterRowSize + (filterRowSize - 1) * (dilation - 1);
            filterColumnSize = filterColumnSize + (filterColumnSize - 1) * (dilation - 1);

            if ((previousLayerWidth - filterRowSize) % stride != 0)  throw new NeuralNetworkException("Convolutional layer widthIn: " + previousLayerWidth + " - filterRowSize: " + filterRowSize + " must be divisible by stride: " + stride + " using dilation: " + dilation);
            if ((previousLayerHeight - filterColumnSize) % stride != 0)  throw new NeuralNetworkException("Convolutional layer heightIn: " + previousLayerHeight + " - filterColumnSize: " + filterColumnSize + " must be divisible by stride: " + stride + " using dilation: " + dilation);

            layerWidth = ((previousLayerWidth - filterRowSize) / stride) + 1;
            layerHeight = ((previousLayerHeight - filterColumnSize) / stride) + 1;
        }


        if (layerWidth < 1) throw new NeuralNetworkException("Convolutional layer width cannot be less than 1: " + layerWidth);
        if (layerHeight < 1) throw new NeuralNetworkException("Convolutional layer height cannot be less than 1: " + layerHeight);
        if (numberOfFilters < 1) throw new NeuralNetworkException("At least one filter must be defined");

        setLayerWidth(layerWidth);
        setLayerHeight(layerHeight);
        setLayerDepth(numberOfFilters);

        for (int filterIndex = 0; filterIndex < numberOfFilters; filterIndex++) {

            Matrix filterWeight = new DMatrix(filterRowSize, filterColumnSize, this.initialization, previousLayerDepth * filterRowSize * filterColumnSize, numberOfFilters * filterRowSize * filterColumnSize, "Wf" + filterIndex);
            filterWeights.put(filterIndex, filterWeight);
            registerWeight(filterWeight, regulateWeights, true);

            Matrix filterBias = new DMatrix(0, "Bf" + filterIndex);
            filterBiases.put(filterIndex, filterBias);
            registerWeight(filterBias, false, false);

        }
    }

    /**
     * Reinitializes layer.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void reinitialize() throws MatrixException, NeuralNetworkException {
        for (Matrix weight : filterWeights.values()) weight.initialize(this.initialization, previousLayerDepth * filterRowSize * filterColumnSize, numberOfFilters * filterRowSize * filterColumnSize);
        for (Matrix bias : filterBiases.values()) bias.reset();

        super.reinitialize();
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
        for (int filterIndex = 0; filterIndex < numberOfFilters; filterIndex++) {
            Matrix output = filterBiases.get(filterIndex);
            for (int channelIndex = 0; channelIndex < previousLayerDepth; channelIndex++) {
                Matrix Wf = filterWeights.get(filterIndex);
                Matrix input = inputs.get(channelIndex);
                input.setStride(stride);
                input.setDilation(dilation);
                input.setFilterRowSize(filterRowSize);
                input.setFilterColumnSize(filterColumnSize);
                input.setRegularize(true);
                if (asWinogradConvolution) output = output.add(input.winogradConvolve(Wf));
                else {
                    if (asConvolution) output = output.add(input.convolve(Wf));
                    else output = output.add(input.crosscorrelate(Wf));
                }
            }
            output.setNormalize(true);
            output = output.apply(activationFunction);
            output.setName("Output" + filterIndex);
            outputs.put(filterIndex, output);
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
        layerDetailsByName += "Pooling type: " + (asConvolution ? "Convolution" : "Crosscorrelation") + ", ";
        layerDetailsByName += "Number of filters: " + numberOfFilters + ", ";
        layerDetailsByName += "Filter row size: " + filterRowSize + ", ";
        layerDetailsByName += "Filter column size: " + filterColumnSize + ", ";
        layerDetailsByName += "Stride: " + stride + ", ";
        layerDetailsByName += "Dilation: " + dilation + ", ";
        layerDetailsByName += "Activation function: " + activationFunction.getName();
        return layerDetailsByName;
    }

}
