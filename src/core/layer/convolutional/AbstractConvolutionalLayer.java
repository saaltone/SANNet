/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.layer.convolutional;

import core.activation.ActivationFunction;
import core.layer.AbstractExecutionLayer;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.io.Serial;
import java.util.HashMap;
import java.util.HashSet;

/**
 * Implements AbstractConvolutionalLayer which implements common functionality for convolutional layer.
 *
 */
public abstract class AbstractConvolutionalLayer extends AbstractExecutionLayer {

    @Serial
    private static final long serialVersionUID = -5937040933415814069L;

    /**
     * Parameter name types for AbstractConvolutionalLayer.
     *     - filters: number of filters.<br>
     *     - regulateWeights: true if filter weights are regulated otherwise false (default false).<br>
     *
     */
    private final static String paramNameTypes = "(filters:INT), " +
            "(regulateWeights:BOOLEAN)";

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
    private int filterRowSize;

    /**
     * Defines filter dimension in terms of columns.
     *
     */
    private int filterColumnSize;

    /**
     * Defines stride i.e. size of step when moving filter over image.
     *
     */
    private int stride;

    /**
     * Defines dilation step for filter.
     *
     */
    private int dilation;

    /**
     * True is filter weights are regulated otherwise weights are not regulated.
     *
     */
    private boolean regulateWeights;

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
     * Input matrices for procedure construction.
     *
     */
    private MMatrix inputs;

    /**
     * Constructor for AbstractConvolutionalLayer.
     *
     * @param layerIndex layer Index.
     * @param activationFunction activation function used.
     * @param initialization initialization function for weight maps.
     * @param params parameters for convolutional layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public AbstractConvolutionalLayer(int layerIndex, ActivationFunction activationFunction, Initialization initialization, String params) throws DynamicParamException, NeuralNetworkException, MatrixException {
        super (layerIndex, initialization, params);
        if (activationFunction != null) this.activationFunction = activationFunction;
        else this.activationFunction = new ActivationFunction(UnaryFunctionType.ELU);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        filterRowSize = 3;
        filterColumnSize = 3;
        stride = 1;
        dilation = 1;
        regulateWeights = false;
    }

    /**
     * Returns parameters used for AbstractConvolutionalLayer.
     *
     * @return parameters used for AbstractConvolutionalLayer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + AbstractConvolutionalLayer.paramNameTypes;
    }

    /**
     * Sets parameters used for AbstractConvolutionalLayer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - filters: number of filters.<br>
     *     - regulateWeights: true if filter weights are regulated otherwise false (default false).<br>
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

        int layerWidth = getCurrentLayerWidth();
        int layerHeight = getCurrentLayerHeight();

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
     * Returns current layer width based on layer dimensions.
     *
     * @return current layer width
     * @throws NeuralNetworkException throws exception if layer dimensions do not match.
     */
    protected int getCurrentLayerWidth() throws NeuralNetworkException {
        // Enlarge filter size with dilation factor.
        filterRowSize = filterRowSize + (filterRowSize - 1) * (dilation - 1);

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
        // Enlarge filter size with dilation factor.
        filterColumnSize = filterColumnSize + (filterColumnSize - 1) * (dilation - 1);

        if ((previousLayerHeight - filterColumnSize) % stride != 0)  throw new NeuralNetworkException("Convolutional layer heightIn: " + previousLayerHeight + " - filterColumnSize: " + filterColumnSize + " must be divisible by stride: " + stride + " using dilation: " + dilation);

        return ((previousLayerHeight - filterColumnSize) / stride) + 1;
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
                output = executeConvolutionalOperation(input, Wf, output);
            }
            output = output.apply(activationFunction);
            output.setName("Output" + filterIndex);
            outputs.put(filterIndex, output);
        }
        return outputs;
    }

    /**
     * Executes convolutional operation.
     *
     * @param input input matrix.
     * @param filter filter matrix.
     * @param output output matrix.
     * @return result of convolutional operation.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract Matrix executeConvolutionalOperation(Matrix input, Matrix filter, Matrix output) throws MatrixException;

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
        layerDetailsByName += "Convolution type: " + getConvolutionType() + ", ";
        layerDetailsByName += "Number of filters: " + numberOfFilters + ", ";
        layerDetailsByName += "Filter row size: " + filterRowSize + ", ";
        layerDetailsByName += "Filter column size: " + filterColumnSize + ", ";
        layerDetailsByName += "Stride: " + stride + ", ";
        layerDetailsByName += "Dilation: " + dilation + ", ";
        layerDetailsByName += "Activation function: " + activationFunction.getName();
        return layerDetailsByName;
    }

    /**
     * Returns convolution type.
     *
     * @return convolution type.
     */
    protected abstract String getConvolutionType();

}
