/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.layer.convolutional;

import core.activation.ActivationFunction;
import core.layer.AbstractExecutionLayer;
import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeMap;

/**
 * Implements abstract depth-wise separable convolutional layer which implements common functionality for convolutional layer.<br>
 * Reference: https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
 *
 */
public abstract class AbstractDSConvolutionalLayer extends AbstractExecutionLayer {

    /**
     * Parameter name types for abstract depth-wise separable convolutional layer.
     *     - filters: number of filters.<br>
     *     - regulateWeights: true if filter weights are regulated otherwise false (default false).<br>
     *
     */
    private final static String paramNameTypes = "(filters:INT), " +
            "(regulateWeights:BOOLEAN)";

    /**
     * Implements weight set for layer.
     *
     */
    protected class DSConvolutionWeightSet implements WeightSet, Serializable {

        @Serial
        private static final long serialVersionUID = -1987157248603388756L;

        /**
         * Filter row size.
         *
         */
        private final int filterRowSize;

        /**
         * Filter column size.
         *
         */
        private final int filterColumnSize;

        /**
         * Number of filters.
         *
         */
        private final int numberOfFilters;

        /**
         * Treemap for depth-wise filter maps (weights).
         *
         */
        private final HashMap<Integer, Matrix> filterWeightsDepthWise = new HashMap<>();

        /**
         * Tree map for depth-wise biases.
         *
         */
        private final HashMap<Integer, Matrix> filterBiasesDepthWise = new HashMap<>();

        /**
         * Treemap for point-wise filter maps (weights).
         *
         */
        private final HashMap<Integer, HashMap<Integer, Matrix>> filterWeightsPointWise = new HashMap<>();

        /**
         * Treemap for point-wise biases.
         *
         */
        private final HashMap<Integer, Matrix> filterBiasesPointWise = new HashMap<>();

        /**
         * Set of weights.
         *
         */
        private final HashSet<Matrix> weights = new HashSet<>();

        /**
         * Constructor for weight set
         *
         * @param initialization weight initialization function.
         * @param filterRowSize filter row size.
         * @param filterColumnSize filter column size.
         * @param numberOfFilters number of filters.
         * @param previousLayerDepth previous layer depth.
         * @param regulateWeights if true weights are regulated.
         */
        DSConvolutionWeightSet(Initialization initialization, int filterRowSize, int filterColumnSize, int numberOfFilters, int previousLayerDepth, boolean regulateWeights) {
            this.filterRowSize = filterRowSize;
            this.filterColumnSize = filterColumnSize;
            this.numberOfFilters = numberOfFilters;

            for (int channelIndex = 0; channelIndex < previousLayerDepth; channelIndex++) {
                Matrix filterWeight = new DMatrix(filterRowSize, filterColumnSize, initialization, filterRowSize * filterColumnSize, filterRowSize * filterColumnSize);
                filterWeight.setName("WfD" + channelIndex);
                filterWeightsDepthWise.put(channelIndex, filterWeight);
                weights.add(filterWeight);
                registerWeight(filterWeight, regulateWeights, true);

                Matrix filterBias = new DMatrix(0);
                filterBias.setName("BfD" + channelIndex);
                filterBiasesDepthWise.put(channelIndex, filterBias);
                weights.add(filterBias);
                registerWeight(filterBias, false, false);
            }

            for (int filterIndex = 0; filterIndex < numberOfFilters; filterIndex++) {
                HashMap<Integer, Matrix> filterWeightsPointWisePerChannel = new HashMap<>();
                filterWeightsPointWise.put(filterIndex, filterWeightsPointWisePerChannel);
                for (int channelIndex = 0; channelIndex < previousLayerDepth; channelIndex++) {
                    Matrix filterWeight = new DMatrix(1, 1, initialization, 1, numberOfFilters);
                    filterWeight.setName("WfP" + filterIndex);
                    filterWeightsPointWisePerChannel.put(channelIndex, filterWeight);
                    weights.add(filterWeight);
                    registerWeight(filterWeight, regulateWeights, true);
                }

                Matrix filterBias = new DMatrix(0);
                filterBias.setName("BfP" + filterIndex);
                filterBiasesPointWise.put(filterIndex, filterBias);
                weights.add(filterBias);
                registerWeight(filterBias, false, false);

            }
        }

        /**
         * Returns set of weights.
         *
         * @return set of weights.
         */
        public HashSet<Matrix> getWeights() {
            return weights;
        }

        /**
         * Reinitializes weights.
         *
         */
        public void reinitialize() {
            for (Matrix weight : filterWeightsDepthWise.values()) weight.initialize(initialization, filterRowSize * filterColumnSize, numberOfFilters * filterRowSize);
            for (Matrix bias : filterBiasesDepthWise.values()) bias.reset();
            for (HashMap<Integer, Matrix> filterWeightsPointWisePerChannel : filterWeightsPointWise.values()) {
                for (Matrix weight : filterWeightsPointWisePerChannel.values()) weight.initialize(initialization, 1, numberOfFilters);
            }
            for (Matrix bias : filterBiasesPointWise.values()) bias.reset();
        }

        /**
         * Returns number of parameters.
         *
         * @return number of parameters.
         */
        public int getNumberOfParameters() {
            int numberOfParameters = 0;
            for (Matrix weight : weights) numberOfParameters += weight.size();
            return numberOfParameters;
        }

    }

    /**
     * Weight set.
     *
     */
    protected DSConvolutionWeightSet weightSet;

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
     * Activation function for convolutional layer.
     *
     */
    protected ActivationFunction activationFunction;

    /**
     * Input matrices for procedure construction.
     *
     */
    private TreeMap<Integer, MMatrix> inputs;

    /**
     * Constructor for abstract depth-wise separable convolutional layer.
     *
     * @param layerIndex layer index
     * @param activationFunction activation function used.
     * @param initialization initialization function for weight maps.
     * @param params parameters for abstract depth-wise separable convolutional layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     */
    public AbstractDSConvolutionalLayer(int layerIndex, ActivationFunction activationFunction, Initialization initialization, String params) throws DynamicParamException, NeuralNetworkException {
        super (layerIndex, initialization, params);
        this.activationFunction = activationFunction;
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
     * Returns parameters used for abstract depth-wise separable convolutional layer.
     *
     * @return parameters used for abstract depth-wise separable convolutional layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + AbstractDSConvolutionalLayer.paramNameTypes;
    }

    /**
     * Sets parameters used for abstract depth-wise separable convolutional layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - filters: number of filters.<br>
     *     - regulateWeights: true if filter weights are regulated otherwise false (default false).<br>
     *
     * @param params parameters used for abstract depth-wise separable convolutional layer.
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
     * Checks if layer works with recurrent layers.
     *
     * @return if true layer works with recurrent layers otherwise false.
     */
    public boolean worksWithRecurrentLayer() {
        return true;
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
    }

    /**
     * Returns weight set.
     *
     * @return weight set.
     */
    protected WeightSet getWeightSet() {
        return weightSet;
    }

    /**
     * Initializes neural network layer weights.
     *
     */
    public void initializeWeights() {
        weightSet = new DSConvolutionWeightSet(initialization, filterRowSize, filterColumnSize, numberOfFilters, previousLayerDepth, regulateWeights);
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
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets previous input.
     * @return input matrices for procedure construction.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public TreeMap<Integer, MMatrix> getInputMatrices(boolean resetPreviousInput) throws MatrixException {
        inputs = new TreeMap<>();
        for (int index = 0; index < previousLayerDepth; index++) {
            Matrix input = new DMatrix(previousLayerWidth, previousLayerHeight);
            input.setName("Input" + index);
            inputs.put(index, new MMatrix(input));
        }
        return inputs;
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public MMatrix getForwardProcedure() throws MatrixException {
        // Depth-wise separable convolution
        HashMap<Integer, Matrix> dwOutputs = new HashMap<>();
        for (int channelIndex = 0; channelIndex < previousLayerDepth; channelIndex++) {
            Matrix input = inputs.get(channelIndex).get(0);
            input.setStride(stride);
            input.setDilation(dilation);
            input.setFilterRowSize(filterRowSize);
            input.setFilterColumnSize(filterColumnSize);
            Matrix Wf = weightSet.filterWeightsDepthWise.get(channelIndex);
            Matrix dwOutput = executeConvolutionalOperation(input, Wf, weightSet.filterBiasesDepthWise.get(channelIndex));
            dwOutputs.put(channelIndex, dwOutput);
            dwOutput.setName("DWOutput" + channelIndex);
        }

        // Point-wise separable convolution
        MMatrix outputs = new MMatrix(numberOfFilters, "Outputs");
        for (int filterIndex = 0; filterIndex < numberOfFilters; filterIndex++) {
            HashMap<Integer, Matrix> filterWeightsPointWisePerChannel = weightSet.filterWeightsPointWise.get(filterIndex);
            Matrix output = weightSet.filterBiasesPointWise.get(filterIndex);
            for (int channelIndex = 0; channelIndex < previousLayerDepth; channelIndex++) {
                Matrix Wf = filterWeightsPointWisePerChannel.get(channelIndex);
                Matrix dwInput = dwOutputs.get(channelIndex);
                dwInput.setStride(1);
                dwInput.setDilation(1);
                dwInput.setFilterRowSize(1);
                dwInput.setFilterColumnSize(1);
                output = executeConvolutionalOperation(dwInput, Wf, output);
            }
            if (activationFunction != null) output = output.apply(activationFunction);
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
        layerDetailsByName += "Convolution type: " + getConvolutionType() + ", ";
        layerDetailsByName += "Number of filters: " + numberOfFilters + ", ";
        layerDetailsByName += "Filter row size: " + filterRowSize + ", ";
        layerDetailsByName += "Filter column size: " + filterColumnSize + ", ";
        layerDetailsByName += "Stride: " + stride + ", ";
        layerDetailsByName += "Dilation: " + dilation + ", ";
        if (activationFunction != null) layerDetailsByName += "Activation function: " + activationFunction.getName();
        return layerDetailsByName;
    }

    /**
     * Returns convolution type.
     *
     * @return convolution type.
     */
    protected abstract String getConvolutionType();

}
