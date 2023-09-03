/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
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
import java.util.HashSet;
import java.util.TreeMap;

/**
 * Implements abstract depth-wise separable convolutional layer which implements common functionality for convolutional layer.<br>
 * Reference: <a href="https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728">...</a>
 *
 */
public abstract class AbstractDSConvolutionalLayer extends AbstractExecutionLayer {

    /**
     * Parameter name types for abstract depth-wise separable convolutional layer.
     *     - filters: number of filters.<br>
     *     - filterSize size of filter. Default value 3.<br>
     *     - filterRowSize size of filter in terms of rows. Overrides filterSize parameter. Default value 3.<br>
     *     - filterColumnSize size of filter in terms of columns. Overrides filterSize parameter. Default value 3.<br>
     *     - stride: size of stride. Default size 1.<br>
     *     - dilation: dilation step for filter. Default step 1.<br>
     *     - regulateWeights: true if filter weights are regulated otherwise false (default false).<br>
     *
     */
    private final static String paramNameTypes = "(filters:INT), " +
            "(filterSize:INT), " +
            "(filterRowSize:INT), " +
            "(filterColumnSize:INT), " +
            "(stride:INT), " +
            "(dilation:INT), " +
            "(regulateWeights:BOOLEAN)";

    /**
     * Implements weight set for layer.
     *
     */
    protected class DSConvolutionWeightSet implements WeightSet, Serializable {

        @Serial
        private static final long serialVersionUID = 8669829840637421343L;

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
        private final Matrix filterWeightDepthWise;

        /**
         * Tree map for depth-wise biases.
         *
         */
        private final Matrix filterBiasDepthWise;

        /**
         * Treemap for point-wise filter maps (weights).
         *
         */
        private final Matrix filterWeightPointWise;

        /**
         * Treemap for point-wise biases.
         *
         */
        private final Matrix filterBiasPointWise;

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

            filterWeightDepthWise = new DMatrix(filterRowSize, filterColumnSize, previousLayerDepth, initialization, filterRowSize * filterColumnSize * previousLayerDepth, filterRowSize * filterColumnSize * previousLayerDepth);
            filterWeightDepthWise.setName("WfDW");
            weights.add(filterWeightDepthWise);
            registerWeight(filterWeightDepthWise, regulateWeights, true);

            filterBiasDepthWise = new DMatrix(getLayerWidth(), getLayerHeight(), previousLayerDepth);
            filterBiasDepthWise.setName("BfDW");
            weights.add(filterBiasDepthWise);
            registerWeight(filterBiasDepthWise, false, false);

            filterWeightPointWise = new DMatrix(1, 1, numberOfFilters, initialization, previousLayerDepth, numberOfFilters);
            filterWeightPointWise.setName("WfPW");
            weights.add(filterWeightPointWise);
            registerWeight(filterWeightPointWise, regulateWeights, true);

            filterBiasPointWise = new DMatrix(getLayerWidth(), getLayerHeight(), numberOfFilters);
            filterBiasPointWise.setName("BfPW");
            weights.add(filterBiasPointWise);
            registerWeight(filterBiasPointWise, false, false);
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
            filterWeightDepthWise.initialize(initialization, filterRowSize * filterColumnSize * previousLayerDepth, filterRowSize * filterColumnSize * previousLayerDepth);
            filterBiasDepthWise.reset();
            filterWeightPointWise.initialize(initialization, 1, numberOfFilters);
            filterBiasPointWise.reset();
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
    private TreeMap<Integer, Matrix> inputs;

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
     *     - filterSize size of filter. Default value 3.<br>
     *     - filterRowSize size of filter in terms of rows. Overrides filterSize parameter. Default value 3.<br>
     *     - filterColumnSize size of filter in terms of columns. Overrides filterSize parameter. Default value 3.<br>
     *     - stride: size of stride. Default size 1.<br>
     *     - dilation: dilation step for filter. Default step 1.<br>
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
        if (params.hasParam("filterSize")) {
            int filterSize = params.getValueAsInteger("filterSize");
            if (filterSize < 1) throw new NeuralNetworkException("Filter size must be at least 1.");
            setFilterRowSize(filterSize);
            setFilterColumnSize(filterSize);
        }
        if (params.hasParam("filterRowSize")) {
            int filterRowSize = params.getValueAsInteger("filterRowSize");
            if (filterRowSize < 1) throw new NeuralNetworkException("Filter row size must be at least 1.");
            setFilterRowSize(filterRowSize);
        }
        if (params.hasParam("filterColumnSize")) {
            int filterColumnSize = params.getValueAsInteger("filterColumnSize");
            if (filterColumnSize < 1) throw new NeuralNetworkException("Filter column size must be at least 1.");
            setFilterColumnSize(filterColumnSize);
        }
        if (params.hasParam("stride")) {
            int stride = params.getValueAsInteger("stride");
            if (stride < 1) throw new NeuralNetworkException("Stride must be at least 1.");
            setStride(stride);
        }
        if (params.hasParam("dilation")) {
            int dilation = params.getValueAsInteger("dilation");
            if (dilation < 1) throw new NeuralNetworkException("Dilation must be at least 1.");
            setDilation(dilation);
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
        previousLayerWidth = getDefaultPreviousLayer().getLayerWidth();
        previousLayerHeight = getDefaultPreviousLayer().getLayerHeight();
        previousLayerDepth = getDefaultPreviousLayer().getLayerDepth();
        if (previousLayerWidth < 1) throw new NeuralNetworkException("Default previous layer width must be positive. Invalid value: " + previousLayerWidth);
        if (previousLayerHeight < 1) throw new NeuralNetworkException("Default previous height width must be positive. Invalid value: " + previousLayerHeight);
        if (previousLayerDepth < 1) throw new NeuralNetworkException("Default previous depth width must be positive. Invalid value: " + previousLayerDepth);

        if (getCurrentLayerWidth() < 1) throw new NeuralNetworkException("Convolutional layer width cannot be less than 1: " + getCurrentLayerWidth());
        if (getCurrentLayerHeight() < 1) throw new NeuralNetworkException("Convolutional layer height cannot be less than 1: " + getCurrentLayerHeight());
        if (numberOfFilters < 1) throw new NeuralNetworkException("At least one filter must be defined");

        setLayerWidth(getCurrentLayerWidth());
        setLayerHeight(getCurrentLayerHeight());
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
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets previous input.
     * @return input matrices for procedure construction.
     */
    public TreeMap<Integer, Matrix> getInputMatrices(boolean resetPreviousInput) {
        inputs = new TreeMap<>();
        Matrix input = new DMatrix(previousLayerWidth, previousLayerHeight, previousLayerDepth);
        input.setName("Input" + getDefaultPreviousLayer().getLayerIndex());
        inputs.put(0, input);
        return inputs;
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix getForwardProcedure() throws MatrixException {
        // Depth-wise separable convolution
        Matrix input = inputs.get(0);
        input.setFilterRowSize(filterRowSize);
        input.setFilterColumnSize(filterColumnSize);
        input.setStride(stride);
        input.setDilation(dilation);
        input.setIsDepthSeparable(true);
        Matrix dwOutput = weightSet.filterBiasDepthWise.add(executeConvolutionalOperation(input, weightSet.filterWeightDepthWise));
        dwOutput.setName("DWOutput");

        // Point-wise convolution
        dwOutput.setFilterRowSize(1);
        dwOutput.setFilterColumnSize(1);
        dwOutput.setStride(1);
        dwOutput.setDilation(1);
        dwOutput.setIsDepthSeparable(false);
        Matrix pwOutput = weightSet.filterBiasPointWise.add(executeConvolutionalOperation(dwOutput, weightSet.filterWeightPointWise));

        if (activationFunction != null) pwOutput = pwOutput.apply(activationFunction);

        pwOutput.setName("Output");

        return pwOutput;
    }

    /**
     * Executes convolutional operation.
     *
     * @param input  input matrix.
     * @param filter filter matrix.
     * @return result of convolutional operation.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract Matrix executeConvolutionalOperation(Matrix input, Matrix filter) throws MatrixException;

    /**
     * Returns matrices for which gradient is not calculated.
     *
     * @return matrices for which gradient is not calculated.
     */
    public HashSet<Matrix> getStopGradients() {
        return new HashSet<>();
    }

    /**
     * Returns constant matrices.
     *
     * @return constant matrices.
     */
    public HashSet<Matrix> getConstantMatrices() {
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
        layerDetailsByName += "Dilation: " + dilation;
        if (activationFunction != null) layerDetailsByName += ", Activation function: " + activationFunction.getName();
        return layerDetailsByName;
    }

    /**
     * Returns convolution type.
     *
     * @return convolution type.
     */
    protected abstract String getConvolutionType();

}
