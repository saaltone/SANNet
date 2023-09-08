/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer.convolutional;

import core.activation.ActivationFunction;
import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.Initialization;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashSet;
import java.util.TreeMap;

/**
 * Implements abstract convolutional layer which implements common functionality for convolutional layer.
 *
 */
public abstract class AbstractConvolutionalLayer extends AbstractConvolutionLayer {

    /**
     * Parameter name types for abstract convolutional layer.
     *     - filters: number of filters (default 1).<br>
     *     - isDepthSeparable: true if layer is depth separable (default false).<br>
     *     - regulateWeights: true if filter weights are regulated otherwise false (default false).<br>
     *
     */
    private final static String paramNameTypes = "(filters:INT), " +
            "(isDepthSeparable:BOOLEAN), "  +
            "(regulateWeights:BOOLEAN)";

    /**
     * Implements weight set for layer.
     *
     */
    protected class ConvolutionWeightSet implements WeightSet, Serializable {

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
         * Previous layer depth.
         *
         */
        private final int previousLayerDepth;

        /**
         * Treemap for filter maps (weights).
         *
         */
        private final Matrix filterWeight;

        /**
         * Treemap for biases.
         *
         */
        private final Matrix filterBias;

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
         * @param isDepthSeparable true if layer is depth separable otherwise false
         * @param regulateWeights if true weights are regulated.
         */
        ConvolutionWeightSet(Initialization initialization, int filterRowSize, int filterColumnSize, int numberOfFilters, int previousLayerDepth, boolean isDepthSeparable, boolean regulateWeights) {
            this.filterRowSize = filterRowSize;
            this.filterColumnSize = filterColumnSize;
            this.numberOfFilters = numberOfFilters;
            this.previousLayerDepth = previousLayerDepth;

            filterWeight = new DMatrix(filterRowSize, filterColumnSize, isDepthSeparable ? previousLayerDepth : previousLayerDepth * numberOfFilters, initialization, filterRowSize * filterColumnSize * previousLayerDepth,  filterRowSize * filterColumnSize * numberOfFilters);
            filterWeight.setName("Wf");
            weights.add(filterWeight);
            registerWeight(filterWeight, regulateWeights, true);

            filterBias = new DMatrix(getLayerWidth(), getLayerHeight(), numberOfFilters);
            filterBias.setName("Bf");
            weights.add(filterBias);
            registerWeight(filterBias, false, false);
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
            filterWeight.initialize(initialization, filterRowSize * filterColumnSize * previousLayerDepth, filterRowSize * filterColumnSize * numberOfFilters);
            filterBias.reset();
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
    protected ConvolutionWeightSet weightSet;

    /**
     * Defines number of filters.
     *
     */
    private int numberOfFilters;

    /**
     * Defines if layer is depth separable.
     *
     */
    private boolean isDepthSeparable;

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
     * Constructor for abstract convolutional layer.
     *
     * @param layerIndex layer index
     * @param activationFunction activation function used.
     * @param initialization initialization function for weight maps.
     * @param params parameters for abstract convolutional layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     */
    public AbstractConvolutionalLayer(int layerIndex, ActivationFunction activationFunction, Initialization initialization, String params) throws DynamicParamException, NeuralNetworkException {
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
        numberOfFilters = 1;
        stride = 1;
        dilation = 1;
        isDepthSeparable = false;
        regulateWeights = false;
    }

    /**
     * Returns parameters used for abstract convolutional layer.
     *
     * @return parameters used for abstract convolutional layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + AbstractConvolutionalLayer.paramNameTypes;
    }

    /**
     * Sets parameters used for abstract convolutional layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - filters: number of filters (default 1).<br>
     *     - isDepthSeparable: true if layer is depth separable (default false).<br>
     *     - regulateWeights: true if filter weights are regulated otherwise false (default false).<br>
     *
     * @param params parameters used for abstract convolutional layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("filters")) {
            numberOfFilters = params.getValueAsInteger("filters");
            if (numberOfFilters < 1) throw new NeuralNetworkException("At least one filter must be defined. Number of filters cannot be: " + numberOfFilters);
        }
        if (params.hasParam("isDepthSeparable")) isDepthSeparable = params.getValueAsBoolean("isDepthSeparable");
        if (params.hasParam("regulateWeights")) regulateWeights = params.getValueAsBoolean("regulateWeights");
    }

    /**
     * Initializes neural network layer dimensions.
     *
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     */
    public void initializeDimensions() throws NeuralNetworkException {
        super.initializeDimensions();

        if (isDepthSeparable && numberOfFilters != previousLayerDepth) throw new NeuralNetworkException("For depth separable layer depth of previous layer " + previousLayerDepth + " and number of filters " + numberOfFilters +  " must be same");
        if (numberOfFilters < 1) throw new NeuralNetworkException("At least one filter must be defined");

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
        weightSet = new ConvolutionWeightSet(initialization, filterRowSize, filterColumnSize, numberOfFilters, previousLayerDepth, isDepthSeparable, regulateWeights);
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
        Matrix input = inputs.get(0);
        input.setFilterRowSize(filterRowSize);
        input.setFilterColumnSize(filterColumnSize);
        input.setFilterDepth(numberOfFilters);
        input.setStride(stride);
        input.setDilation(dilation);
        input.setIsDepthSeparable(isDepthSeparable);
        Matrix output = weightSet.filterBias.add(executeConvolutionalOperation(input, weightSet.filterWeight));
        if (activationFunction != null) output = output.apply(activationFunction);
        output.setName("Output");
        return output;
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
        String layerDetailsByName = super.getLayerDetailsByName() + ", ";
        layerDetailsByName += "Number of filters: " + numberOfFilters + ", ";
        layerDetailsByName += "Is depth separable: " + isDepthSeparable + ", ";
        layerDetailsByName += "Convolution type: " + getConvolutionType();
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
