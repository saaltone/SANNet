/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer.convolutional;

import core.layer.NeuralNetworkLayer;
import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashSet;
import java.util.Map;
import java.util.TreeMap;

/**
 * Implements abstract convolutional layer which implements common functionality for convolutional layer.
 *
 */
public abstract class AbstractSingleConvolutionalLayer extends AbstractConvolutionLayer {


    /**
     * Parameter name types for single convolution layer.
     *     - filterSize size of filter. Default value 3.<br>
     *     - filterRowSize size of filter in terms of rows. Overrides filterSize parameter. Default value 3.<br>
     *     - filterColumnSize size of filter in terms of columns. Overrides filterSize parameter. Default value 3.<br>
     *     - stride: size of stride. Default size 1.<br>
     *     - dilation: dilation step for filter. Default step 1.<br>
     *     - regulateWeights: true if filter weights are regulated otherwise false (default false).<br>
     *
     */
    private final static String paramNameTypes = "(filterSize:INT), " +
            "(filterRowSize:INT), " +
            "(filterColumnSize:INT), " +
            "(stride:INT), " +
            "(dilation:INT), " +
            "(regulateWeights:BOOLEAN)";

    /**
     * Implements weight set for layer.
     *
     */
    protected class ConvolutionWeightSet implements WeightSet, Serializable {

        @Serial
        private static final long serialVersionUID = 3726198810615147311L;

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
         * @param initialization   weight initialization function.
         * @param filterRowSize    filter row size.
         * @param filterColumnSize filter column size.
         * @param regulateWeights  if true weights are regulated.
         */
        ConvolutionWeightSet(Initialization initialization, int filterRowSize, int filterColumnSize, boolean regulateWeights) {
            this.filterRowSize = filterRowSize;
            this.filterColumnSize = filterColumnSize;

            filterWeight = new DMatrix(filterRowSize, filterColumnSize, previousLayerDepth, initialization, filterRowSize * filterColumnSize, filterRowSize * filterColumnSize);
            filterWeight.setName("Wf");
            weights.add(filterWeight);
            registerWeight(filterWeight, regulateWeights, true);

            filterBias = new DMatrix(getLayerWidth(), getLayerHeight(), 1);
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
            filterWeight.initialize(initialization, filterRowSize * filterColumnSize, filterRowSize * filterColumnSize);
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
     * True is filter weights are regulated otherwise weights are not regulated.
     *
     */
    private boolean regulateWeights;

    /**
     * Input matrices for procedure construction.
     *
     */
    private TreeMap<Integer, Matrix> inputs;

    /**
     * Constructor for abstract single convolutional layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight maps.
     * @param params parameters for abstract single convolutional layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     */
    public AbstractSingleConvolutionalLayer(int layerIndex, Initialization initialization, String params) throws DynamicParamException, NeuralNetworkException {
        super (layerIndex, initialization, params);
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
     * Returns parameters used for abstract convolutional layer.
     *
     * @return parameters used for abstract convolutional layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + AbstractSingleConvolutionalLayer.paramNameTypes;
    }

    /**
     * Sets parameters used for abstract convolutional layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - regulateWeights: true if filter weights are regulated otherwise false (default false).<br>
     *
     * @param params parameters used for abstract convolutional layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
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
     * Checks if layer can have multiple previous layers.
     *
     * @return  if true layer can have multiple previous layers otherwise false.
     */
    public boolean canHaveMultiplePreviousLayers() {
        return true;
    }

    /**
     * Initializes neural network layer dimensions.
     *
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     */
    public void initializeDimensions() throws NeuralNetworkException {
        super.initializeDimensions();

        previousLayerDepth = 1;

        setLayerDepth(1);
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
        weightSet = new ConvolutionWeightSet(initialization, filterRowSize, filterColumnSize, regulateWeights);
    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets previous input.
     * @return input matrices for procedure construction.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public TreeMap<Integer, Matrix> getInputMatrices(boolean resetPreviousInput) throws MatrixException {
        inputs = new TreeMap<>();

        TreeMap<Integer, Matrix> inputMatrices = new TreeMap<>();
        int layerWidth = -1;
        int layerHeight = -1;
        for (Map.Entry<Integer, NeuralNetworkLayer> entry : getPreviousLayers().entrySet()) {
            if (layerWidth == -1) layerWidth = entry.getValue().getLayerWidth();
            else if (layerWidth != entry.getValue().getLayerWidth()) throw new MatrixException("All inputs must have same width.");
            if (layerHeight == -1) layerHeight = entry.getValue().getLayerHeight();
            else if (layerHeight != entry.getValue().getLayerHeight()) throw new MatrixException("All inputs must have same height.");
            Matrix input = new DMatrix(layerWidth, layerHeight, 1, Initialization.ONE);
            input.setName("Input" + entry.getValue().getLayerIndex() + "{" + entry.getKey() + "}");
            inputMatrices.put(entry.getKey(), input);
        }

        for (int inputIndex = 0; inputIndex < inputMatrices.size(); inputIndex++) {
            inputs.put(inputIndex, inputMatrices.get(inputIndex));
        }

        return inputs;
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix getForwardProcedure() throws MatrixException {
        Matrix output = weightSet.filterBias;
        for (Matrix matrix : inputs.values()) {
            matrix.setStride(stride);
            matrix.setDilation(dilation);
            matrix.setFilterRowSize(filterRowSize);
            matrix.setFilterColumnSize(filterColumnSize);
            matrix.setFilterDepth(1);
            output = output.add(executeConvolutionalOperation(matrix, weightSet.filterWeight));
        }
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
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        String layerDetailsByName = super.getLayerDetailsByName() + ", ";
        layerDetailsByName += "Convolution type: " + getConvolutionType();
        return layerDetailsByName;
    }

    /**
     * Returns convolution type.
     *
     * @return convolution type.
     */
    protected abstract String getConvolutionType();

}
