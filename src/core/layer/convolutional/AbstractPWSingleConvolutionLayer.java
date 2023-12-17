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
import utils.matrix.DMatrix;
import utils.matrix.Initialization;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashSet;
import java.util.Map;
import java.util.TreeMap;

/**
 * Implements abstract single point-wise separable convolutional layer which implements common functionality for convolutional layer.<br>
 * Reference: <a href="https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728">...</a>
 *
 */
public abstract class AbstractPWSingleConvolutionLayer extends AbstractConvolutionLayer {

    /**
     * Parameter name types for abstract single point-wise separable convolutional layer.
     *     - regulateWeights: true if filter weights are regulated otherwise false (default false).<br>
     *
     */
    private final static String paramNameTypes = "(regulateWeights:BOOLEAN)";

    /**
     * Implements weight set for layer.
     *
     */
    protected class PWConvolutionWeightSet implements WeightSet, Serializable {

        @Serial
        private static final long serialVersionUID = -7133810873264735302L;

        /**
         * Treemap for point-wise filter maps (weights).
         *
         */
        private final TreeMap<Integer, Matrix> filtersWeightPointWise = new TreeMap<>();

        /**
         * Point-wise bias.
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
         * @param regulateWeights if true weights are regulated.
         */
        PWConvolutionWeightSet(Initialization initialization, boolean regulateWeights) {
            for (Integer filterIndex : getPreviousLayers().keySet()) {
                Matrix filterWeightPointWise = new DMatrix(1, 1, 1, initialization, 1, 1);
                filterWeightPointWise.setName("WfPW" + filterIndex);
                weights.add(filterWeightPointWise);
                registerWeight(filterWeightPointWise, regulateWeights, true);
                filtersWeightPointWise.put(filterIndex, filterWeightPointWise);
            }

            filterBiasPointWise = new DMatrix(getLayerWidth(), getLayerHeight(), getLayerDepth());
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
            for (Matrix filterWeightPointWise : filtersWeightPointWise.values()) filterWeightPointWise.initialize(initialization, 1, 1);
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
    protected PWConvolutionWeightSet weightSet;

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
     * Constructor for abstract single point-wise separable convolutional layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight maps.
     * @param params parameters for abstract single point-wise separable convolutional layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     */
    public AbstractPWSingleConvolutionLayer(int layerIndex, Initialization initialization, String params) throws DynamicParamException, NeuralNetworkException {
        super (layerIndex, initialization, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        filterRowSize = 1;
        filterColumnSize = 1;
        stride = 1;
        dilation = 1;
        regulateWeights = false;
    }

    /**
     * Returns parameters used for abstract single point-wise separable convolutional layer.
     *
     * @return parameters used for abstract single point-wise separable convolutional layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + AbstractPWSingleConvolutionLayer.paramNameTypes;
    }

    /**
     * Sets parameters used for abstract single point-wise separable convolutional layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - regulateWeights: true if filter weights are regulated otherwise false (default false).<br>
     *
     * @param params parameters used for abstract single point-wise separable convolutional layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
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
        weightSet = new PWConvolutionWeightSet(initialization, regulateWeights);
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
        // Point-wise separable convolution
        Matrix output = weightSet.filterBiasPointWise;
        for (Map.Entry<Integer, Matrix> entry : inputs.entrySet()) {
            Matrix matrix = entry.getValue();
            matrix.setStride(stride);
            matrix.setDilation(dilation);
            matrix.setFilterRowSize(filterRowSize);
            matrix.setFilterColumnSize(filterColumnSize);
            matrix.setFilterDepth(1);
            output = output.add(executeConvolutionalOperation(matrix, weightSet.filtersWeightPointWise.get(entry.getKey())));
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
