/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.layer.normalization;

import core.layer.AbstractExecutionLayer;
import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashSet;
import java.util.Random;
import java.util.TreeMap;

/**
 * Implements layer for layer normalization.
 *
 */
public class LayerNormalization extends AbstractExecutionLayer {

    /**
     * Parameter name types for layer normalization.
     *     - meanOnly: true if normalization is done only by using mean otherwise false (default value false).<br>
     *
     */
    private final static String paramNameTypes = "(meanOnly:BOOLEAN)";

    /**
     * Implements weight set for layer.
     *
     */
    protected class LayerNormalizationWeightSet implements WeightSet, Serializable {

        @Serial
        private static final long serialVersionUID = 5485247921810092695L;

        /**
         * Learnable parameter gamma of layer normalization layer.
         *
         */
        private final Matrix gamma;

        /**
         * Learnable parameter beta of layer normalization layer.
         *
         */
        private final Matrix beta;

        /**
         * Set of weights.
         *
         */
        private final HashSet<Matrix> weights = new HashSet<>();

        /**
         * Constructor for weight set
         *  @param previousLayerWidth width of previous layer.
         * @param previousLayerHeight height of previous layer.
         */
        LayerNormalizationWeightSet(int previousLayerWidth, int previousLayerHeight) {
            gamma = new DMatrix(previousLayerWidth, previousLayerHeight, (row, col) -> new Random().nextGaussian() * 0.1);
            gamma.setName("Gamma");
            beta = new DMatrix(previousLayerWidth, previousLayerHeight);
            beta.setName("Beta");

            weights.add(gamma);
            weights.add(beta);

            registerWeight(gamma, false, false);
            registerWeight(beta, false, false);
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
            gamma.initialize((row, col) -> new Random().nextGaussian() * 0.1);
            beta.reset();
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
    protected LayerNormalizationWeightSet weightSet;

    /**
     * True if layer normalization is used calculation only with mean and variance excluded.
     *
     */
    private boolean meanOnly;

    /**
     * Input matrix for procedure construction.
     *
     */
    private Matrix input;

    /**
     * Constructor for layer normalization layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for layer normalization layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public LayerNormalization(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException {
        super (layerIndex, initialization, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        meanOnly = false;
    }

    /**
     * Returns parameters used for layer normalization layer.
     *
     * @return parameters used for layer normalization layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + LayerNormalization.paramNameTypes;
    }

    /**
     * Sets parameters used for layer normalization.<br>
     * <br>
     * Supported parameters are:<br>
     *     - meanOnly: true if normalization is done only by using mean otherwise false (default value false).<br>
     *
     * @param params parameters used for layer normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("meanOnly")) meanOnly = params.getValueAsBoolean("meanOnly");
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
        weightSet = new LayerNormalizationWeightSet(getDefaultPreviousLayer().getLayerWidth(), getDefaultPreviousLayer().getLayerHeight());
    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return input matrix for procedure construction.
     */
    public TreeMap<Integer, MMatrix> getInputMatrices(boolean resetPreviousInput) throws MatrixException {
        input = new DMatrix(getDefaultPreviousLayer().getLayerWidth(), getDefaultPreviousLayer().getLayerHeight(), Initialization.ONE);
        input.setName("Input" + getDefaultPreviousLayer().getLayerIndex());
        return new TreeMap<>() {{ put(0, new MMatrix(input)); }};
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public MMatrix getForwardProcedure() throws MatrixException, DynamicParamException {
        Matrix output = !meanOnly ? input.subtract(input.meanAsMatrix()).divide(input.varianceAsMatrix().apply(UnaryFunctionType.SQRT)).multiply(weightSet.gamma).add(weightSet.beta) : input.subtract(input.meanAsMatrix()).multiply(weightSet.gamma).add(weightSet.beta);
        output.setName("Output");

        MMatrix outputs = new MMatrix(1, "Output");
        outputs.put(0, output);
        return outputs;
    }

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
        return "Mean only: " + meanOnly;
    }

}
