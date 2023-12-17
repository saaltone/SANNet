/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
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
 * Implements abstract class that provides common functionalities for normalization.
 *
 */
public abstract class AbstractNormalization extends AbstractExecutionLayer {

    /**
     * Parameter name types for abstract normalization.
     *     - meanOnly: true if normalization is done only by using mean otherwise false (default value false).<br>
     *
     */
    private final static String paramNameTypes = "(meanOnly:BOOLEAN)";

    /**
     * Implements weight set for layer.
     *
     */
    protected class AbstractNormalizationWeightSet implements WeightSet, Serializable {

        @Serial
        private static final long serialVersionUID = 5485247921810092695L;

        /**
         * Learnable parameter gamma of abstract normalization layer.
         *
         */
        private final Matrix gamma;

        /**
         * Learnable parameter beta of abstract normalization layer.
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
         *
         * @param previousLayerWidth  width of previous layer.
         * @param previousLayerHeight height of previous layer.
         * @param previousLayerDepth depth of previous layer.
         */
        AbstractNormalizationWeightSet(int previousLayerWidth, int previousLayerHeight, int previousLayerDepth) {
            gamma = new DMatrix(previousLayerWidth, previousLayerHeight, previousLayerDepth, (row, col) -> new Random().nextGaussian() * 0.1);
            gamma.setName("Gamma");
            beta = new DMatrix(previousLayerWidth, previousLayerHeight, previousLayerDepth);
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
            beta.initialize((row, col) -> new Random().nextGaussian() * 0.1);
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
    protected AbstractNormalizationWeightSet weightSet;

    /**
     * True if abstract normalization is used calculation only with mean and variance excluded.
     *
     */
    private boolean meanOnly;

    /**
     * Input matrix for procedure construction.
     *
     */
    private Matrix input;

    /**
     * Matrix for epsilon value.
     *
     */
    private Matrix epsilonMatrix;

    /**
     * Square root function.
     *
     */
    private final UnaryFunction sqrtFunction = new UnaryFunction(UnaryFunctionType.SQRT);

    /**
     * Inverse function.
     *
     */
    private final UnaryFunction invFunction = new UnaryFunction(UnaryFunctionType.INV);

    /**
     * If value is one applies operation over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     *
     */
    private final int direction;

    /**
     * Constructor for abstract normalization layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for abstract normalization layer.
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     * @throws MatrixException        throws exception if matrix operation fails.
     */
    public AbstractNormalization(int layerIndex, Initialization initialization, String params, int direction) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, params);
        this.direction = direction;
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        epsilonMatrix = new DMatrix(10E-8);
        epsilonMatrix.setName("Epsilon");
        registerConstantMatrix(epsilonMatrix);
        registerStopGradient(epsilonMatrix);
        meanOnly = false;
    }

    /**
     * Returns parameters used for abstract normalization layer.
     *
     * @return parameters used for abstract normalization layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + AbstractNormalization.paramNameTypes;
    }

    /**
     * Sets parameters used for abstract normalization.<br>
     * <br>
     * Supported parameters are:<br>
     *     - meanOnly: true if normalization is done only by using mean otherwise false (default value false).<br>
     *
     * @param params parameters used for abstract normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("meanOnly")) meanOnly = params.getValueAsBoolean("meanOnly");
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
        weightSet = new AbstractNormalizationWeightSet(getDefaultPreviousLayer().getLayerWidth(), getDefaultPreviousLayer().getLayerHeight(), getDefaultPreviousLayer().getLayerDepth());
    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     */
    public TreeMap<Integer, Matrix> getInputMatrices(boolean resetPreviousInput) {
        input = new DMatrix(getDefaultPreviousLayer().getLayerWidth(), getDefaultPreviousLayer().getLayerHeight(), getDefaultPreviousLayer().getLayerDepth(), Initialization.ONE);
        input.setName("Input" + getDefaultPreviousLayer().getLayerIndex());
        return new TreeMap<>() {{ put(0, input); }};
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException       throws exception if matrix operation fails.
     */
    public Matrix getForwardProcedure() throws MatrixException {
        Matrix output = input.subtract(input.meanAsMatrix(direction));
        if (!meanOnly) output = output.multiply(input.varianceAsMatrix(direction).add(epsilonMatrix).apply(sqrtFunction).apply(invFunction));
        output = output.multiply(weightSet.gamma).add(weightSet.beta);
        output.setName("Output");

        return output;
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
