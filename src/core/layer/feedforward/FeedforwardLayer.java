/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer.feedforward;

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
 * Implements non-recurrent feedforward layer.
 *
 */
public class FeedforwardLayer extends AbstractExecutionLayer {

    /**
     * Parameter name types for feedforward layer.
     *     - regulateDirectWeights: true if (direct) weights are regulated otherwise false. Default value true.<br>
     *
     */
    private final static String paramNameTypes = "(regulateDirectWeights:BOOLEAN)";

    /**
     * Implements weight set for layer.
     *
     */
    protected class FeedforwardWeightSet implements WeightSet, Serializable {

        @Serial
        private static final long serialVersionUID = 7859757124635021579L;

        /**
         * Weight matrix.
         *
         */
        private final Matrix weight;

        /**
         * Bias matrix.
         *
         */
        private final Matrix bias;

        /**
         * Set of weights.
         *
         */
        private final HashSet<Matrix> weights = new HashSet<>();

        /**
         * Constructor for weight set
         *
         * @param initialization        weight initialization function.
         * @param previousLayerWidth    width of previous layer.
         * @param layerWidth            width of current layer.
         * @param previousLayerDepth    depth of previous layer.
         * @param regulateDirectWeights if true direct weights are regulated.
         */
        FeedforwardWeightSet(Initialization initialization, int previousLayerWidth, int layerWidth, int previousLayerDepth, boolean regulateDirectWeights) {
            weight = new DMatrix(layerWidth, previousLayerWidth, previousLayerDepth, initialization);
            weight.setName("Weight");
            bias = new DMatrix(layerWidth, 1, previousLayerDepth);
            bias.setName("Bias");

            weights.add(weight);
            weights.add(bias);

            registerWeight(weight, regulateDirectWeights, true);
            registerWeight(bias, false, false);
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
            weight.initialize(initialization);
            bias.reset();
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
    protected FeedforwardWeightSet weightSet;

    /**
     * Activation function for feedforward layer.
     *
     */
    protected final ActivationFunction activationFunction;

    /**
     * True if weights are regulated otherwise weights are not regulated.
     *
     */
    private boolean regulateDirectWeights;

    /**
     * Input matrix for procedure construction.
     *
     */
    private Matrix input;

    /**
     * Constructor for feedforward layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for feedforward layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public FeedforwardLayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException {
        super (layerIndex, initialization, params);
        this.activationFunction = null;
    }

    /**
     * Constructor for feedforward layer.
     *
     * @param layerIndex layer index
     * @param activationFunction activation function for layer.
     * @param initialization initialization function for weight.
     * @param params parameters for feedforward layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this
     */
    public FeedforwardLayer(int layerIndex, ActivationFunction activationFunction, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, params);
        this.activationFunction = activationFunction != null ? activationFunction : new ActivationFunction(UnaryFunctionType.RELU);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        regulateDirectWeights = true;
    }

    /**
     * Returns parameters used for feedforward layer.
     *
     * @return parameters used for feedforward layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + FeedforwardLayer.paramNameTypes;
    }

    /**
     * Sets parameters used for feedforward layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - regulateDirectWeights: true if (direct) weights are regulated otherwise false. Default value true.<br>
     *
     * @param params parameters used for feedforward layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("regulateDirectWeights")) regulateDirectWeights = params.getValueAsBoolean("regulateDirectWeights");
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
        weightSet = new FeedforwardWeightSet(initialization, getDefaultPreviousLayer().getLayerWidth(), getLayerWidth(), getDefaultPreviousLayer().getLayerDepth(), regulateDirectWeights);
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
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix getForwardProcedure() throws MatrixException {
        Matrix output = weightSet.weight.dot(input);
        output = output.add(weightSet.bias);

        if (activationFunction != null) output = output.apply(activationFunction);

        output.setName("Output");
        return output;
    }

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
        return (activationFunction == null ? "" : "Activation function: " + activationFunction.getName());
    }

}
