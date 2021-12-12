/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.layer.recurrent;

import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import core.activation.ActivationFunction;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashSet;

/**
 * Implements basic simple recurrent layer.<br>
 * This layer is by nature prone to numerical instability with limited temporal memory.<br>
 *
 */
public class RecurrentLayer extends AbstractRecurrentLayer {

    /**
     * Parameter name types for recurrent layer.
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value true).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value false).<br>
     *
     */
    private final static String paramNameTypes = "(regulateDirectWeights:BOOLEAN), " +
            "(regulateRecurrentWeights:BOOLEAN)";

    /**
     * Class that defines weight set for layer.
     *
     */
    protected class RecurrentWeightSet implements WeightSet, Serializable {

        @Serial
        private static final long serialVersionUID = -6354547644237149706L;

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
         * Weight matrix for recurrent input.
         *
         */
        private final Matrix recurrentWeight;

        /**
         * Set of weights.
         *
         */
        private final HashSet<Matrix> weights = new HashSet<>();

        /**
         * Constructor for weight set
         *
         * @param initialization weight initialization function.
         * @param previousLayerWidth width of previous layer.
         * @param layerWidth width of current layer.
         * @param regulateDirectWeights if true direct weights are regulated.
         * @param regulateRecurrentWeights if true recurrent weight are regulated.
         */
        RecurrentWeightSet(Initialization initialization, int previousLayerWidth, int layerWidth, boolean regulateDirectWeights, boolean regulateRecurrentWeights) {
            weight = new DMatrix(layerWidth, previousLayerWidth, initialization, "Weight");
            recurrentWeight = new DMatrix(layerWidth, layerWidth, initialization, "RecurrentWeight");
            bias = new DMatrix(layerWidth, 1, "bias");

            weights.add(weight);
            weights.add(recurrentWeight);
            weights.add(bias);

            registerWeight(weight, regulateDirectWeights, true);
            registerWeight(recurrentWeight, regulateRecurrentWeights, true);
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
            recurrentWeight.initialize(initialization);
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
    protected RecurrentWeightSet weightSet;

    /**
     * Current weight set.
     *
     */
    protected RecurrentWeightSet currentWeightSet;

    /**
     * Activation function for neural network layer.
     *
     */
    protected final ActivationFunction activationFunction;

    /**
     * Matrix to store previous output.
     *
     */
    private Matrix previousOutput;

    /**
     * If true regulates direct feedforward weights (W).
     *
     */
    private boolean regulateDirectWeights;

    /**
     * If true regulates recurrent input weights (Wl).
     *
     */
    private boolean regulateRecurrentWeights;

    /**
     * Input matrix for procedure construction.
     *
     */
    private Matrix input;

    /**
     * Constructor for recurrent layer.
     *
     * @param layerIndex layer Index.
     * @param activationFunction activation function used.
     * @param initialization initialization function for weight.
     * @param params parameters for recurrent layer.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public RecurrentLayer(int layerIndex, ActivationFunction activationFunction, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
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
        regulateRecurrentWeights = false;
    }

    /**
     * Returns parameters used for recurrent layer.
     *
     * @return return parameters used for recurrent layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + RecurrentLayer.paramNameTypes;
    }

    /**
     * Sets parameters used for recurrent layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value true).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value false).<br>
     *
     * @param params parameters used for recurrent layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("regulateDirectWeights")) regulateDirectWeights = params.getValueAsBoolean("regulateDirectWeights");
        if (params.hasParam("regulateRecurrentWeights")) regulateRecurrentWeights = params.getValueAsBoolean("regulateRecurrentWeights");
    }

    /**
     * Returns if direct weights are regulated.
     *
     * @return true if direct weights are regulated otherwise false.
     */
    protected boolean getRegulateDirectWeights() {
        return regulateDirectWeights;
    }

    /**
     * Returns if recurrent weights are regulated.
     *
     * @return true if recurrent weights are regulated otherwise false.
     */
    protected boolean getRegulateRecurrentWeights() {
        return regulateRecurrentWeights;
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
        weightSet = new RecurrentWeightSet(initialization, getPreviousLayerWidth(), super.getLayerWidth(), regulateDirectWeights, regulateRecurrentWeights);
        currentWeightSet = weightSet;
    }

    /**
     * Defines layer procedure for forward and backward calculation (automatic gradient) by applying procedure factory.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected void defineProcedure() throws MatrixException, DynamicParamException, NeuralNetworkException {
        currentWeightSet = weightSet;
        super.defineProcedure();
    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     */
    public MMatrix getInputMatrices(boolean resetPreviousInput) {
        input = new DMatrix(getPreviousLayerWidth(), 1, Initialization.ONE, "Input");
        if (resetPreviousInput) previousOutput = new DMatrix(super.getLayerWidth(), 1);
        return new MMatrix(input);
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public MMatrix getForwardProcedure() throws MatrixException {
        previousOutput.setName("PrevOutput");

        Matrix output = currentWeightSet.weight.dot(input).add(currentWeightSet.bias).add(currentWeightSet.recurrentWeight.dot(previousOutput));

        output = output.apply(activationFunction);

        previousOutput = output;

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
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        return "Activation function: " + activationFunction.getName();
    }

}
