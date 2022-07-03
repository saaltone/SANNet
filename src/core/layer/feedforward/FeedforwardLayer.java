/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
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
     * Parameter name types for dense layer.
     *     - regulateDirectWeights: true if (direct) weights are regulated otherwise false. Default value true.<br>
     *     - splitOutputAtPosition: splits output at specific position.<br>
     *     - connectFromPreviousLayer: creates connection from output of specified previous layer.<br>
     *     - joinPreviousLayerInput: if true join inputs of previous layers otherwise connects via weight and summation. Default value true.<br>
     *
     */
    private final static String paramNameTypes = "(regulateDirectWeights:BOOLEAN), " +
            "(splitOutputAtPosition:INT), " +
            "(connectFromPreviousLayer:INT), " +
            "(joinPreviousLayerInput:BOOLEAN)";

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
         * Previous connection layer weight matrix.
         *
         */
        private final Matrix previousConnectLayerWeight;

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
         * @param previousConnectLayerWidth width of previous connection layer.
         * @param joinPreviousLayerInput if true input and previous connect input are joined otherwise previous connect layer input is added through dedicated weight.
         * @param regulateDirectWeights if true direct weights are regulated.
         */
        FeedforwardWeightSet(Initialization initialization, int previousLayerWidth, int layerWidth, int previousConnectLayerWidth, boolean joinPreviousLayerInput, boolean regulateDirectWeights) {
            weight = (previousConnectLayerWidth < 0 || !joinPreviousLayerInput) ? new DMatrix(layerWidth, previousLayerWidth, initialization) : new DMatrix(layerWidth, previousLayerWidth + previousConnectLayerWidth, initialization);
            weight.setName("Weight");
            bias = new DMatrix(layerWidth, 1);
            bias.setName("Bias");

            weights.add(weight);
            weights.add(bias);

            registerWeight(weight, regulateDirectWeights, true);
            registerWeight(bias, false, false);

            if (previousConnectLayerWidth > -1 && !joinPreviousLayerInput) {
                previousConnectLayerWeight = new DMatrix(layerWidth, previousConnectLayerWidth, initialization);
                previousConnectLayerWeight.setName("PreviousConnectWeight");
                weights.add(previousConnectLayerWeight);
                registerWeight(previousConnectLayerWeight, false, false);
            }
            else previousConnectLayerWeight = null;
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
            if (previousConnectLayerWeight != null) previousConnectLayerWeight.initialize(initialization);
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
     * Splits output at given position.
     *
     */
    private int splitOutputAtPosition;

    /**
     * Index of previous connection from output of specified earlier layer.
     *
     */
    private int connectFromPreviousLayer;

    /**
     * If true input and previous connect input are joined otherwise previous connect input is added through dedicated weight.
     *
     */
    private boolean joinPreviousLayerInput;

    /**
     * Input matrix for procedure construction.
     *
     */
    private TreeMap<Integer, MMatrix> inputs;

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
        splitOutputAtPosition = -1;
        connectFromPreviousLayer = -1;
        joinPreviousLayerInput = true;
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
     *     - splitOutputAtPosition: splits output at specific position.<br>
     *     - connectFromPreviousLayer: creates connection from output of specified previous layer.<br>
     *     - joinPreviousLayerInput: if true join inputs of previous layers otherwise connects via weight and summation. Default value true.<br>
     *
     * @param params parameters used for feedforward layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("regulateDirectWeights")) regulateDirectWeights = params.getValueAsBoolean("regulateDirectWeights");
        if (params.hasParam("splitOutputAtPosition")) splitOutputAtPosition = params.getValueAsInteger("splitOutputAtPosition");
        if (params.hasParam("connectFromPreviousLayer")) {
            connectFromPreviousLayer = params.getValueAsInteger("connectFromPreviousLayer");
            if (connectFromPreviousLayer < 0 || connectFromPreviousLayer > getLayerIndex() - 2) throw new DynamicParamException("Previous connect layer index must be positive value and connection must be created from a layer having index at least 2 smaller than this layer: " + getLayerIndex());
        }
        if (params.hasParam("joinPreviousLayerInput")) joinPreviousLayerInput = params.getValueAsBoolean("joinPreviousLayerInput");
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
     * Returns true if input is unjoined otherwise returns false.
     *
     * @return true if input is unjoined otherwise returns false.
     */
    protected boolean isUnjoinedInput() {
        return !joinPreviousLayerInput;
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
        weightSet = new FeedforwardWeightSet(initialization, getPreviousLayerWidth(), super.getLayerWidth(), connectFromPreviousLayer > -1 ? getPreviousLayerWidth(connectFromPreviousLayer) : -1, joinPreviousLayerInput, regulateDirectWeights);
    }

    /**
     * Defines layer procedure for forward and backward calculation (automatic gradient) by applying procedure factory.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected void defineProcedure() throws MatrixException, DynamicParamException, NeuralNetworkException {
        super.defineProcedure();
        if (connectFromPreviousLayer > -1) addInputSequence(connectFromPreviousLayer);
    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public TreeMap<Integer, MMatrix> getInputMatrices(boolean resetPreviousInput) throws MatrixException {
        int numberOfInputs = connectFromPreviousLayer < 0 ? 1 : 2;
        inputs = new TreeMap<>();
        if (isUnjoinedInput()) {
            for (int inputIndex = 0; inputIndex < numberOfInputs; inputIndex++) {
                int connectFromLayer = inputIndex == 0 ? getLayerIndex() - 1 : connectFromPreviousLayer;
                Matrix input = new DMatrix(getPreviousLayerWidth(connectFromLayer), 1, Initialization.ONE);
                if (inputIndex == 0) input = handleBidirectionalInput(input);
                input.setName("Input" + connectFromLayer);
                inputs.put(inputIndex, new MMatrix(input));
            }
        }
        else {
            Matrix[] inputMatrices = new Matrix[numberOfInputs];
            for (int inputIndex = 0; inputIndex < numberOfInputs; inputIndex++) {
                int connectFromLayer = inputIndex == 0 ? getLayerIndex() - 1 : connectFromPreviousLayer;
                Matrix input = new DMatrix(getPreviousLayerWidth(connectFromLayer), 1, Initialization.ONE);
                if (inputIndex == 0) input = handleBidirectionalInput(input);
                input.setName("Input" + connectFromLayer);
                inputMatrices[inputIndex] = input;
            }
            inputs.put(0, new MMatrix(new JMatrix(inputMatrices, true)));
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
        Matrix output;
        if (connectFromPreviousLayer > -1) {
            if (joinPreviousLayerInput) {
                Matrix joinedInput = inputs.get(0).get(0);
                joinedInput.setName("JoinedInput");
                output = weightSet.weight.dot(joinedInput);
            }
            else {
                output = weightSet.weight.dot(inputs.get(0).get(0));
                Matrix previousConnectOutput = weightSet.previousConnectLayerWeight.dot(inputs.get(1).get(0));
                output = output.add(previousConnectOutput);
            }
        }
        else {
            output = weightSet.weight.dot(inputs.get(0).get(0));
        }

        output = output.add(weightSet.bias);

        if (splitOutputAtPosition == -1) {
            if (activationFunction != null) output = output.apply(activationFunction);
        }
        else {
            Matrix result = output.split(splitOutputAtPosition, true);
            if (activationFunction != null) output.apply(result, activationFunction);
            output = result;
        }

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
        return (activationFunction == null ? "" : "Activation function: " + activationFunction.getName() + ", ") + "Split output at: " + (splitOutputAtPosition != -1 ? splitOutputAtPosition : "N/A" + ", Connect from previous layer: " + (connectFromPreviousLayer != -1 ? connectFromPreviousLayer : "N/A") + ", Join previous layer inputs: " + (joinPreviousLayerInput ? "Yes" : "No"));
    }

}
