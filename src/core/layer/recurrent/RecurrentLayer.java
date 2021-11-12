/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.layer.recurrent;

import core.network.NeuralNetworkException;
import core.activation.ActivationFunction;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

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
     * Weight matrix.
     *
     */
    private Matrix weight;

    /**
     * Bias matrix.
     *
     */
    private Matrix bias;

    /**
     * Weight matrix for recurrent input.
     *
     */
    private Matrix recurrentWeight;

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
     * Initializes recurrent layer.<br>
     * Initializes weights and bias and their gradients.<br>
     *
     */
    public void initialize() {
        int previousLayerWidth = getPreviousLayerWidth();
        int layerWidth = getLayerWidth();

        weight = new DMatrix(layerWidth, previousLayerWidth, initialization, "Weight");

        recurrentWeight = new DMatrix(layerWidth, layerWidth, initialization, "RecurrentWeight");

        bias = new DMatrix(layerWidth, 1, "bias");

        registerWeight(weight, regulateDirectWeights, true);

        registerWeight(recurrentWeight, regulateRecurrentWeights, true);

        registerWeight(bias, false, false);

    }

    /**
     * Reinitializes layer.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void reinitialize() throws MatrixException, NeuralNetworkException {
        weight.initialize(this.initialization);
        recurrentWeight.initialize(this.initialization);
        bias.reset();

        super.reinitialize();
    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     */
    public MMatrix getInputMatrices(boolean resetPreviousInput) {
        input = new DMatrix(getPreviousLayerWidth(), 1, Initialization.ONE, "Input");
        if (resetPreviousInput) previousOutput = new DMatrix(getLayerWidth(), 1);
        return new MMatrix(input);
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public MMatrix getForwardProcedure() throws MatrixException {
        input.setNormalize(true);
        input.setRegularize(true);

        previousOutput.setName("PrevOutput");

        Matrix output = weight.dot(input).add(bias).add(recurrentWeight.dot(previousOutput));

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
