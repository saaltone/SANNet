/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.layer.recurrent;

import core.network.NeuralNetworkException;
import core.activation.ActivationFunction;
import utils.*;
import utils.matrix.*;

import java.util.HashSet;

/**
 * Implements minimal gated recurrent unit (GRU) layer.<br>
 * <br>
 * Reference: https://en.wikipedia.org/wiki/Gated_recurrent_unit<br>
 * <br>
 * Equations applied for forward operation:<br>
 *     f = sigmoid(Wf * x + Uf * out(t-1) + bf) → Forget gate<br>
 *     h = tanh(Wh * x + Uh * out(t-1) * r + bh) → Input activation<br>
 *     s = (1 - f) x h + f x out(t-1) → Internal state<br>
 *
 */
public class MinGRULayer extends AbstractRecurrentLayer {

    /**
     * Parameter name types for abstract Min GRU layer.
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value true).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value false).<br>
     *
     */
    private final static String paramNameTypes = "(regulateDirectWeights:BOOLEAN), " +
            "(regulateRecurrentWeights:BOOLEAN)";

    /**
     * Weights for forget gate
     *
     */
    private Matrix Wf;

    /**
     * Weights for input activation
     *
     */
    private Matrix Wh;

    /**
     * Weights for recurrent forget gate
     *
     */
    private Matrix Uf;

    /**
     * Weights for current input activation
     *
     */
    private Matrix Uh;

    /**
     * Bias for forget gate
     *
     */
    private Matrix bf;

    /**
     * Bias for input activation
     *
     */
    private Matrix bh;

    /**
     * Ones matrix for calculation of z
     *
     */
    private Matrix ones;

    /**
     * Matrix to store previous output
     *
     */
    private Matrix previousOutput;

    /**
     * Tanh activation function needed for Minimal GRU
     *
     */
    private final ActivationFunction tanh;

    /**
     * Sigmoid activation function needed for Minimal GRU
     *
     */
    private final ActivationFunction sigmoid;

    /**
     * Flag if direct (non-recurrent) weights are regulated.
     *
     */
    private boolean regulateDirectWeights;

    /**
     * Flag if recurrent weights are regulated.
     *
     */
    private boolean regulateRecurrentWeights;

    /**
     * Input matrix for procedure construction.
     *
     */
    private Matrix input;

    /**
     * Constructor for Minimal GRU layer.
     *
     * @param layerIndex layer Index.
     * @param initialization initialization function for weight.
     * @param params parameters for minimal GRU layer.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public MinGRULayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, params);
        tanh = new ActivationFunction(UnaryFunctionType.TANH);
        sigmoid = new ActivationFunction(UnaryFunctionType.SIGMOID);
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
     * Returns parameters used for Minimal GRU layer.
     *
     * @return parameters used for Minimal GRU layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + MinGRULayer.paramNameTypes;
    }

    /**
     * Sets parameters used for Minimal GRU layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value true).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value false).<br>
     *
     * @param params parameters used for Minimal GRU layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("regulateDirectWeights")) regulateDirectWeights = params.getValueAsBoolean("regulateDirectWeights");
        if (params.hasParam("regulateRecurrentWeights")) regulateRecurrentWeights = params.getValueAsBoolean("regulateRecurrentWeights");
    }

    /**
     * Initializes Minimal GRU layer.<br>
     * Initializes weights and bias and their gradients.<br>
     *
     */
    public void initialize() {
        int previousLayerWidth = getPreviousLayerWidth();
        int layerWidth = getLayerWidth();

        Wf = new DMatrix(layerWidth, previousLayerWidth, initialization, "Wf");
        Wh = new DMatrix(layerWidth, previousLayerWidth, initialization, "Wh");

        Uf = new DMatrix(layerWidth, layerWidth, initialization, "Uf");
        Uh = new DMatrix(layerWidth, layerWidth, initialization, "Uh");

        bf = new DMatrix(layerWidth, 1, "bf");
        bh = new DMatrix(layerWidth, 1, "bh");

        registerWeight(Wf, regulateDirectWeights, true);
        registerWeight(Wh, regulateDirectWeights, true);

        registerWeight(Uf, regulateRecurrentWeights, true);
        registerWeight(Uh, regulateRecurrentWeights, true);

        registerWeight(bf, false, false);
        registerWeight(bh, false, false);

    }

    /**
     * Reinitializes layer.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void reinitialize() throws MatrixException, NeuralNetworkException {
        Wf.initialize(this.initialization);
        Wh.initialize(this.initialization);

        Uf.initialize(this.initialization);
        Uh.initialize(this.initialization);

        bf.reset();
        bh.reset();

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

        // f = sigmoid(Wf * x + Uf * out(t-1) + bf) → Forget gate
        Matrix f = Wf.dot(input).add(Uf.dot(previousOutput)).add(bf);
        f = f.apply(sigmoid);
        f.setName("f");

        // h = tanh(Wh * x + Uh * out(t-1) * r + bh) → Input activation
        Matrix h = Wh.dot(input).add(Uh.dot(previousOutput).multiply(f)).add(bh);
        h = h.apply(tanh);
        h.setName("h");

        // s = (1 - f) x h + f x out(t-1) → Internal state
        ones = (ones == null) ? new DMatrix(f.getRows(), f.getColumns(), Initialization.ONE) : ones;
        ones.setName("1");

        Matrix s = ones.subtract(f).multiply(h).add(f.multiply(previousOutput));
        s.setName("Output");

        previousOutput = s;

        MMatrix outputs = new MMatrix(1, "Output");
        outputs.put(0, s);
        return outputs;

    }

    /**
     * Returns matrices for which gradient is not calculated.
     *
     * @return matrices for which gradient is not calculated.
     */
    protected HashSet<Matrix> getStopGradients() {
        HashSet<Matrix> stopGradients = new HashSet<>();
        stopGradients.add(ones);
        return stopGradients;
    }

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        return null;
    }

}

