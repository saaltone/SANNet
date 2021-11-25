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
 * Implements Peephole Long Short Term Memory (LSTM)<br>
 * <br>
 * Reference: https://en.wikipedia.org/wiki/Long_short-term_memory<br>
 * <br>
 * Equations applied for forward operation:<br>
 *   i = sigmoid(Wi * x + Ui * c(t-1) + bi) → Input gate<br>
 *   f = sigmoid(Wf * x + Uf * c(t-1) + bf) → Forget gate<br>
 *   o = sigmoid(Wo * x + Uo * c(t-1) + bo) → Output gate<br>
 *   s = tanh(Ws * x + bs) → State update<br>
 *   c = i x s + f x c-1 → Internal cell state<br>
 *   h = tanh(c) x o or h = c x o → Output<br>
 *
 */
public class PeepholeLSTMLayer extends AbstractRecurrentLayer {

    /**
     * Parameter name types for Peephole LSTM layer.
     *     - doubleTanh: true if tanh operation at final output step is executed otherwise false (default value true).<br>
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value true).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value false).<br>
     *
     */
    private final static String paramNameTypes = "(doubleTanh:BOOLEAN), " +
            "(regulateDirectWeights:BOOLEAN), " +
            "(regulateRecurrentWeights:BOOLEAN)";

    /**
     * Weights for input gate
     *
     */
    private Matrix Wi;

    /**
     * Weights for forget gate
     *
     */
    private Matrix Wf;

    /**
     * Weights for output gate
     *
     */
    private Matrix Wo;

    /**
     * Weights for state
     *
     */
    private Matrix Ws;

    /**
     * Weights for recurrent input gate
     *
     */
    private Matrix Ui;

    /**
     * Weights for recurrent forget gate
     *
     */
    private Matrix Uf;

    /**
     * Weights for recurrent output gate
     *
     */
    private Matrix Uo;

    /**
     * Bias for input gate
     *
     */
    private Matrix bi;

    /**
     * Bias for forget gate
     *
     */
    private Matrix bf;

    /**
     * Bias for output gate
     *
     */
    private Matrix bo;

    /**
     * Bias for state
     *
     */
    private Matrix bs;

    /**
     * Matrix to store previous state.
     *
     */
    private Matrix previousCellState;

    /**
     * Tanh activation function needed for Peephole LSTM
     *
     */
    private final ActivationFunction tanh;

    /**
     * Sigmoid activation function needed for Peephole LSTM
     *
     */
    private final ActivationFunction sigmoid;

    /**
     * Sigmoid activation function for output
     *
     */
    private final ActivationFunction activationFunction;

    /**
     * Flag if tanh operation is performed also for last output function.
     *
     */
    private boolean doubleTanh;

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
     * Constructor for Peephole LSTM layer.
     *
     * @param layerIndex layer Index.
     * @param initialization initialization function for weight.
     * @param params parameters for Peephole LSTM layer.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public PeepholeLSTMLayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, params);
        tanh = new ActivationFunction(UnaryFunctionType.TANH);
        sigmoid = new ActivationFunction(UnaryFunctionType.SIGMOID);
        activationFunction = tanh;
    }

    /**
     * Constructor for Peephole LSTM layer.
     *
     * @param layerIndex layer Index.
     * @param activationFunction activation function used.
     * @param initialization initialization function for weight.
     * @param params parameters for Peephole LSTM layer.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public PeepholeLSTMLayer(int layerIndex, ActivationFunction activationFunction, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, params);
        tanh = new ActivationFunction(UnaryFunctionType.TANH);
        sigmoid = new ActivationFunction(UnaryFunctionType.SIGMOID);
        this.activationFunction = activationFunction == null ? tanh : activationFunction;
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        doubleTanh = true;
        regulateDirectWeights = true;
        regulateRecurrentWeights = false;
    }

    /**
     * Returns parameters used for Peephole LSTM layer.
     *
     * @return parameters used for Peephole LSTM layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + PeepholeLSTMLayer.paramNameTypes;
    }

    /**
     * Sets parameters used for Peephole LSTM layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - doubleTanh: true if tanh operation at final output step is executed otherwise false (default value true).<br>
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value true).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value false).<br>
     *
     * @param params parameters used for Peephole LSTM layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("doubleTanh")) doubleTanh = params.getValueAsBoolean("doubleTanh");
        if (params.hasParam("regulateDirectWeights")) regulateDirectWeights = params.getValueAsBoolean("regulateDirectWeights");
        if (params.hasParam("regulateRecurrentWeights")) regulateRecurrentWeights = params.getValueAsBoolean("regulateRecurrentWeights");
    }

    /**
     * Initializes Peephole LSTM layer.<br>
     * Initializes weights and bias and their gradients.<br>
     *
     */
    public void initialize() {
        int previousLayerWidth = getPreviousLayerWidth();
        int layerWidth = getLayerWidth();

        Wi = new DMatrix(layerWidth, previousLayerWidth, initialization, "Wi");
        Wf = new DMatrix(layerWidth, previousLayerWidth, initialization, "Wf");
        Wo = new DMatrix(layerWidth, previousLayerWidth, initialization, "Wo");
        Ws = new DMatrix(layerWidth, previousLayerWidth, initialization, "Ws");

        Ui = new DMatrix(layerWidth, layerWidth, initialization, "Ui");
        Uf = new DMatrix(layerWidth, layerWidth, initialization, "Uf");
        Uo = new DMatrix(layerWidth, layerWidth, initialization, "Uo");

        bi = new DMatrix(layerWidth, 1, "bi");
        bf = new DMatrix(layerWidth, 1, "bf");
        bo = new DMatrix(layerWidth, 1, "bo");
        bs = new DMatrix(layerWidth, 1, "bs");

        registerWeight(Wi, regulateDirectWeights, true);
        registerWeight(Wf, regulateDirectWeights, true);
        registerWeight(Wo, regulateDirectWeights, true);
        registerWeight(Ws, regulateDirectWeights, true);

        registerWeight(Ui, regulateRecurrentWeights, true);
        registerWeight(Uf, regulateRecurrentWeights, true);
        registerWeight(Uo, regulateRecurrentWeights, true);

        registerWeight(bi, false, false);
        registerWeight(bf, false, false);
        registerWeight(bo, false, false);
        registerWeight(bs, false, false);

    }

    /**
     * Reinitializes layer.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void reinitialize() throws MatrixException, NeuralNetworkException {
        Wi.initialize(this.initialization);
        Wf.initialize(this.initialization);
        Wo.initialize(this.initialization);
        Ws.initialize(this.initialization);

        Ui.initialize(this.initialization);
        Uf.initialize(this.initialization);
        Uo.initialize(this.initialization);

        bi.reset();
        bf.reset();
        bo.reset();
        bs.reset();

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
        if (resetPreviousInput) previousCellState = new DMatrix(getLayerWidth(), 1);
        return new MMatrix(input);
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public MMatrix getForwardProcedure() throws MatrixException {
        previousCellState.setName("PrevCellState");

        // i = sigmoid(Wi * x + Ui * c(t-1) + bi) → Input gate
        Matrix i = Wi.dot(input).add(Ui.dot(previousCellState)).add(bi);
        i = i.apply(sigmoid);
        i.setName("i");

        // f = sigmoid(Wf * x + Uf * c(t-1) + bf) → Forget gate
        Matrix f = Wf.dot(input).add(Uf.dot(previousCellState)).add(bf);
        f = f.apply(sigmoid);
        f.setName("f");

        // o = sigmoid(Wo * x + Uo * c(t-1) + bo) → Output gate
        Matrix o = Wo.dot(input).add(Uo.dot(previousCellState)).add(bo);
        o = o.apply(sigmoid);
        o.setName("o");

        // s = tanh(Ws * x + bs) → State update
        Matrix s = Ws.dot(input).add(bs);
        s = s.apply(tanh);
        s.setName("s");

        // c = i x s + f x c-1 → Internal cell state
        Matrix c = i.multiply(s).add(previousCellState.multiply(f));
        c.setName("c");

        previousCellState = c;

        // h = activationFunction(c) x o or h = c x o → Output
        Matrix h = (doubleTanh ? c.apply(activationFunction) : c).multiply(o);
        h.setName("Output");

        MMatrix outputs = new MMatrix(1, "Output");
        outputs.put(0, h);
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
        return null;
    }

}
