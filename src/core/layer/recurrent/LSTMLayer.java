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
 * Implements Long Short Term Memory (LSTM)<br>
 * <br>
 * Reference: https://en.wikipedia.org/wiki/Long_short-term_memory<br>
 * <br>
 * Equations applied for forward operation:<br>
 *   i = sigmoid(Wi * x + Ui * out(t-1) + bi) → Input gate<br>
 *   f = sigmoid(Wf * x + Uf * out(t-1) + bf) → Forget gate<br>
 *   o = sigmoid(Wo * x + Uo * out(t-1) + bo) → Output gate<br>
 *   s = tanh(Ws * x + Us * out(t-1) + bs) → State update<br>
 *   c = i x s + f x c-1 → Internal cell state<br>
 *   h = tanh(c) x o or h = c x o → Output<br>
 *
 */
public class LSTMLayer extends AbstractRecurrentLayer {

    /**
     * Parameter name types for LSTM layer.
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
     * Weights for recurrent state
     *
     */
    private Matrix Us;

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
     * Matrix to store previous output.
     *
     */
    private Matrix previousOutput;

    /**
     * Matrix to store previous state.
     *
     */
    private Matrix previousCellState;

    /**
     * Tanh activation function needed for LSTM
     *
     */
    private final ActivationFunction tanh;

    /**
     * Sigmoid activation function needed for LSTM
     *
     */
    private final ActivationFunction sigmoid;

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
     * Constructor for LSTM layer.
     *
     * @param layerIndex layer Index.
     * @param initialization initialization function for weight.
     * @param params parameters for LSTM layer.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public LSTMLayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
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
        doubleTanh = true;
        regulateDirectWeights = true;
        regulateRecurrentWeights = false;
    }

    /**
     * Returns parameters used for LSTM layer.
     *
     * @return parameters used for LSTM layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + LSTMLayer.paramNameTypes;
    }

    /**
     * Sets parameters used for LSTM layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - doubleTanh: true if tanh operation at final output step is executed otherwise false (default value true).<br>
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value true).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value false).<br>
     *
     * @param params parameters used for LSTM layer.
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
     * Initializes LSTM layer.<br>
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
        Us = new DMatrix(layerWidth, layerWidth, initialization, "Us");

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
        registerWeight(Us, regulateRecurrentWeights, true);

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
        Us.initialize(this.initialization);

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
        if (resetPreviousInput) {
            previousOutput = new DMatrix(getLayerWidth(), 1);
            previousCellState = new DMatrix(getLayerWidth(), 1);
        }
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
        previousCellState.setName("PrevC");

        // i = sigmoid(Wi * x + Ui * out(t-1) + bi) → Input gate
        Matrix i = Wi.dot(input).add(Ui.dot(previousOutput)).add(bi);
        i = i.apply(sigmoid);
        i.setName("i");

        // f = sigmoid(Wf * x + Uf * out(t-1) + bf) → Forget gate
        Matrix f = Wf.dot(input).add(Uf.dot(previousOutput)).add(bf);
        f = f.apply(sigmoid);
        f.setName("f");

        // o = sigmoid(Wo * x + Uo * out(t-1) + bo) → Output gate
        Matrix o = Wo.dot(input).add(Uo.dot(previousOutput)).add(bo);
        o = o.apply(sigmoid);
        o.setName("o");

        // s = tanh(Ws * x + Us * out(t-1) + bs) → State update
        Matrix s = Ws.dot(input).add(Us.dot(previousOutput)).add(bs);
        s = s.apply(tanh);
        s.setName("s");

        // c = i x s + f x c-1 → Internal cell state
        Matrix c = i.multiply(s).add(previousCellState.multiply(f));
        c.setName("c");

        previousCellState = c;

        // h = tanh(c) x o or h = c x o → Output
        Matrix h = (doubleTanh ? c.apply(tanh) : c).multiply(o);
        h.setName("Output");

        previousOutput = h;

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
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        return null;
    }

}
