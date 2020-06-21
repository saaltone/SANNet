/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.layer.recurrent;

import core.NeuralNetworkException;
import core.activation.ActivationFunction;
import utils.*;
import utils.matrix.*;

import java.util.HashMap;

/**
 * Class for Graves type of Long Short Term Memory (LSTM)<br>
 * <br>
 * Reference: https://www.cs.toronto.edu/~graves/phd.pdf<br>
 * <br>
 * Equations applied for forward operation:<br>
 *   i = sigmoid(Wi * x + Ui * out(t-1) + Ci * c(t-1) + bi) → Input gate<br>
 *   f = sigmoid(Wf * x + Uf * out(t-1) + Cf * c(t-1) + bf) → Forget gate<br>
 *   s = tanh(Ws * x + Us * out(t-1) + bs) → State update<br>
 *   c = i x s + f x c(t-1) → Internal cell state<br>
 *   o = sigmoid(Wo * x + Uo * out(t-1) + Co * ct + bo) → Output gate<br>
 *   h = tanh(c) x o or h = c x o → Output<br>
 *
 */
public class GravesLSTMLayer extends AbstractRecurrentLayer {

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
     * Weights for input cell state
     *
     */
    private Matrix Ci;

    /**
     * Weights for forget cell state
     *
     */
    private Matrix Cf;

    /**
     * Weights for output cell state
     *
     */
    private Matrix Co;

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
     * Tanh activation function needed for Graves LSTM
     *
     */
    private final ActivationFunction tanh;

    /**
     * Sigmoid activation function needed for Graves LSTM
     *
     */
    private final ActivationFunction sigmoid;

    /**
     * Flag if tanh operation is performed also for last output function.
     *
     */
    private boolean doubleTanh = true;

    /**
     * Flag if direct (non-recurrent) weights are regulated.
     *
     */
    private boolean regulateDirectWeights = true;

    /**
     * Flag if recurrent weights are regulated.
     *
     */
    private boolean regulateRecurrentWeights = false;

    /**
     * Flag if state weights are regulated.
     *
     */
    private boolean regulateStateWeights = false;

    /**
     * Input matrix for procedure construction.
     *
     */
    private Matrix input;

    /**
     * Constructor for Graves LSTM layer.<br>
     * Supported parameters are:<br>
     *
     * @param layerIndex layer Index.
     * @param initialization initialization function for weight.
     * @param params parameters for Graves LSTM layer.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public GravesLSTMLayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, params);
        setParams(new DynamicParam(params, getParamDefs()));
        tanh = new ActivationFunction(UnaryFunctionType.TANH);
        sigmoid = new ActivationFunction(UnaryFunctionType.SIGMOID);
    }

    /**
     * Returns parameters used for Graves LSTM layer.
     *
     * @return parameters used for Graves LSTM layer.
     */
    public HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>(super.getParamDefs());
        paramDefs.put("doubleTanh", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("regulateDirectWeights", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("regulateRecurrentWeights", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("regulateStateWeights", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for Graves LSTM layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - doubleTanh: true if tanh operation at final output step is executed otherwise false (default value true).<br>
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value).<br>
     *     - regulateStateWeights: true if state weights are regulated otherwise false (default value).<br>
     *
     * @param params parameters used for Graves LSTM layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("doubleTanh")) doubleTanh = params.getValueAsBoolean("doubleTanh");
        if (params.hasParam("regulateDirectWeights")) regulateDirectWeights = params.getValueAsBoolean("regulateDirectWeights");
        if (params.hasParam("regulateRecurrentWeights")) regulateRecurrentWeights = params.getValueAsBoolean("regulateRecurrentWeights");
        if (params.hasParam("regulateStateWeights")) regulateStateWeights = params.getValueAsBoolean("regulateStateWeights");
    }

    /**
     * Initializes Graves LSTM layer.<br>
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

        Ci = new DMatrix(layerWidth, 1, initialization, "Ci");
        Cf = new DMatrix(layerWidth, 1, initialization, "Cf");
        Co = new DMatrix(layerWidth, 1, initialization, "Co");

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

        registerWeight(Ci, regulateStateWeights, true);
        registerWeight(Cf, regulateStateWeights, true);
        registerWeight(Co, regulateStateWeights, true);

        registerWeight(bi, false, false);
        registerWeight(bf, false, false);
        registerWeight(bo, false, false);
        registerWeight(bs, false, false);

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

        // i = sigmoid(Wi * x + Ui * out(t-1) + Ci * c(t-1) + bi) → Input gate
        Matrix i = Wi.dot(input).add(Ui.dot(previousOutput)).add(Ci.multiply(previousCellState)).add(bi);
        i = i.apply(sigmoid);
        i.setName("i");

        // f = sigmoid(Wf * x + Uf * out(t-1) + Cf * c(t-1) + bf) → Forget gate
        Matrix f = Wf.dot(input).add(Uf.dot(previousOutput)).add(Cf.multiply(previousCellState)).add(bf);
        f = f.apply(sigmoid);
        f.setName("f");

        // s = tanh(Ws * x + Us * out(t-1) + bs) → State update
        Matrix s = Ws.dot(input).add(Us.dot(previousOutput)).add(bs);
        s = s.apply(tanh);
        s.setName("s");

        // c = i x s + f x c(t-1) → Internal cell state
        Matrix c = i.multiply(s).add(previousCellState.multiply(f));
        c.setName("c");

        previousCellState = c;

        // o = sigmoid(Wo * x + Uo * out(t-1) + Co * ct + bo) → Output gate
        Matrix o = Wo.dot(input).add(Uo.dot(previousOutput)).add(Co.multiply(c)).add(bo);
        o = o.apply(sigmoid);
        o.setName("o");

        // h = tanh(c) x o or h = c x o → Output
        Matrix h = (doubleTanh ? c.apply(tanh) : c).multiply(o);
        h.setName("Output");

        previousOutput = h;

        MMatrix outputs = new MMatrix(1, "Output");
        outputs.put(0, h);
        return outputs;

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
