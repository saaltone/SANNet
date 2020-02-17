/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.layer.recurrent;

import core.NeuralNetworkException;
import core.activation.ActivationFunction;
import core.layer.AbstractExecutionLayer;
import core.layer.AbstractLayer;
import core.normalization.Normalization;
import utils.*;
import utils.matrix.*;

import java.util.HashMap;
import java.util.HashSet;

/**
 * Class for Long Short Term Memory (LSTM)<br>
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
public class LSTMLayer extends AbstractExecutionLayer {

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
    private Matrix outPrev;

    /**
     * Matrix to store previous state.
     *
     */
    private Matrix cPrev;

    /**
     * Tanh activation function needed for LSTM
     *
     */
    private ActivationFunction tanh;

    /**
     * Sigmoid activation function needed for LSTM
     *
     */
    private ActivationFunction sigmoid;

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
     * Input matrix for procedure construction.
     *
     */
    private Matrix input;

    /**
     * Constructor for LSTM layer.
     *
     * @param parent reference to parent layer.
     * @param activation activation function used. Not relevant for this layer.
     * @param initialization intialization function for weight.
     * @param params parameters for LSTM layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public LSTMLayer(AbstractLayer parent, ActivationFunction activation, Init initialization, String params) throws NeuralNetworkException, DynamicParamException {
        super (parent, activation, initialization, params);
        tanh = new ActivationFunction(UnaryFunctionType.TANH);
        sigmoid = new ActivationFunction(UnaryFunctionType.SIGMOID);
    }

    /**
     * Returns parameters used for LSTM layer.
     *
     * @return parameters used for LSTM layer.
     */
    public HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("width", DynamicParam.ParamType.INT);
        paramDefs.put("doubleTanh", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("resetStateTraining", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("resetStateTesting", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("regulateDirectWeights", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("regulateRecurrentWeights", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for LSTM layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - width: width (number of nodes) of LSTM layer.<br>
     *     - doubleTanh: true if tanh operation at final output step is executed otherwise false (default value true).<br>
     *     - resetStateTraining: true if output is reset prior training forward step start otherwise false (default value).<br>
     *     - resetStateTesting: true if output is reset prior test forward step start otherwise false (default value).<br>
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value).<br>
     *
     * @param params parameters used for LSTM layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("width")) parent.setWidth(params.getValueAsInteger("width"));
        if (params.hasParam("doubleTanh")) doubleTanh = params.getValueAsBoolean("doubleTanh");
        if (params.hasParam("resetStateTraining")) resetStateTraining = params.getValueAsBoolean("resetStateTraining");
        if (params.hasParam("resetStateTesting")) resetStateTesting = params.getValueAsBoolean("resetStateTesting");
        if (params.hasParam("regulateDirectWeights")) regulateDirectWeights = params.getValueAsBoolean("regulateDirectWeights");
        if (params.hasParam("regulateRecurrentWeights")) regulateRecurrentWeights = params.getValueAsBoolean("regulateRecurrentWeights");
    }

    /**
     * Checks if layer is recurrent layer type.
     *
     * @return always true.
     */
    public boolean isRecurrentLayer() { return true; }

    /**
     * Checks if layer is convolutional layer type.
     *
     * @return always false.
     */
    public boolean isConvolutionalLayer() { return false; }

    /**
     * Initializes LSTM layer.<br>
     * Initializes weights and bias and their gradients.<br>
     *
     */
    public void initialize() throws MatrixException {
        int pLayerWidth = parent.getBackward().getPLayerWidth();
        int nLayerWidth = parent.getBackward().getNLayerWidth();

        Wi = new DMatrix(nLayerWidth, pLayerWidth);
        Wi.init(this.initialization);

        Wf = new DMatrix(nLayerWidth, pLayerWidth);
        Wf.init(this.initialization);

        Wo = new DMatrix(nLayerWidth, pLayerWidth);
        Wo.init(this.initialization);

        Ws = new DMatrix(nLayerWidth, pLayerWidth);
        Ws.init(this.initialization);

        Ui = new DMatrix(nLayerWidth, nLayerWidth);
        Ui.init(this.initialization);

        Uf = new DMatrix(nLayerWidth, nLayerWidth);
        Uf.init(this.initialization);

        Uo = new DMatrix(nLayerWidth, nLayerWidth);
        Uo.init(this.initialization);

        Us = new DMatrix(nLayerWidth, nLayerWidth);
        Us.init(this.initialization);

        bi = new DMatrix(nLayerWidth, 1);

        bf = new DMatrix(nLayerWidth, 1);

        bo = new DMatrix(nLayerWidth, 1);

        bs = new DMatrix(nLayerWidth, 1);

        parent.getBackward().registerWeight(Wi, true, regulateDirectWeights, true);
        parent.getBackward().registerWeight(Wf, true, regulateDirectWeights, true);
        parent.getBackward().registerWeight(Wo, true, regulateDirectWeights, true);
        parent.getBackward().registerWeight(Ws, true, regulateDirectWeights, true);

        parent.getBackward().registerWeight(Ui, true, regulateRecurrentWeights, true);
        parent.getBackward().registerWeight(Uf, true, regulateRecurrentWeights, true);
        parent.getBackward().registerWeight(Uo, true, regulateRecurrentWeights, true);
        parent.getBackward().registerWeight(Us, true, regulateRecurrentWeights, true);

        parent.getBackward().registerWeight(bi, true, false, false);
        parent.getBackward().registerWeight(bf, true, false, false);
        parent.getBackward().registerWeight(bo, true, false, false);
        parent.getBackward().registerWeight(bs, true, false, false);

    }

    /**
     * Resets input.
     *
     * @param resetPreviousInput if true resets also previous input.
     */
    protected void resetInput(boolean resetPreviousInput) throws MatrixException {
        input = new DMatrix(parent.getBackward().getPLayerWidth(), 1, Init.ONE);
        if (resetPreviousInput) {
            outPrev = new DMatrix(parent.getBackward().getNLayer().getWidth(), 1);
            cPrev = new DMatrix(parent.getBackward().getNLayer().getWidth(), 1);
        }
    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @return input matrices for procedure construction.
     */
    protected Sample getInputMatrices() throws MatrixException {
        Sample inputs = new Sample(1);
        inputs.put(0, input);
        return inputs;
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @param normalizers normalizers for layer normalization.
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Sample getForwardProcedure(HashSet<Normalization> normalizers) throws MatrixException {
        if (normalizers.size() > 0) input.setNormalization(normalizers);

        // i = sigmoid(Wi * x + Ui * out(t-1) + bi) → Input gate
        Matrix i = Wi.dot(input).add(Ui.dot(outPrev)).add(bi);
        i = i.apply(sigmoid);

        // f = sigmoid(Wf * x + Uf * out(t-1) + bf) → Forget gate
        Matrix f = Wf.dot(input).add(Uf.dot(outPrev)).add(bf);
        f = f.apply(sigmoid);

        // o = sigmoid(Wo * x + Uo * out(t-1) + bo) → Output gate
        Matrix o = Wo.dot(input).add(Uo.dot(outPrev)).add(bo);
        o = o.apply(sigmoid);

        // s = tanh(Ws * x + Us * out(t-1) + bs) → State update
        Matrix s = Ws.dot(input).add(Us.dot(outPrev)).add(bs);
        s = s.apply(tanh);

        // c = i x s + f x c-1 → Internal cell state
        Matrix c = i.multiply(s).add(cPrev.multiply(f));
        cPrev = c;

        // h = tanh(c) x o or h = c x o → Output
        Matrix c_ = doubleTanh ? c.apply(tanh) : c;
        Matrix h = c_.multiply(o);

        outPrev = h;

        Sample outputs = new Sample(1);
        outputs.put(0, h);
        return outputs;

    }

}
