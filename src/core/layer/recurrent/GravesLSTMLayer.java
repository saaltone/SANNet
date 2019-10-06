/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.layer.recurrent;

import core.NeuralNetworkException;
import core.activation.ActivationFunction;
import core.layer.AbstractExecutionLayer;
import core.layer.AbstractLayer;
import utils.*;

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
public class GravesLSTMLayer extends AbstractExecutionLayer {

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
    private Matrix outPrev;

    /**
     * Matrix to store previous state.
     *
     */
    private Matrix cPrev;

    /**
     * Tanh activation function needed for Graves LSTM
     *
     */
    private ActivationFunction tanh;

    /**
     * Sigmoid activation function needed for Graves LSTM
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
     * Flag if state weights are regulated.
     *
     */
    private boolean regulateStateWeights = false;

    /**
     * Constructor for Graves LSTM layer.<br>
     * Supported parameters are:<br>
     *
     * @param parent reference to parent layer.
     * @param activation activation function used. Not relevant for this layer.
     * @param initialization intialization function for weight.
     * @param params parameters for Graves LSTM layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public GravesLSTMLayer(AbstractLayer parent, ActivationFunction activation, Init initialization, String params) throws NeuralNetworkException, DynamicParamException {
        super (parent, activation, initialization, params);
        tanh = new ActivationFunction(UniFunctionType.TANH);
        sigmoid = new ActivationFunction(UniFunctionType.SIGMOID);
    }

    /**
     * Gets parameters used for Graves LSTM layer.
     *
     * @return parameters used for Graves LSTM layer.
     */
    public HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("width", DynamicParam.ParamType.INT);
        paramDefs.put("doubleTanh", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("resetStateTraining", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("resetStateTesting", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("regulateDirectWeights", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("regulateRecurrentWeights", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("regulateStateWeights", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for Graves LSTM layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - width: width (number of nodes) of Graves LSTM layer.<br>
     *     - doubleTanh: true if tanh operation at final output step is executed otherwise false (default value true).<br>
     *     - resetStateTraining: true if output is reset prior training forward step start otherwise false (default value).<br>
     *     - resetStateTesting: true if output is reset prior test forward step start otherwise false (default value).<br>
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value).<br>
     *     - regulateStateWeights: true if state weights are regulated otherwise false (default value).<br>
     *
     * @param params parameters used for Graves LSTM layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("width")) setWidth(params.getValueAsInteger("width"));
        if (params.hasParam("doubleTanh")) doubleTanh = params.getValueAsBoolean("doubleTanh");
        if (params.hasParam("resetStateTraining")) resetStateTraining = params.getValueAsBoolean("resetStateTraining");
        if (params.hasParam("resetStateTesting")) resetStateTesting = params.getValueAsBoolean("resetStateTesting");
        if (params.hasParam("regulateDirectWeights")) regulateDirectWeights = params.getValueAsBoolean("regulateDirectWeights");
        if (params.hasParam("regulateRecurrentWeights")) regulateRecurrentWeights = params.getValueAsBoolean("regulateRecurrentWeights");
        if (params.hasParam("regulateStateWeights")) regulateStateWeights = params.getValueAsBoolean("regulateStateWeights");
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
     * Initializes Graves LSTM layer.<br>
     * Initializes weights and bias and their gradients.<br>
     *
     */
    public void initialize() {
        int pLayerWidth = parent.getBackward().getPLayer().getWidth();
        int nLayerWidth = parent.getBackward().getNLayer().getWidth();

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

        Ci = new DMatrix(nLayerWidth, 1);
        Ci.init(this.initialization);

        Cf = new DMatrix(nLayerWidth, 1);
        Cf.init(this.initialization);

        Co = new DMatrix(nLayerWidth, 1);
        Co.init(this.initialization);

        bi = new DMatrix(nLayerWidth, 1);
        bi.init(this.initialization);

        bf = new DMatrix(nLayerWidth, 1);
        bf.init(this.initialization);

        bo = new DMatrix(nLayerWidth, 1);
        bo.init(this.initialization);

        bs = new DMatrix(nLayerWidth, 1);
        bs.init(this.initialization);

        backward.registerWeight(Wi, true, regulateDirectWeights, true);
        backward.registerWeight(Wf, true, regulateDirectWeights, true);
        backward.registerWeight(Wo, true, regulateDirectWeights, true);
        backward.registerWeight(Ws, true, regulateDirectWeights, true);

        backward.registerWeight(Ui, true, regulateRecurrentWeights, true);
        backward.registerWeight(Uf, true, regulateRecurrentWeights, true);
        backward.registerWeight(Uo, true, regulateRecurrentWeights, true);
        backward.registerWeight(Us, true, regulateRecurrentWeights, true);

        backward.registerWeight(Ci, true, regulateStateWeights, true);
        backward.registerWeight(Cf, true, regulateStateWeights, true);
        backward.registerWeight(Co, true, regulateStateWeights, true);

        backward.registerWeight(bi, true, false, false);
        backward.registerWeight(bf, true, false, false);
        backward.registerWeight(bo, true, false, false);
        backward.registerWeight(bs, true, false, false);
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @param input input of forward procedure.
     * @param reset reset recurring inputs of procedure.
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix getForwardProcedure(Matrix input, boolean reset) throws MatrixException {
        if (reset) {
            outPrev = new DMatrix(parent.getBackward().getNLayer().getWidth(), 1);
            cPrev = new DMatrix(parent.getBackward().getNLayer().getWidth(), 1);
        }

        // i = sigmoid(Wi * x + Ui * out(t-1) + Ci * c(t-1) + bi) → Input gate
        Matrix i = Wi.dot(input).add(Ui.dot(outPrev)).add(Ci.multiply(cPrev)).add(bi);
        i = i.apply(sigmoid);

        // f = sigmoid(Wf * x + Uf * out(t-1) + Cf * c(t-1) + bf) → Forget gate
        Matrix f = Wf.dot(input).add(Uf.dot(outPrev)).add(Cf.multiply(cPrev)).add(bf);
        f = f.apply(sigmoid);

        // s = tanh(Ws * x + Us * out(t-1) + bs) → State update
        Matrix s = Ws.dot(input).add(Us.dot(outPrev)).add(bs);
        s = s.apply(tanh);

        // c = i x s + f x c(t-1) → Internal cell state
        Matrix c = i.multiply(s).add(cPrev.multiply(f));
        cPrev = c;

        // o = sigmoid(Wo * x + Uo * out(t-1) + Co * ct + bo) → Output gate
        Matrix o = Wo.dot(input).add(Uo.dot(outPrev)).add(Co.multiply(c)).add(bo);
        o = o.apply(sigmoid);

        // h = tanh(c) x o or h = c x o → Output
        Matrix c_ = doubleTanh ? c.apply(tanh) : c;
        Matrix h = c_.multiply(o);

        outPrev = h;

        return h;

    }

}
