/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.layer.recurrent;

import core.NeuralNetworkException;
import core.activation.ActivationFunction;
import core.activation.ActivationFunctionType;
import core.layer.AbstractExecutionLayer;
import core.layer.AbstractLayer;
import utils.*;

import java.util.HashMap;
import java.util.TreeMap;

/**
 * Class for Long Short Term Memory (LSTM)<br>
 * <br>
 * Reference: https://www.cs.toronto.edu/~tingwuwang/rnn_tutorial.pdf<br>
 * <br>
 * Equations applied for forward operation:<br>
 *   i = sigmoid(Wi * x + Ui * out(t-1) + bi) → Input gate<br>
 *   f = sigmoid(Wf * x + Uf * out(t-1) + bf) → Forget gate<br>
 *   s = tanh(Ws * x + Us * out(t-1) + bs) → State update<br>
 *   c = i x s + f x c-1 → Internal cell state<br>
 *   o = sigmoid(Wo * x + Uo * out(t-1) + bo) → Output gate<br>
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
     * Tanh activation function needed for LSTM
     *
     */
    private ActivationFunction tanh = new ActivationFunction(ActivationFunctionType.TANH);

    /**
     * Sigmoid activation function needed for LSTM
     *
     */
    private ActivationFunction sigmoid = new ActivationFunction(ActivationFunctionType.SIGMOID);

    /**
     * Flag if tanh operation is performed also for last output function.
     *
     */
    private boolean doubleTanh = true;

    /**
     * Flag if state is reset prior start of next training sequence.
     *
     */
    private boolean resetStateTraining = true;

    /**
     * Flag if state is reset prior start of next test sequence.
     *
     */
    private boolean resetStateTesting = false;

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
     * Stores previous output of layer.
     *
     */
    private transient Matrix outPrev = null;

    /**
     * Stores previous state of layer.
     *
     */
    private transient Matrix cPrev = null;

    /**
     * Stores outputs of input gate for each sample.<br>
     * Used as cache for backpropagation phase.<br>
     *
     */
    private transient TreeMap<Integer, Matrix> iS;

    /**
     * Stores outputs of forget gate for each sample.<br>
     * Used as cache for backpropagation phase.<br>
     *
     */
    private transient TreeMap<Integer, Matrix> fS;

    /**
     * Stores outputs of output gate for each sample.<br>
     * Used as cache for backpropagation phase.<br>
     *
     */
    private transient TreeMap<Integer, Matrix> oS;

    /**
     * Stores outputs of state prior activation function for each sample.<br>
     * Used as cache for backpropagation phase.<br>
     *
     */
    private transient TreeMap<Integer, Matrix> s_S;

    /**
     * Stores outputs of state post activation function for each sample.<br>
     * Used as cache for backpropagation phase.<br>
     *
     */
    private transient TreeMap<Integer, Matrix> sS;

    /**
     * Stores outputs of internal cell state prior activation function for each sample.<br>
     * Used as cache for backpropagation phase.<br>
     *
     */
    private transient TreeMap<Integer, Matrix> c_S;

    /**
     * Stores outputs of internal cell state post activation function for each sample.<br>
     * Used as cache for backpropagation phase.<br>
     *
     */
    private transient TreeMap<Integer, Matrix> cS;

    /**
     * Constructor for LSTM layer.
     *
     * @param parent reference to parent layer.
     * @param activation activation function used. Not relevant for this layer.
     * @param initialization intialization function for weight.
     * @param params parameters for LSTM layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public LSTMLayer(AbstractLayer parent, ActivationFunction activation, Init initialization, String params) throws DynamicParamException {
        super (parent, activation, initialization, params);
    }

    /**
     * Gets parameters used for LSTM layer.
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
        if (params.hasParam("width")) setWidth(params.getValueAsInteger("width"));
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

        backward.registerWeight(bi, true, false, false);
        backward.registerWeight(bf, true, false, false);
        backward.registerWeight(bo, true, false, false);
        backward.registerWeight(bs, true, false, false);
    }

    /**
     * Takes single forward processing step for recurrent layer to process sequence.<br>
     * Executes LSTM forward step equations, stores intermediate values to cache and writes layer output.<br>
     * Additionally applies any regularization defined for the layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    public void forwardProcess() throws MatrixException, NeuralNetworkException {
        backward.regulateForwardPre(getOutsP(), -1);
        backward.normalizeForwardPre(getOutsP(), 1);

        iS = new TreeMap<>();
        fS = new TreeMap<>();
        oS = new TreeMap<>();
        s_S = new TreeMap<>();
        sS = new TreeMap<>();
        c_S = new TreeMap<>();
        cS = new TreeMap<>();

        parent.resetOuts();

        if (backward.isTraining() && resetStateTraining) {
            outPrev = null;
            cPrev = null;
        }
        if (!backward.isTraining() && resetStateTesting) {
            outPrev = null;
            cPrev = null;
        }

        for (Integer index : getOutsP().keySet()) {
            backward.regulateForwardPre(getOutsP(), index);

            Matrix in = getOutsP().get(index);

            // i = sigmoid(Wi * x + Ui * out(t-1) + bi) → Input gate
            Matrix i = outPrev == null ? Wi.dot(in).add(bi) : Wi.dot(in).add(Ui.dot(outPrev)).add(bi);
            i.apply(i, sigmoid.getFunction());
            iS.put(index, i);

            // f = sigmoid(Wf * x + Uf * out(t-1) + bf) → Forget gate
            Matrix f = outPrev == null ? Wf.dot(in).add(bf) : Wf.dot(in).add(Uf.dot(outPrev)).add(bf);
            f.apply(f, sigmoid.getFunction());
            fS.put(index, f);

            // o = sigmoid(Wo * x + Uo * out(t-1) + bo) → Output gate
            Matrix o = outPrev == null ? Wo.dot(in).add(bo) : Wo.dot(in).add(Uo.dot(outPrev)).add(bo);
            o.apply(o, sigmoid.getFunction());
            oS.put(index, o);

            // s = tanh(Ws * x + Us * out(t-1) + bs) → State update
            Matrix s = outPrev == null ? Ws.dot(in).add(bs) : Ws.dot(in).add(Us.dot(outPrev)).add(bs);
            s_S.put(index, s);
            s = s.apply(tanh.getFunction());
            sS.put(index, s);

            // c = i x s + f x c-1 → Internal cell state
//            if (index > 0) cPrev = cS.get(index - 1);
            Matrix c = (cPrev == null) ? i.multiply(s) : i.multiply(s).add(cPrev.multiply(f));
            cPrev = c;
            cS.put(index, c);

            // h = tanh(c) x o or h = c x o → Output
            Matrix c_ = doubleTanh ? c.apply(tanh.getFunction()) : c;
            c_S.put(index, c_);
            Matrix h = c_.multiply(o);
            parent.getOuts().put(index, h);

            outPrev = h;

        }

        backward.regulateForwardPost(parent.getOuts());
        backward.normalizeForwardPost(parent.getOuts());

        if (forward == null) parent.updateOutputError();
    }

    /**
     * Takes single backward processing step for LSTM layer to process sequence.<br>
     * Calculates gradients for weights and biases and gradient (error signal) towards previous layer.<br>
     * Additionally applies any regularization defined for the layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backwardProcess() throws MatrixException {
        parent.resetOutGrads();

        backward.resetGrad();

        backward.regulateBackward(-1);

        Matrix dhPrev = null;
        Matrix dcPrev = null;
        Matrix.MatrixUniOperation oneminusp2 = (value) -> 1 - Math.pow(value, 2);

        TreeMap<Integer, Matrix> dWi = backward.getdWs(Wi);
        TreeMap<Integer, Matrix> dWf = backward.getdWs(Wf);
        TreeMap<Integer, Matrix> dWo = backward.getdWs(Wo);
        TreeMap<Integer, Matrix> dWs = backward.getdWs(Ws);
        TreeMap<Integer, Matrix> dUi = backward.getdWs(Ui);
        TreeMap<Integer, Matrix> dUf = backward.getdWs(Uf);
        TreeMap<Integer, Matrix> dUo = backward.getdWs(Uo);
        TreeMap<Integer, Matrix> dUs = backward.getdWs(Us);
        TreeMap<Integer, Matrix> dbi = backward.getdWs(bi);
        TreeMap<Integer, Matrix> dbf = backward.getdWs(bf);
        TreeMap<Integer, Matrix> dbo = backward.getdWs(bo);
        TreeMap<Integer, Matrix> dbs = backward.getdWs(bs);
        for (Integer index : parent.getOuts().descendingKeySet()) {
            backward.regulateBackward(index);

            // dh = do + dhPrev
            Matrix dh = parent.getdEosN().get(index);
            if (dhPrev != null) dh.add(dhPrev);

            Matrix dou;
            // do = dh * tanh(c) * dsigmoid(o) -> tanh(c) = c_
            if (doubleTanh) dou = dh.multiply(c_S.get(index)).multiply(oS.get(index).apply(sigmoid.getDerivative()));
            // do = dh * c * dsigmoid(o)
            else dou = dh.multiply(cS.get(index)).multiply(oS.get(index).apply(sigmoid.getDerivative()));

            Matrix dc;
            // dc = dh * o * dtanh(c) + dcPrev -> dtanh(c) = 1 - c_^2
            if (doubleTanh) dc = dh.multiply(oS.get(index)).multiply(cS.get(index).apply(tanh.getDerivative()));
            // dc = dh * o + dcPrev
            else dc = dh.multiply(oS.get(index));
            if (dcPrev != null) dc.add(dcPrev);

            // ds = dc * i * dtanh(s)
            Matrix ds = dc.multiply(iS.get(index)).multiply(s_S.get(index).apply(tanh.getDerivative()));

            // df = dc * ct-1 * dsigmoid(f)
            Matrix df = null;
            if (index > 0) df = dc.multiply(cS.get(index -1)).multiply(fS.get(index).apply(sigmoid.getDerivative()));

            // di =  dc * s * dsigmoid(i)
            Matrix di = dc.multiply(sS.get(index)).multiply(iS.get(index).apply(sigmoid.getDerivative()));

            // Update weight deltas and previous layer delta
            dhPrev = null;

            Matrix outP = getOutsP().get(index).T();
            Matrix outH = index > 0 ? parent.getOuts().get(index - 1).T() : null;

            dWo.put(index, dou.dot(outP));
            dbo.put(index, dou);
            Matrix dEo = Wo.T().dot(dou);

            if (index > 0) {
                dUo.put(index, dou.dot(outH));
                dhPrev = Uo.T().dot(dou);
            }

            if (df != null) {
                dWf.put(index, df.dot(outP));
                dbf.put(index, df);
                dUf.put(index, df.dot(outH));
                dEo.add(Wf.T().dot(df), dEo);
                dhPrev.add(Uf.T().dot(df), dhPrev);
            }

            dWi.put(index, di.dot(outP));
            dbi.put(index, di);
            dEo.add(Wi.T().dot(di), dEo);

            if (index > 0) {
                dUi.put(index, di.dot(outH));
                dhPrev.add(Ui.T().dot(di), dhPrev);
            }

            dWs.put(index, ds.dot(outP));
            dbs.put(index, ds);
            dEo.add(Ws.T().dot(ds), dEo);

            if (index > 0) {
                dUs.put(index, ds.dot(outH));
                dhPrev.add(Us.T().dot(ds), dhPrev);
            }

            parent.getdEos().put(index, dEo);

            // dcPrev = dc * f
            dcPrev = dc.multiply(fS.get(index));

        }

        backward.normalizeBackward();

        backward.sumGrad();

    }

}