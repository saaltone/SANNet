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
 * Implements gated recurrent unit (GRU) layer.<br>
 * <br>
 * Reference: https://en.wikipedia.org/wiki/Gated_recurrent_unit and https://github.com/erikvdplas/gru-rnn<br>
 * <br>
 * Equations applied for forward operation:<br>
 *     z = sigmoid(Wz * x + Uz * out(t-1) + bz) → Update gate<br>
 *     r = sigmoid(Wr * x + Ur * out(t-1) + br) → Reset gate<br>
 *     h = tanh(Wh * x + Uh * out(t-1) * r + bh) → Input activation<br>
 *     s = (1 - z) x h + z x out(t-1) → Internal state<br>
 *
 */
public class GRULayer extends AbstractExecutionLayer {

    /**
     * Weights for update gate
     *
     */
    private Matrix Wz;

    /**
     * Weights for reset gate
     *
     */
    private Matrix Wr;

    /**
     * Weights for input activation
     *
     */
    private Matrix Wh;

    /**
     * Weights for recurrent update gate
     *
     */
    private Matrix Uz;

    /**
     * Weights for recurrent reset gate
     *
     */
    private Matrix Ur;

    /**
     * Weights for current input activation
     *
     */
    private Matrix Uh;

    /**
     * Bias for update gate
     *
     */
    private Matrix bz;

    /**
     * Bias for reset gate
     *
     */
    private Matrix br;

    /**
     * Bias for input activation
     *
     */
    private Matrix bh;

    /**
     * Tanh activation function needed for GRU
     *
     */
    private final ActivationFunction tanh = new ActivationFunction(ActivationFunctionType.TANH);

    /**
     * Sigmoid activation function needed for GRU
     *
     */
    private final ActivationFunction sigmoid = new ActivationFunction(ActivationFunctionType.SIGMOID);

    /**
     * Stores previous output of layer.
     *
     */
    private transient Matrix outPrev = null;

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
     * Stores outputs of update gate for each sample.<br>
     * Used as cache for backpropagation phase.<br>
     *
     */
    private transient TreeMap<Integer, Matrix> zS;

    /**
     * Stores outputs of reset gate for each sample.<br>
     * Used as cache for backpropagation phase.<br>
     *
     */
    private transient TreeMap<Integer, Matrix> rS;

    /**
     * Stores outputs of input activation for each sample.<br>
     * Used as cache for backpropagation phase.<br>
     *
     */
    private transient TreeMap<Integer, Matrix> hS;

    /**
     * Constructor for GRU layer.
     *
     * @param parent reference to parent layer.
     * @param activation activation function used. Not relevant for this layer.
     * @param initialization intialization function for weight.
     * @param params parameters for GRU layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public GRULayer(AbstractLayer parent, ActivationFunction activation, Init initialization, String params) throws DynamicParamException {
        super (parent, activation, initialization, params);
    }

    /**
     * Gets parameters used for GRU layer.
     *
     * @return parameters used for GRU layer.
     */
    public HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("width", DynamicParam.ParamType.INT);
        paramDefs.put("resetStateTraining", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("resetStateTesting", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("regulateDirectWeights", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("regulateRecurrentWeights", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for GRU layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - width: width (number of nodes) of GRU layer.<br>
     *     - resetStateTraining: true if output is reset prior training forward step start otherwise false (default value).<br>
     *     - resetStateTesting: true if output is reset prior test forward step start otherwise false (default value).<br>
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value).<br>
     *
     * @param params parameters used for GRU layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("width")) setWidth(params.getValueAsInteger("width"));
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
     * Initializes GRU layer.<br>
     * Initializes weights and bias and their gradients.<br>
     *
     */
    public void initialize() {
        int pLayerWidth = parent.getBackward().getPLayer().getWidth();
        int nLayerWidth = parent.getBackward().getNLayer().getWidth();

        Wz = new DMatrix(nLayerWidth, pLayerWidth);
        Wz.init(this.initialization);

        Wr = new DMatrix(nLayerWidth, pLayerWidth);
        Wr.init(this.initialization);

        Wh = new DMatrix(nLayerWidth, pLayerWidth);
        Wh.init(this.initialization);

        Uz = new DMatrix(nLayerWidth, nLayerWidth);
        Uz.init(this.initialization);

        Ur = new DMatrix(nLayerWidth, nLayerWidth);
        Ur.init(this.initialization);

        Uh = new DMatrix(nLayerWidth, nLayerWidth);
        Uh.init(this.initialization);

        bz = new DMatrix(nLayerWidth, 1);
        bz.init(this.initialization);

        br = new DMatrix(nLayerWidth, 1);
        br.init(this.initialization);

        bh = new DMatrix(nLayerWidth, 1);
        bh.init(this.initialization);

        backward.registerWeight(Wz, true, regulateDirectWeights, true);
        backward.registerWeight(Wr, true, regulateDirectWeights, true);
        backward.registerWeight(Wh, true, regulateDirectWeights, true);

        backward.registerWeight(Uz, true, regulateRecurrentWeights, true);
        backward.registerWeight(Ur, true, regulateRecurrentWeights, true);
        backward.registerWeight(Uh, true, regulateRecurrentWeights, true);

        backward.registerWeight(bz, true, false, false);
        backward.registerWeight(br, true, false, false);
        backward.registerWeight(bh, true, false, false);
    }

    /**
     * Takes single forward processing step for GRU layer to process sequence.<br>
     * Executes GRU forward step equations, stores intermediate values to cache and writes layer output.<br>
     * Additionally applies any regularization defined for the layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    public void forwardProcess() throws MatrixException, NeuralNetworkException {
        backward.regulateForwardPre(getOutsP(), -1);
        backward.normalizeForwardPre(getOutsP(), 1);

        zS = new TreeMap<>();
        rS = new TreeMap<>();
        hS = new TreeMap<>();

        parent.resetOuts();

        if (backward.isTraining() && resetStateTraining) outPrev = null;
        if (!backward.isTraining() && resetStateTesting) outPrev = null;

        for (Integer index : getOutsP().keySet()) {
            backward.regulateForwardPre(getOutsP(), index);

            Matrix in = getOutsP().get(index);

            // z = sigmoid(Wz * x + Uz * out(t-1) + bz) → Update gate
            Matrix z = outPrev == null ? Wz.dot(in).add(bz) : Wz.dot(in).add(Uz.dot(outPrev)).add(bz);
            z.apply(z, sigmoid.getFunction());
            zS.put(index, z);

            // r = sigmoid(Wr * x + Ur * out(t-1) + br) → Reset gate
            Matrix r = outPrev == null ? Wr.dot(in).add(br) : Wr.dot(in).add(Ur.dot(outPrev)).add(br);
            r.apply(r, sigmoid.getFunction());
            rS.put(index, r);

            // h = tanh(Wh * x + Uh * out(t-1) * r + bh) → Input activation
            Matrix h = outPrev == null ? Wh.dot(in).add(bh) : Wh.dot(in).add(Uh.dot(outPrev).multiply(r)).add(bh);
            h.apply(h, tanh.getFunction());
            hS.put(index, h);

            // s = (1 - z) x h + z x out(t-1) → Internal state
            Matrix.MatrixUniOperation oneneg = (value) -> 1 - value;
            Matrix s = outPrev == null ? z.apply(oneneg).multiply(h) : z.apply(oneneg).multiply(h).add(z.multiply(outPrev));
            parent.getOuts().put(index, s);

            outPrev = s;

        }

        backward.regulateForwardPost(parent.getOuts());
        backward.normalizeForwardPost(parent.getOuts());

        if (forward == null) parent.updateOutputError();
    }

    /**
     * Takes single backward processing step for GRU layer to process sequence.<br>
     * Calculates gradients for weights and biases and gradient (error signal) towards previous layer.<br>
     * Additionally applies any regularization defined for the layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backwardProcess() throws MatrixException {
        parent.resetOutGrads();

        backward.resetGrad();

        backward.regulateBackward(-1);

        Matrix dsPrev = null;

        TreeMap<Integer, Matrix> dWz = backward.getdWs(Wz);
        TreeMap<Integer, Matrix> dWr = backward.getdWs(Wr);
        TreeMap<Integer, Matrix> dWh = backward.getdWs(Wh);
        TreeMap<Integer, Matrix> dUz = backward.getdWs(Uz);
        TreeMap<Integer, Matrix> dUr = backward.getdWs(Ur);
        TreeMap<Integer, Matrix> dUh = backward.getdWs(Uh);
        TreeMap<Integer, Matrix> dbz = backward.getdWs(bz);
        TreeMap<Integer, Matrix> dbr = backward.getdWs(br);
        TreeMap<Integer, Matrix> dbh = backward.getdWs(bh);
        for (Integer index : parent.getOuts().descendingKeySet()) {
            backward.regulateBackward(index);

            // ds = do + dsPrev
            Matrix ds = parent.getdEosN().get(index);
            if (dsPrev != null) ds.add(dsPrev);

            // dh = ds * (1 - z) * dtanh(h)
            Matrix.MatrixUniOperation oneminus = (value) -> 1 - value;
            Matrix dh = ds.multiply(zS.get(index).apply(oneminus)).multiply(hS.get(index).apply(tanh.getDerivative()));

            // dr = UhT x dh * st-1 * dsigmoid(r)
            Matrix dr = null;
            Matrix drp = Uh.T().dot(dh);
            if (index > 0) dr = drp.multiply(parent.getOuts().get(index - 1)).multiply(rS.get(index).apply(sigmoid.getDerivative()));

            // dz = ds * (st-1 - h) * dsigmoid(z);
            Matrix dz = null;
            if (index > 0) dz = ds.multiply(parent.getOuts().get(index - 1).subtract(hS.get(index))).multiply(zS.get(index).apply(sigmoid.getDerivative()));

            Matrix dEo = null;
            dsPrev = null;

            // Update weight deltas and previous layer delta
            if (dr != null) {
                dWr.put(index, dr.dot(getOutsP().get(index).T()));
                dbr.put(index, dr);
                dEo = Wr.T().dot(dr);

                dUr.put(index, dr.dot(parent.getOuts().get(index - 1).T()));
                dsPrev = Ur.T().dot(dr);
            }

            if (dz != null) {
                dWz.put(index, dz.dot(getOutsP().get(index).T()));
                dbz.put(index, dz);
                if (dEo == null) dEo = Wz.T().dot(dz);
                else dEo.add(Wz.T().dot(dz), dEo);

                dUz.put(index, dz.dot(parent.getOuts().get(index - 1).T()));
                if (dsPrev == null) dsPrev = Uz.T().dot(dz).add(ds.multiply(zS.get(index)));
                else dsPrev.add(Uz.T().dot(dz).add(ds.multiply(zS.get(index))), dsPrev);
            }

            dWh.put(index, dh.dot(getOutsP().get(index).T()));
            dbh.put(index, dh);
            if (dEo == null) dEo = Wh.T().dot(dh);
            else dEo.add(Wh.T().dot(dh), dEo);

            if (index > 0) {
                dUh.put(index, dh.dot(parent.getOuts().get(index - 1).multiply(rS.get(index)).T()));
                if (dsPrev == null) dsPrev = drp.multiply(rS.get(index));
                else dsPrev.add(drp.multiply(rS.get(index)), dsPrev);
            }

            parent.getdEos().put(index, dEo);

        }

        backward.normalizeBackward();

        backward.sumGrad();

    }

}
