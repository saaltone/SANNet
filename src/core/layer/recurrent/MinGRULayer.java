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
public class MinGRULayer extends AbstractExecutionLayer {

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
    private Matrix outPrev;

    /**
     * Tanh activation function needed for GRU
     *
     */
    private ActivationFunction tanh;

    /**
     * Sigmoid activation function needed for GRU
     *
     */
    private ActivationFunction sigmoid;

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
     * Constructor for minimal GRU layer.
     *
     * @param parent reference to parent layer.
     * @param activation activation function used. Not relevant for this layer.
     * @param initialization intialization function for weight.
     * @param params parameters for minimal GRU layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public MinGRULayer(AbstractLayer parent, ActivationFunction activation, Init initialization, String params) throws NeuralNetworkException, DynamicParamException {
        super (parent, activation, initialization, params);
        tanh = new ActivationFunction(UniFunctionType.TANH);
        sigmoid = new ActivationFunction(UniFunctionType.SIGMOID);
    }

    /**
     * Gets parameters used for minimal GRU layer.
     *
     * @return parameters used for minimal GRU layer.
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
     * Sets parameters used for minimal GRU layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - width: width (number of nodes) of minimal GRU layer.<br>
     *     - resetStateTraining: true if output is reset prior training forward step start otherwise false (default value).<br>
     *     - resetStateTesting: true if output is reset prior test forward step start otherwise false (default value).<br>
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value).<br>
     *
     * @param params parameters used for minimal GRU layer.
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
     * Initializes minimal GRU layer.<br>
     * Initializes weights and bias and their gradients.<br>
     *
     */
    public void initialize() {
        int pLayerWidth = parent.getBackward().getPLayer().getWidth();
        int nLayerWidth = parent.getBackward().getNLayer().getWidth();

        Wf = new DMatrix(nLayerWidth, pLayerWidth);
        Wf.init(this.initialization);

        Wh = new DMatrix(nLayerWidth, pLayerWidth);
        Wh.init(this.initialization);

        Uf = new DMatrix(nLayerWidth, nLayerWidth);
        Uf.init(this.initialization);

        Uh = new DMatrix(nLayerWidth, nLayerWidth);
        Uh.init(this.initialization);

        bf = new DMatrix(nLayerWidth, 1);
        bf.init(this.initialization);

        bh = new DMatrix(nLayerWidth, 1);
        bh.init(this.initialization);

        backward.registerWeight(Wf, true, regulateDirectWeights, true);
        backward.registerWeight(Wh, true, regulateDirectWeights, true);

        backward.registerWeight(Uf, true, regulateRecurrentWeights, true);
        backward.registerWeight(Uh, true, regulateRecurrentWeights, true);

        backward.registerWeight(bf, true, false, false);
        backward.registerWeight(bh, true, false, false);
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @param input input of forward procedure.
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix getForwardProcedure(Matrix input, boolean reset) throws MatrixException {
        if (reset) outPrev = new DMatrix(parent.getBackward().getNLayer().getWidth(), 1);

        // f = sigmoid(Wf * x + Uf * out(t-1) + bf) → Forget gate
        Matrix f = Wf.dot(input).add(Uf.dot(outPrev)).add(bf);
        f = f.apply(sigmoid);

        // h = tanh(Wh * x + Uh * out(t-1) * r + bh) → Input activation
        Matrix h = Wh.dot(input).add(Uh.dot(outPrev).multiply(f)).add(bh);
        h = h.apply(tanh);

        // s = (1 - f) x h + f x out(t-1) → Internal state
        ones = (ones == null) ? new DMatrix(f.getRows(), f.getCols(), Init.ONE) : ones;
        Matrix s = ones.subtract(f).multiply(h).add(f.multiply(outPrev));

        outPrev = s;

        return s;

    }

}

