/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.layer.feedforward;

import core.NeuralNetworkException;
import core.activation.ActivationFunction;
import core.layer.AbstractExecutionLayer;
import core.layer.AbstractLayer;
import utils.*;

import java.util.HashMap;
import java.util.TreeMap;

/**
 * Implements non-recurrent feed forward layer.
 *
 */
public class FeedforwardLayer extends AbstractExecutionLayer {

    /**
     * Weight matrix.
     *
     */
    private Matrix W;

    /**
     * Bias matrix.
     *
     */
    private Matrix B;

    /**
     * True if weights are regulated otherwise weights are not regulated.
     *
     */
    private boolean regulateDirectWeights = true;

    /**
     * Constructor for feed forward layer.
     *
     * @param parent reference to parent layer.
     * @param activation activation function used.
     * @param initialization intialization function for weight.
     * @param params parameters for feed forward layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public FeedforwardLayer(AbstractLayer parent, ActivationFunction activation, Init initialization, String params) throws DynamicParamException {
        super (parent, activation, initialization, params);
    }

    /**
     * Gets parameters used for feed forward layer.
     *
     * @return parameters used for feed forward layer.
     */
    public HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("width", DynamicParam.ParamType.INT);
        paramDefs.put("regulateDirectWeights", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for feed forward layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - width: width (number of nodes) of feed forward layer.<br>
     *     - regulateDirectWeights: true if (direct) weights are regulated otherwise false (default value).<br>
     *
     * @param params parameters used for feed forward layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("width")) setWidth(params.getValueAsInteger("width"));
        if (params.hasParam("regulateDirectWeights")) regulateDirectWeights = params.getValueAsBoolean("regulateDirectWeights");
    }

    /**
     * Checks if layer is recurrent layer type.
     *
     * @return always false.
     */
    public boolean isRecurrentLayer() { return false; }

    /**
     * Checks if layer is convolutional layer type.
     *
     * @return always false.
     */
    public boolean isConvolutionalLayer() { return false; }

    /**
     * Initializes feed forward layer.<br>
     * Initializes weight and bias and their gradients.<br>
     *
     */
    public void initialize() {
        int pLayerWidth = parent.getBackward().getPLayer().getWidth();
        int nLayerWidth = parent.getBackward().getNLayer().getWidth();

        W = new DMatrix(nLayerWidth, pLayerWidth, this.initialization);

        B = new DMatrix(nLayerWidth, 1);
        B.initialize(W.getInit());

        backward.registerWeight(W, true, regulateDirectWeights, true);

        backward.registerWeight(B, true, false, false);
    }

    /**
     * Takes single forward processing step for feed forward layer to process input(s).<br>
     * Applies weight dot product to input and adds bias finally applying activation function (non-linearity).<br>
     * Additionally applies any regularization defined for the layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    public void forwardProcess() throws MatrixException, NeuralNetworkException {
        backward.regulateForwardPre(getOutsP(), -1);
        backward.normalizeForwardPre(getOutsP(), 1);

        parent.resetOuts();

        for (Integer index : getOutsP().keySet()) {
            backward.regulateForwardPre(getOutsP(), index);

            Matrix output = W.dot(getOutsP().get(index)).add(B);
            parent.getOuts().put(index, output);

        }

        applyActivationFunction(parent.getOuts());

        backward.regulateForwardPost(parent.getOuts());
        backward.normalizeForwardPost(parent.getOuts());

        if (forward == null) parent.updateOutputError();
    }

    /**
     * Takes single backward processing step for feed forward layer to process input(s).<br>
     * Calculates gradients for weights and biases and gradient (error signal) towards previous layer.<br>
     * Additionally applies any regularization defined for the layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backwardProcess() throws MatrixException {
        parent.resetOutGrads();

        backward.resetGrad();

        backward.regulateBackward(-1);

        TreeMap<Integer, Matrix> dW = backward.getdWs(W);
        TreeMap<Integer, Matrix> dB = backward.getdWs(B);
        for (Integer index : parent.getOuts().descendingKeySet()) {
            backward.regulateBackward(index);

            Matrix out = parent.getOuts().get(index);
            Matrix dEo = parent.getdEosN().get(index);
            Matrix dEi = getdEi(out, dEo);

            parent.getdEos().put(index, W.T().dot(dEi));

            dW.put(index, dEi.dot(getOutsP().get(index).T()));
            dB.put(index, dEi);

        }

        backward.normalizeBackward();

        backward.sumGrad();

    }

}

