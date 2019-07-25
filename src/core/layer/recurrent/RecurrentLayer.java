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
import java.util.TreeMap;

/**
 * Implements basic simple recurrent layer.<br>
 * This layer is by nature prone to numerical unstability with limited temporal memory.<br>
 *
 */
public class RecurrentLayer extends AbstractExecutionLayer {

    /**
     * Limits number of backward propagation sequence steps.
     *
     */
    private int truncateSteps;

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
     * Weight matrix for recurrent input.
     *
     */
    private Matrix Wl;

    /**
     * Stores output of previous sequence step.
     *
     */
    private transient Matrix outPrev = null;

    /**
     * If true does not use output of previous sequence when training.
     *
     */
    private boolean resetStateTraining = true;

    /**
     * If true does not use output of previous sequence when testing.
     *
     */
    private boolean resetStateTesting = false;

    /**
     * If true regulates direct feedforward weights (W).
     *
     */
    private boolean regulateDirectWeights = true;

    /**
     * If true regulates recurrent input weights (Wl).
     *
     */
    private boolean regulateRecurrentWeights = false;

    /**
     * Constructor for recurrent layer.
     *
     * @param parent reference to parent layer.
     * @param activation activation function used.
     * @param initialization intialization function for weight.
     * @param params parameters for recurrent layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public RecurrentLayer(AbstractLayer parent, ActivationFunction activation, Init initialization, String params) throws DynamicParamException {
        super (parent, activation, initialization, params);
    }

    /**
     * Gets parameters used for recurrent layer.
     *
     * @return return parameters used for recurrent layer.
     */
    public HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("width", DynamicParam.ParamType.INT);
        paramDefs.put("truncateSteps", DynamicParam.ParamType.INT);
        paramDefs.put("resetStateTraining", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("resetStateTesting", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("regulateDirectWeights", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("regulateRecurrentWeights", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for recurrent layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - width: width (number of nodes) of recurrent layer.<br>
     *     - truncateSteps: number of sequence steps taken in backpropagation phase.<br>
     *     - resetStateTraining: true if output is reset prior training forward step start otherwise false (default value).<br>
     *     - resetStateTesting: true if output is reset prior test forward step start otherwise false (default value).<br>
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value).<br>
     *
     * @param params parameters used for recurrent layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("width")) setWidth(params.getValueAsInteger("width"));
        if (params.hasParam("truncateSteps")) truncateSteps = params.getValueAsInteger("truncateSteps");
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
     * Check if layer is convolutional layer type.
     *
     * @return always false.
     */
    public boolean isConvolutionalLayer() { return false; }

    /**
     * Initializes recurrent layer.<br>
     * Initializes weights and bias and their gradients.<br>
     *
     */
    public void initialize() {
        int pLayerWidth = parent.getBackward().getPLayer().getWidth();
        int nLayerWidth = parent.getBackward().getNLayer().getWidth();

        W = new DMatrix(nLayerWidth, pLayerWidth, this.initialization);
        B = new DMatrix(nLayerWidth, 1);
        B.initialize(W.getInit());

        Wl = new DMatrix(nLayerWidth, nLayerWidth);
        Wl.init(this.initialization);

        backward.registerWeight(W, true, regulateDirectWeights, true);

        backward.registerWeight(Wl, true, regulateRecurrentWeights, true);

        backward.registerWeight(B, true, false, false);

        truncateSteps = -1;
    }

    /**
     * Takes single forward processing step for recurrent layer to process sequence.<br>
     * Applies weight dot product to input and adds bias and recurrent input finally applying activation function (non-linearity).<br>
     * Additionally applies any regularization defined for the layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    public void forwardProcess() throws MatrixException, NeuralNetworkException {
        backward.regulateForwardPre(getOutsP(), -1);
        backward.normalizeForwardPre(getOutsP(), 1);

        if (backward.isTraining() && resetStateTraining) outPrev = null;
        if (!backward.isTraining() && resetStateTesting) outPrev = null;

        parent.resetOuts();

        for (Integer index : getOutsP().keySet()) {
            backward.regulateForwardPre(getOutsP(), index);

            Matrix output = W.dot(getOutsP().get(index)).add(B);
            if (outPrev != null) output.add(Wl.dot(outPrev), output);
            parent.getOuts().put(index, output);
            outPrev = output;

            applyActivationFunction(output);
        }

        backward.regulateForwardPost(parent.getOuts());
        backward.normalizeForwardPost(parent.getOuts());

        if (forward == null) parent.updateOutputError();
    }

    /**
     * Takes single backward processing step for recurrent layer to process sequence.<br>
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

        TreeMap<Integer, Matrix> dW = backward.getdWs(W);
        TreeMap<Integer, Matrix> dWl = backward.getdWs(Wl);
        TreeMap<Integer, Matrix> dB = backward.getdWs(B);
        int errorStep = parent.getdEosN().size() - 1;
        errorStep = Math.min(errorStep, (truncateSteps < 0 ? Integer.MAX_VALUE : truncateSteps - 1));
        for (Integer index : parent.getOuts().descendingKeySet()) {
            backward.regulateBackward(index);

            Matrix out = parent.getOuts().get(index);
            Matrix dEo = parent.getdEosN().get(index);
            dhPrev = (dhPrev == null) ? dEo : dhPrev.add(dEo);
            Matrix dEi = getdEi(out, dhPrev);

            parent.getdEos().put(index, W.T().dot(dEi));

            dW.put(index, dEi.dot(getOutsP().get(index).T()));
            dB.put(index, dEi);

            dhPrev = null;
            if (index > 0) {
                dWl.put(index, dEi.dot(parent.getOuts().get(index - 1).T()));
                dhPrev = Wl.T().dot(dEi);
            }

            if(--errorStep < 0) break;
        }

        backward.normalizeBackward();

        backward.sumGrad();

    }

}

