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
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public FeedforwardLayer(AbstractLayer parent, ActivationFunction activation, Init initialization, String params) throws NeuralNetworkException, DynamicParamException {
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
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @param input input of forward procedure.
     * @param reset reset recurring inputs of procedure.
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix getForwardProcedure(Matrix input, boolean reset) throws MatrixException {
        Matrix output = W.dot(input).add(B);
        output = output.apply(activation);

        return output;

    }

}
