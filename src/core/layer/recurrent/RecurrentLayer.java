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
 * Implements basic simple recurrent layer.<br>
 * This layer is by nature prone to numerical unstability with limited temporal memory.<br>
 *
 */
public class RecurrentLayer extends AbstractExecutionLayer {

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
     * Matrix to store previous output.
     *
     */
    private Matrix outPrev;

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
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public RecurrentLayer(AbstractLayer parent, ActivationFunction activation, Init initialization, String params) throws NeuralNetworkException, DynamicParamException {
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
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @param input input of forward procedure.
     * @param reset reset recurring inputs of procedure.
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix getForwardProcedure(Matrix input, boolean reset) throws MatrixException {
        if (reset) outPrev = new DMatrix(parent.getBackward().getNLayer().getWidth(), 1);

        Matrix output = W.dot(input).add(B).add(Wl.dot(outPrev));
        output = output.apply(activation);

        outPrev = output;

        return output;

    }

}
