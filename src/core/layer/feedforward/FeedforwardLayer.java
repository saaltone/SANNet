/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.layer.feedforward;

import core.NeuralNetworkException;
import core.activation.ActivationFunction;
import core.layer.AbstractExecutionLayer;
import core.layer.AbstractLayer;
import core.normalization.Normalization;
import utils.*;
import utils.matrix.DMatrix;
import utils.matrix.Init;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashMap;
import java.util.HashSet;

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
     * Input matrix for procedure construction.
     *
     */
    private Matrix input;

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
     * Returns parameters used for feed forward layer.
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
        if (params.hasParam("width")) parent.setWidth(params.getValueAsInteger("width"));
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
    public void initialize() throws MatrixException {
        int pLayerWidth = parent.getBackward().getPLayerWidth();
        int nLayerWidth = parent.getBackward().getNLayerWidth();

        W = new DMatrix(nLayerWidth, pLayerWidth, this.initialization);

        B = new DMatrix(nLayerWidth, 1);

        parent.getBackward().registerWeight(W, true, regulateDirectWeights, true);

        parent.getBackward().registerWeight(B, true, false, false);

    }

    /**
     * Resets input.
     *
     * @param resetPreviousInput if true resets also previous input.
     */
    protected void resetInput(boolean resetPreviousInput) throws MatrixException {
        input = new DMatrix(parent.getBackward().getPLayerWidth(), 1, Init.ONE);
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
        Matrix output = W.dot(input);

        if (normalizers.size() > 0) output.setNormalization(normalizers);

        output = output.add(B);

        output = output.apply(activation);

        Sample outputs = new Sample(1);
        outputs.put(0, output);
        return outputs;

    }

}
