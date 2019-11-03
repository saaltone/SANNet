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
import core.normalization.Normalization;
import utils.*;
import utils.matrix.DMatrix;
import utils.matrix.Init;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashMap;
import java.util.HashSet;

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
     * Input matrix for procedure construction.
     *
     */
    private Matrix input;

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
     * Returns parameters used for recurrent layer.
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
        if (params.hasParam("width")) parent.setWidth(params.getValueAsInteger("width"));
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
    public void initialize() throws MatrixException {
        int pLayerWidth = parent.getBackward().getPLayerWidth();
        int nLayerWidth = parent.getBackward().getNLayerWidth();

        W = new DMatrix(nLayerWidth, pLayerWidth, this.initialization);

        B = new DMatrix(nLayerWidth, 1);
        B.initialize(W.getInitializer());

        Wl = new DMatrix(nLayerWidth, nLayerWidth);
        Wl.init(this.initialization);

        parent.getBackward().registerWeight(W, true, regulateDirectWeights, true);

        parent.getBackward().registerWeight(Wl, true, regulateRecurrentWeights, true);

        parent.getBackward().registerWeight(B, true, false, false);

        truncateSteps = -1;

    }

    /**
     * Resets input.
     *
     * @param resetPreviousInput if true resets also previous input.
     */
    protected void resetInput(boolean resetPreviousInput) throws MatrixException {
        input = new DMatrix(parent.getBackward().getPLayerWidth(), 1, Init.ONE);
        if (resetPreviousInput) outPrev = new DMatrix(parent.getBackward().getNLayer().getWidth(), 1);
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

        Matrix output = W.dot(input).add(B).add(Wl.dot(outPrev));

        output = output.apply(activation);

        outPrev = output;

        Sample outputs = new Sample(1);
        outputs.put(0, output);
        return outputs;

    }

}
