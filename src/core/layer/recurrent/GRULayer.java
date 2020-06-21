/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.layer.recurrent;

import core.NeuralNetworkException;
import core.activation.ActivationFunction;
import utils.*;
import utils.matrix.*;

import java.util.HashMap;

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
public class GRULayer extends AbstractRecurrentLayer {

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
     * Ones matrix for calculation of z
     *
     */
    private Matrix ones;

    /**
     * Matrix to store previous output
     *
     */
    private Matrix previousOutput;

    /**
     * Tanh activation function needed for Graves LSTM
     *
     */
    private final ActivationFunction tanh;

    /**
     * Sigmoid activation function needed for Graves LSTM
     *
     */
    private final ActivationFunction sigmoid;

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
     * Input matrix for procedure construction.
     *
     */
    private Matrix input;

    /**
     * Constructor for GRU layer.
     *
     * @param layerIndex layer Index.
     * @param initialization initialization function for weight.
     * @param params parameters for GRU layer.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public GRULayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, params);
        setParams(new DynamicParam(params, getParamDefs()));
        tanh = new ActivationFunction(UnaryFunctionType.TANH);
        sigmoid = new ActivationFunction(UnaryFunctionType.SIGMOID);
    }

    /**
     * Returns parameters used for GRU layer.
     *
     * @return parameters used for GRU layer.
     */
    public HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>(super.getParamDefs());
        paramDefs.put("regulateDirectWeights", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("regulateRecurrentWeights", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for GRU layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value).<br>
     *
     * @param params parameters used for GRU layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("regulateDirectWeights")) regulateDirectWeights = params.getValueAsBoolean("regulateDirectWeights");
        if (params.hasParam("regulateRecurrentWeights")) regulateRecurrentWeights = params.getValueAsBoolean("regulateRecurrentWeights");
    }

    /**
     * Initializes GRU layer.<br>
     * Initializes weights and bias and their gradients.<br>
     *
     */
    public void initialize() {
        int previousLayerWidth = getPreviousLayerWidth();
        int layerWidth = getLayerWidth();

        Wz = new DMatrix(layerWidth, previousLayerWidth, initialization, "Wz");
        Wr = new DMatrix(layerWidth, previousLayerWidth, initialization, "Wr");
        Wh = new DMatrix(layerWidth, previousLayerWidth, initialization, "Wh");

        Uz = new DMatrix(layerWidth, layerWidth, initialization, "Uz");
        Ur = new DMatrix(layerWidth, layerWidth, initialization, "Ur");
        Uh = new DMatrix(layerWidth, layerWidth, initialization, "Uh");

        bz = new DMatrix(layerWidth, 1, "bz");
        br = new DMatrix(layerWidth, 1, "br");
        bh = new DMatrix(layerWidth, 1, "bh");

        registerWeight(Wz, regulateDirectWeights, true);
        registerWeight(Wr, regulateDirectWeights, true);
        registerWeight(Wh, regulateDirectWeights, true);

        registerWeight(Uz, regulateRecurrentWeights, true);
        registerWeight(Ur, regulateRecurrentWeights, true);
        registerWeight(Uh, regulateRecurrentWeights, true);

        registerWeight(bz, false, false);
        registerWeight(br, false, false);
        registerWeight(bh, false, false);

    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     */
    public MMatrix getInputMatrices(boolean resetPreviousInput) {
        input = new DMatrix(getPreviousLayerWidth(), 1, Initialization.ONE, "Input");
        if (resetPreviousInput) previousOutput = new DMatrix(getLayerWidth(), 1);
        return new MMatrix(input);
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public MMatrix getForwardProcedure() throws MatrixException {
        input.setNormalize(true);
        input.setRegularize(true);

        previousOutput.setName("PrevOutput");

        // z = sigmoid(Wz * x + Uz * out(t-1) + bz) → Update gate
        Matrix z = Wz.dot(input).add(Uz.dot(previousOutput)).add(bz);
        z = z.apply(sigmoid);
        z.setName("z");

        // r = sigmoid(Wr * x + Ur * out(t-1) + br) → Reset gate
        Matrix r = Wr.dot(input).add(Ur.dot(previousOutput)).add(br);
        r = r.apply(sigmoid);
        r.setName("r");

        // h = tanh(Wh * x + Uh * out(t-1) * r + bh) → Input activation
        Matrix h = Wh.dot(input).add(Uh.dot(previousOutput).multiply(r)).add(bh);
        h = h.apply(tanh);
        h.setName("h");

        // s = (1 - z) x h + z x out(t-1) → Internal state
        ones = (ones == null) ? new DMatrix(z.getRows(), z.getColumns(), Initialization.ONE) : ones;
        ones.setName("1");

        Matrix s = ones.subtract(z).multiply(h).add(z.multiply(previousOutput));
        s.setName("Output");

        previousOutput = s;

        MMatrix outputs = new MMatrix(1, "Output");
        outputs.put(0, s);
        return outputs;

    }

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        return null;
    }

}

