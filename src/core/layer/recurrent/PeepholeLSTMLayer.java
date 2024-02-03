/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.layer.recurrent;

import core.activation.ActivationFunctionType;
import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import core.activation.ActivationFunction;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashSet;
import java.util.TreeMap;

/**
 * Implements peephole Long Short Term Memory (LSTM)<br>
 * <br>
 * Reference: <a href="https://en.wikipedia.org/wiki/Long_short-term_memory">...</a><br>
 * <br>
 * Equations applied for forward operation:<br>
 *   i = sigmoid(Wi * x + Ui * c(t-1) + bi) → Input gate<br>
 *   f = sigmoid(Wf * x + Uf * c(t-1) + bf) → Forget gate<br>
 *   o = sigmoid(Wo * x + Uo * c(t-1) + bo) → Output gate<br>
 *   s = tanh(Ws * x + bs) → State update<br>
 *   c = i x s + f x c-1 → Internal cell state<br>
 *   h = tanh(c) x o or h = c x o → Output<br>
 *
 */
public class PeepholeLSTMLayer extends AbstractRecurrentLayer {

    /**
     * Parameter name types for peephole LSTM layer.
     *     - doubleTanh: true if tanh operation at final output step is executed otherwise false (default value true).<br>
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value true).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value false).<br>
     *
     */
    private final static String paramNameTypes = "(doubleTanh:BOOLEAN), " +
            "(regulateDirectWeights:BOOLEAN), " +
            "(regulateRecurrentWeights:BOOLEAN)";

    /**
     * Implements weight set for layer.
     *
     */
    protected class PeepholeLSTMWeightSet implements WeightSet, Serializable {

        @Serial
        private static final long serialVersionUID = -2306728084542274726L;

        /**
         * Weights for input gate
         *
         */
        private final Matrix Wi;

        /**
         * Weights for forget gate
         *
         */
        private final Matrix Wf;

        /**
         * Weights for output gate
         *
         */
        private final Matrix Wo;

        /**
         * Weights for state
         *
         */
        private final Matrix Ws;

        /**
         * Weights for recurrent input gate
         *
         */
        private final Matrix Ui;

        /**
         * Weights for recurrent forget gate
         *
         */
        private final Matrix Uf;

        /**
         * Weights for recurrent output gate
         *
         */
        private final Matrix Uo;

        /**
         * Bias for input gate
         *
         */
        private final Matrix bi;

        /**
         * Bias for forget gate
         *
         */
        private final Matrix bf;

        /**
         * Bias for output gate
         *
         */
        private final Matrix bo;

        /**
         * Bias for state
         *
         */
        private final Matrix bs;

        /**
         * Set of weights.
         *
         */
        private final HashSet<Matrix> weights = new HashSet<>();

        /**
         * Constructor for weight set
         *
         * @param initialization weight initialization function.
         * @param previousLayerWidth width of previous layer.
         * @param layerWidth width of current layer.
         * @param regulateDirectWeights if true direct weights are regulated.
         * @param regulateRecurrentWeights if true recurrent weight are regulated.
         */
        PeepholeLSTMWeightSet(Initialization initialization, int previousLayerWidth, int layerWidth, boolean regulateDirectWeights, boolean regulateRecurrentWeights) {
            Wi = new DMatrix(layerWidth, previousLayerWidth, 1, initialization);
            Wi.setName("Wi");
            Wf = new DMatrix(layerWidth, previousLayerWidth, 1, initialization);
            Wf.setName("Wf");
            Wo = new DMatrix(layerWidth, previousLayerWidth, 1, initialization);
            Wo.setName("Wo");
            Ws = new DMatrix(layerWidth, previousLayerWidth, 1, initialization);
            Ws.setName("Ws");

            Ui = new DMatrix(layerWidth, layerWidth, 1, initialization);
            Ui.setName("Ui");
            Uf = new DMatrix(layerWidth, layerWidth, 1, initialization);
            Uf.setName("Uf");
            Uo = new DMatrix(layerWidth, layerWidth, 1, initialization);
            Uo.setName("Uo");

            bi = new DMatrix(layerWidth, 1, 1);
            bi.setName("bi");
            bf = new DMatrix(layerWidth, 1, 1);
            bf.setName("bf");
            bo = new DMatrix(layerWidth, 1, 1);
            bo.setName("bo");
            bs = new DMatrix(layerWidth, 1, 1);
            bs.setName("bs");

            weights.add(Wi);
            weights.add(Wf);
            weights.add(Wo);
            weights.add(Ws);

            weights.add(Ui);
            weights.add(Uf);
            weights.add(Uo);

            weights.add(bi);
            weights.add(bf);
            weights.add(bo);
            weights.add(bs);

            registerWeight(Wi, regulateDirectWeights, true);
            registerWeight(Wf, regulateDirectWeights, true);
            registerWeight(Wo, regulateDirectWeights, true);
            registerWeight(Ws, regulateDirectWeights, true);

            registerWeight(Ui, regulateRecurrentWeights, true);
            registerWeight(Uf, regulateRecurrentWeights, true);
            registerWeight(Uo, regulateRecurrentWeights, true);

            registerWeight(bi, false, false);
            registerWeight(bf, false, false);
            registerWeight(bo, false, false);
            registerWeight(bs, false, false);
        }

        /**
         * Returns set of weights.
         *
         * @return set of weights.
         */
        public HashSet<Matrix> getWeights() {
            return weights;
        }

        /**
         * Reinitializes weights.
         *
         */
        public void reinitialize() {
            Wi.initialize(initialization);
            Wf.initialize(initialization);
            Wo.initialize(initialization);
            Ws.initialize(initialization);

            Ui.initialize(initialization);
            Uf.initialize(initialization);
            Uo.initialize(initialization);

            bi.reset();
            bf.reset();
            bo.reset();
            bs.reset();
        }

        /**
         * Returns number of parameters.
         *
         * @return number of parameters.
         */
        public int getNumberOfParameters() {
            int numberOfParameters = 0;
            for (Matrix weight : weights) numberOfParameters += weight.size();
            return numberOfParameters;
        }

    }

    /**
     * Weight set.
     *
     */
    protected PeepholeLSTMWeightSet weightSet;

    /**
     * Current weight set.
     *
     */
    protected PeepholeLSTMWeightSet currentWeightSet;

    /**
     * Matrix to store previous state.
     *
     */
    private Matrix previousCellState;

    /**
     * Tanh activation function needed for Peephole LSTM
     *
     */
    private final ActivationFunction tanh;

    /**
     * Sigmoid activation function needed for Peephole LSTM
     *
     */
    private final ActivationFunction sigmoid;

    /**
     * Activation function for output
     *
     */
    private final ActivationFunction activationFunction;

    /**
     * Flag if tanh operation is performed also for last output function.
     *
     */
    private boolean doubleTanh;

    /**
     * Flag if direct (non-recurrent) weights are regulated.
     *
     */
    private boolean regulateDirectWeights;

    /**
     * Flag if recurrent weights are regulated.
     *
     */
    private boolean regulateRecurrentWeights;

    /**
     * Input matrix for procedure construction.
     *
     */
    private Matrix input;

    /**
     * Constructor for peephole LSTM layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for peephole LSTM layer.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    protected PeepholeLSTMLayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, params);
        tanh = new ActivationFunction(ActivationFunctionType.TANH);
        sigmoid = new ActivationFunction(ActivationFunctionType.SIGMOID);
        activationFunction = tanh;
    }

    /**
     * Constructor for peephole LSTM layer.
     *
     * @param layerIndex layer index
     * @param activationFunction activation function used.
     * @param initialization initialization function for weight.
     * @param params parameters for peephole LSTM layer.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public PeepholeLSTMLayer(int layerIndex, ActivationFunction activationFunction, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, params);
        tanh = new ActivationFunction(ActivationFunctionType.TANH);
        sigmoid = new ActivationFunction(ActivationFunctionType.SIGMOID);
        this.activationFunction = activationFunction == null ? tanh : activationFunction;
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        doubleTanh = true;
        regulateDirectWeights = true;
        regulateRecurrentWeights = false;
    }

    /**
     * Returns parameters used for peephole LSTM layer.
     *
     * @return parameters used for peephole LSTM layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + PeepholeLSTMLayer.paramNameTypes;
    }

    /**
     * Sets parameters used for peephole LSTM layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - doubleTanh: true if tanh operation at final output step is executed otherwise false (default value true).<br>
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value true).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value false).<br>
     *
     * @param params parameters used for peephole LSTM layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("doubleTanh")) doubleTanh = params.getValueAsBoolean("doubleTanh");
        if (params.hasParam("regulateDirectWeights")) regulateDirectWeights = params.getValueAsBoolean("regulateDirectWeights");
        if (params.hasParam("regulateRecurrentWeights")) regulateRecurrentWeights = params.getValueAsBoolean("regulateRecurrentWeights");
    }

    /**
     * Returns weight set.
     *
     * @return weight set.
     */
    protected WeightSet getWeightSet() {
        return weightSet;
    }

    /**
     * Initializes neural network layer weights.
     *
     */
    public void initializeWeights() {
        currentWeightSet = weightSet = new PeepholeLSTMWeightSet(initialization, getDefaultPreviousLayer().getLayerWidth(), getLayerWidth(), regulateDirectWeights, regulateRecurrentWeights);
    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     */
    public TreeMap<Integer, Matrix> getInputMatrices(boolean resetPreviousInput) {
        input = new DMatrix(getDefaultPreviousLayer().getLayerWidth(), 1, 1, Initialization.ONE);
        input.setName("Input" + getDefaultPreviousLayer().getLayerIndex());
        if (resetPreviousInput) {
            previousCellState = new DMatrix(getLayerWidth(), 1, 1);
        }
        return new TreeMap<>() {{ put(0, input); }};
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix getForwardProcedure() throws MatrixException {
        previousCellState.setName("PrevCellState");

        // i = sigmoid(Wi * x + Ui * c(t-1) + bi) → Input gate
        Matrix i = currentWeightSet.Wi.dot(input).add(currentWeightSet.Ui.dot(previousCellState)).add(currentWeightSet.bi);
        i = i.apply(sigmoid);
        i.setName("i");

        // f = sigmoid(Wf * x + Uf * c(t-1) + bf) → Forget gate
        Matrix f = currentWeightSet.Wf.dot(input).add(currentWeightSet.Uf.dot(previousCellState)).add(currentWeightSet.bf);
        f = f.apply(sigmoid);
        f.setName("f");

        // o = sigmoid(Wo * x + Uo * c(t-1) + bo) → Output gate
        Matrix o = currentWeightSet.Wo.dot(input).add(currentWeightSet.Uo.dot(previousCellState)).add(currentWeightSet.bo);
        o = o.apply(sigmoid);
        o.setName("o");

        // s = tanh(Ws * x + bs) → State update
        Matrix s = currentWeightSet.Ws.dot(input).add(currentWeightSet.bs);
        s = s.apply(tanh);
        s.setName("s");

        // c = i x s + f x c-1 → Internal cell state
        Matrix c = i.multiply(s).add(previousCellState.multiply(f));
        c.setName("c");

        previousCellState = c;

        // h = activationFunction(c) x o or h = c x o → Output
        Matrix h = (doubleTanh ? c.apply(activationFunction) : c).multiply(o);
        h.setName("Output");

        return h;

    }

}
