/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.layer.recurrent;

import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import core.activation.ActivationFunction;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashSet;

/**
 * Implements Graves type of Long Short Term Memory (LSTM)<br>
 * <br>
 * Reference: https://www.cs.toronto.edu/~graves/phd.pdf<br>
 * <br>
 * Equations applied for forward operation:<br>
 *   i = sigmoid(Wi * x + Ui * out(t-1) + Ci * c(t-1) + bi) → Input gate<br>
 *   f = sigmoid(Wf * x + Uf * out(t-1) + Cf * c(t-1) + bf) → Forget gate<br>
 *   s = tanh(Ws * x + Us * out(t-1) + bs) → State update<br>
 *   c = i x s + f x c(t-1) → Internal cell state<br>
 *   o = sigmoid(Wo * x + Uo * out(t-1) + Co * ct + bo) → Output gate<br>
 *   h = tanh(c) x o or h = c x o → Output<br>
 *
 */
public class GravesLSTMLayer extends AbstractRecurrentLayer {

    /**
     * Parameter name types for Graves LSTM layer.
     *     - doubleTanh: true if tanh operation at final output step is executed otherwise false (default value true).<br>
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value true).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value false).<br>
     *     - regulateStateWeights: true if recurrent state weights are regulated otherwise false (default value false).<br>
     *
     */
    private final static String paramNameTypes = "(doubleTanh:BOOLEAN), " +
            "(regulateDirectWeights:BOOLEAN), " +
            "(regulateRecurrentWeights:BOOLEAN), " +
            "(regulateStateWeights:BOOLEAN)";

    /**
     * Implements weight set for layer.
     *
     */
    protected class GravesLSTMWeightSet implements WeightSet, Serializable {

        @Serial
        private static final long serialVersionUID = 5184512014825168522L;

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
         * Weights for recurrent state
         *
         */
        private final Matrix Us;

        /**
         * Weights for input cell state
         *
         */
        private final Matrix Ci;

        /**
         * Weights for forget cell state
         *
         */
        private final Matrix Cf;

        /**
         * Weights for output cell state
         *
         */
        private final Matrix Co;

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
        GravesLSTMWeightSet(Initialization initialization, int previousLayerWidth, int layerWidth, boolean regulateDirectWeights, boolean regulateRecurrentWeights) {
            Wi = new DMatrix(layerWidth, previousLayerWidth, initialization);
            Wi.setName("Wi");
            Wf = new DMatrix(layerWidth, previousLayerWidth, initialization);
            Wf.setName("Wf");
            Wo = new DMatrix(layerWidth, previousLayerWidth, initialization);
            Wo.setName("Wo");
            Ws = new DMatrix(layerWidth, previousLayerWidth, initialization);
            Ws.setName("Ws");

            Ui = new DMatrix(layerWidth, layerWidth, initialization);
            Ui.setName("Ui");
            Uf = new DMatrix(layerWidth, layerWidth, initialization);
            Uf.setName("Uf");
            Uo = new DMatrix(layerWidth, layerWidth, initialization);
            Uo.setName("Uo");
            Us = new DMatrix(layerWidth, layerWidth, initialization);
            Us.setName("Us");

            Ci = new DMatrix(layerWidth, 1, initialization);
            Ci.setName("Ci");
            Cf = new DMatrix(layerWidth, 1, initialization);
            Cf.setName("Cf");
            Co = new DMatrix(layerWidth, 1, initialization);
            Co.setName("Co");

            bi = new DMatrix(layerWidth, 1);
            bi.setName("bi");
            bf = new DMatrix(layerWidth, 1);
            bf.setName("bf");
            bo = new DMatrix(layerWidth, 1);
            bo.setName("bo");
            bs = new DMatrix(layerWidth, 1);
            bs.setName("bs");

            weights.add(Wi);
            weights.add(Wf);
            weights.add(Wo);
            weights.add(Ws);

            weights.add(Ui);
            weights.add(Uf);
            weights.add(Uo);
            weights.add(Us);

            weights.add(Ci);
            weights.add(Cf);
            weights.add(Co);

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
            registerWeight(Us, regulateRecurrentWeights, true);

            registerWeight(Ci, regulateStateWeights, true);
            registerWeight(Cf, regulateStateWeights, true);
            registerWeight(Co, regulateStateWeights, true);

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
            Us.initialize(initialization);

            Ci.initialize(initialization);
            Cf.initialize(initialization);
            Co.initialize(initialization);

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
    protected GravesLSTMWeightSet weightSet;

    /**
     * Current weight set.
     *
     */
    protected GravesLSTMWeightSet currentWeightSet;

    /**
     * Matrix to store previous output.
     *
     */
    private Matrix previousOutput;

    /**
     * Matrix to store previous state.
     *
     */
    private Matrix previousCellState;

    /**
     * Tanh activation function needed for Graves LSTM
     *
     */
    private final ActivationFunction tanh;

    /**
     * Activation function needed for Graves LSTM
     *
     */
    private final ActivationFunction sigmoid;

    /**
     * Sigmoid activation function for output
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
     * Flag if state weights are regulated.
     *
     */
    private boolean regulateStateWeights;

    /**
     * Input matrix for procedure construction.
     *
     */
    private Matrix input;

    /**
     * Constructor for Graves LSTM layer.<br>
     * Supported parameters are:<br>
     *     - doubleTanh: true if tanh operation at final output step is executed otherwise false (default value true).<br>
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value true).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value false).<br>
     *     - regulateStateWeights: true if recurrent state weights are regulated otherwise false (default value false).<br>
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for Graves LSTM layer.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public GravesLSTMLayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        this (layerIndex, initialization, false, params);
    }

    /**
     * Constructor for Graves LSTM layer.<br>
     * Supported parameters are:<br>
     *     - doubleTanh: true if tanh operation at final output step is executed otherwise false (default value true).<br>
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value true).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value false).<br>
     *     - regulateStateWeights: true if recurrent state weights are regulated otherwise false (default value false).<br>
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param isBirectional if true recurrent layer is bidirectional otherwise false
     * @param params parameters for Graves LSTM layer.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    protected GravesLSTMLayer(int layerIndex, Initialization initialization, boolean isBirectional, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, isBirectional, params);
        tanh = new ActivationFunction(UnaryFunctionType.TANH);
        sigmoid = new ActivationFunction(UnaryFunctionType.SIGMOID);
        activationFunction = tanh;
    }

    /**
     * Constructor for Graves LSTM layer.<br>
     * Supported parameters are:<br>
     *     - doubleTanh: true if tanh operation at final output step is executed otherwise false (default value true).<br>
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value true).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value false).<br>
     *     - regulateStateWeights: true if recurrent state weights are regulated otherwise false (default value false).<br>
     *
     * @param layerIndex layer index
     * @param activationFunction activation function used.
     * @param initialization initialization function for weight.
     * @param params parameters for Graves LSTM layer.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public GravesLSTMLayer(int layerIndex, ActivationFunction activationFunction, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        this (layerIndex, activationFunction, initialization, false, params);
    }

    /**
     * Constructor for Graves LSTM layer.<br>
     * Supported parameters are:<br>
     *     - doubleTanh: true if tanh operation at final output step is executed otherwise false (default value true).<br>
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value true).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value false).<br>
     *     - regulateStateWeights: true if recurrent state weights are regulated otherwise false (default value false).<br>
     *
     * @param layerIndex layer index
     * @param activationFunction activation function used.
     * @param initialization initialization function for weight.
     * @param isBirectional if true recurrent layer is bidirectional otherwise false
     * @param params parameters for Graves LSTM layer.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    protected GravesLSTMLayer(int layerIndex, ActivationFunction activationFunction, Initialization initialization, boolean isBirectional, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, isBirectional, params);
        tanh = new ActivationFunction(UnaryFunctionType.TANH);
        sigmoid = new ActivationFunction(UnaryFunctionType.SIGMOID);
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
        regulateStateWeights = false;
    }

    /**
     * Returns parameters used for Graves LSTM layer.
     *
     * @return parameters used for Graves LSTM layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + GravesLSTMLayer.paramNameTypes;
    }

    /**
     * Sets parameters used for Graves LSTM layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - doubleTanh: true if tanh operation at final output step is executed otherwise false (default value true).<br>
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value).<br>
     *     - regulateStateWeights: true if state weights are regulated otherwise false (default value).<br>
     *
     * @param params parameters used for Graves LSTM layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("doubleTanh")) doubleTanh = params.getValueAsBoolean("doubleTanh");
        if (params.hasParam("regulateDirectWeights")) regulateDirectWeights = params.getValueAsBoolean("regulateDirectWeights");
        if (params.hasParam("regulateRecurrentWeights")) regulateRecurrentWeights = params.getValueAsBoolean("regulateRecurrentWeights");
        if (params.hasParam("regulateStateWeights")) regulateStateWeights = params.getValueAsBoolean("regulateStateWeights");
    }

    /**
     * Returns if direct weights are regulated.
     *
     * @return true if direct weights are regulated otherwise false.
     */
    protected boolean getRegulateDirectWeights() {
        return regulateDirectWeights;
    }

    /**
     * Returns if recurrent weights are regulated.
     *
     * @return true if recurrent weights are regulated otherwise false.
     */
    protected boolean getRegulateRecurrentWeights() {
        return regulateRecurrentWeights;
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
        weightSet = new GravesLSTMWeightSet(initialization, getPreviousLayerWidth(), getInternalLayerWidth(), regulateDirectWeights, regulateRecurrentWeights);
        currentWeightSet = weightSet;
    }

    /**
     * Defines layer procedure for forward and backward calculation (automatic gradient) by applying procedure factory.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected void defineProcedure() throws MatrixException, DynamicParamException, NeuralNetworkException {
        currentWeightSet = weightSet;
        super.defineProcedure();
    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public MMatrix getInputMatrices(boolean resetPreviousInput) throws MatrixException {
        input = new DMatrix(getPreviousLayerWidth(), 1, Initialization.ONE);
        input.setName("Input");
        if (getPreviousLayer().isBidirectional()) input = input.split(getPreviousLayerWidth() / 2, true);
        if (resetPreviousInput) {
            previousOutput = new DMatrix(getInternalLayerWidth(), 1);
            previousCellState = new DMatrix(getInternalLayerWidth(), 1);
        }
        return new MMatrix(input);
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public MMatrix getForwardProcedure() throws MatrixException {
        previousOutput.setName("PrevOutput");
        previousCellState.setName("PrevC");

        // i = sigmoid(Wi * x + Ui * out(t-1) + Ci * c(t-1) + bi) → Input gate
        Matrix i = currentWeightSet.Wi.dot(input).add(currentWeightSet.Ui.dot(previousOutput)).add(currentWeightSet.Ci.multiply(previousCellState)).add(currentWeightSet.bi);
        i = i.apply(sigmoid);
        i.setName("i");

        // f = sigmoid(Wf * x + Uf * out(t-1) + Cf * c(t-1) + bf) → Forget gate
        Matrix f = currentWeightSet.Wf.dot(input).add(currentWeightSet.Uf.dot(previousOutput)).add(currentWeightSet.Cf.multiply(previousCellState)).add(currentWeightSet.bf);
        f = f.apply(sigmoid);
        f.setName("f");

        // s = tanh(Ws * x + Us * out(t-1) + bs) → State update
        Matrix s = currentWeightSet.Ws.dot(input).add(currentWeightSet.Us.dot(previousOutput)).add(currentWeightSet.bs);
        s = s.apply(tanh);
        s.setName("s");

        // c = i x s + f x c(t-1) → Internal cell state
        Matrix c = i.multiply(s).add(previousCellState.multiply(f));
        c.setName("c");

        previousCellState = c;

        // o = sigmoid(Wo * x + Uo * out(t-1) + Co * ct + bo) → Output gate
        Matrix o = currentWeightSet.Wo.dot(input).add(currentWeightSet.Uo.dot(previousOutput)).add(currentWeightSet.Co.multiply(c)).add(currentWeightSet.bo);
        o = o.apply(sigmoid);
        o.setName("o");

        // h = activationFunction(c) x o or h = c x o → Output
        Matrix h = (doubleTanh ? c.apply(activationFunction) : c).multiply(o);
        h.setName("Output");

        previousOutput = h;

        MMatrix outputs = new MMatrix(1, "Output");
        outputs.put(0, h);
        return outputs;

    }

    /**
     * Returns matrices for which gradient is not calculated.
     *
     * @return matrices for which gradient is not calculated.
     */
    protected HashSet<Matrix> getStopGradients() {
        return new HashSet<>();
    }

    /**
     * Returns constant matrices.
     *
     * @return constant matrices.
     */
    protected HashSet<Matrix> getConstantMatrices() {
        return new HashSet<>();
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
