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
public class MinGRULayer extends AbstractRecurrentLayer {

    /**
     * Parameter name types for abstract Min GRU layer.
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value true).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value false).<br>
     *
     */
    private final static String paramNameTypes = "(regulateDirectWeights:BOOLEAN), " +
            "(regulateRecurrentWeights:BOOLEAN)";

    /**
     * Class that defines weight set for layer.
     *
     */
    protected class MinGRUWeightSet implements WeightSet, Serializable {

        @Serial
        private static final long serialVersionUID = 5474685289943016274L;

        /**
         * Weights for forget gate
         *
         */
        private final Matrix Wf;

        /**
         * Weights for input activation
         *
         */
        private final Matrix Wh;

        /**
         * Weights for recurrent forget gate
         *
         */
        private final Matrix Uf;

        /**
         * Weights for current input activation
         *
         */
        private final Matrix Uh;

        /**
         * Bias for forget gate
         *
         */
        private final Matrix bf;

        /**
         * Bias for input activation
         *
         */
        private final Matrix bh;

        /**
         * Ones matrix for calculation of z
         *
         */
        private Matrix ones;

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
        MinGRUWeightSet(Initialization initialization, int previousLayerWidth, int layerWidth, boolean regulateDirectWeights, boolean regulateRecurrentWeights) {
            Wf = new DMatrix(layerWidth, previousLayerWidth, initialization, "Wf");
            Wh = new DMatrix(layerWidth, previousLayerWidth, initialization, "Wh");

            Uf = new DMatrix(layerWidth, layerWidth, initialization, "Uf");
            Uh = new DMatrix(layerWidth, layerWidth, initialization, "Uh");

            bf = new DMatrix(layerWidth, 1, "bf");
            bh = new DMatrix(layerWidth, 1, "bh");

            weights.add(Wf);
            weights.add(Wh);

            weights.add(Uf);
            weights.add(Uh);

            weights.add(bf);
            weights.add(bh);

            registerWeight(Wf, regulateDirectWeights, true);
            registerWeight(Wh, regulateDirectWeights, true);

            registerWeight(Uf, regulateRecurrentWeights, true);
            registerWeight(Uh, regulateRecurrentWeights, true);

            registerWeight(bf, false, false);
            registerWeight(bh, false, false);

            ones = (ones == null) ? new DMatrix(layerWidth, 1, Initialization.ONE) : ones;
            ones.setName("1");
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
            Wf.initialize(initialization);
            Wh.initialize(initialization);

            Uf.initialize(initialization);
            Uh.initialize(initialization);

            bf.reset();
            bh.reset();
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
    protected MinGRUWeightSet weightSet;

    /**
     * Current weight set.
     *
     */
    protected MinGRUWeightSet currentWeightSet;

    /**
     * Matrix to store previous output
     *
     */
    private Matrix previousOutput;

    /**
     * Tanh activation function needed for Minimal GRU
     *
     */
    private final ActivationFunction tanh;

    /**
     * Sigmoid activation function needed for Minimal GRU
     *
     */
    private final ActivationFunction sigmoid;

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
     * Constructor for Minimal GRU layer.
     *
     * @param layerIndex layer Index.
     * @param initialization initialization function for weight.
     * @param params parameters for minimal GRU layer.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public MinGRULayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, params);
        tanh = new ActivationFunction(UnaryFunctionType.TANH);
        sigmoid = new ActivationFunction(UnaryFunctionType.SIGMOID);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        regulateDirectWeights = true;
        regulateRecurrentWeights = false;
    }

    /**
     * Returns parameters used for Minimal GRU layer.
     *
     * @return parameters used for Minimal GRU layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + MinGRULayer.paramNameTypes;
    }

    /**
     * Sets parameters used for Minimal GRU layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value true).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value false).<br>
     *
     * @param params parameters used for Minimal GRU layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("regulateDirectWeights")) regulateDirectWeights = params.getValueAsBoolean("regulateDirectWeights");
        if (params.hasParam("regulateRecurrentWeights")) regulateRecurrentWeights = params.getValueAsBoolean("regulateRecurrentWeights");
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
        weightSet = new MinGRUWeightSet(initialization, getPreviousLayerWidth(), super.getLayerWidth(), regulateDirectWeights, regulateRecurrentWeights);
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
        input = new DMatrix(getPreviousLayerWidth(), 1, Initialization.ONE, "Input");
        if (getPreviousLayer().isBidirectional()) input = input.split(getPreviousLayerWidth() / 2, true);
        if (resetPreviousInput) previousOutput = new DMatrix(super.getLayerWidth(), 1);
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

        // f = sigmoid(Wf * x + Uf * out(t-1) + bf) → Forget gate
        Matrix f = currentWeightSet.Wf.dot(input).add(currentWeightSet.Uf.dot(previousOutput)).add(currentWeightSet.bf);
        f = f.apply(sigmoid);
        f.setName("f");

        // h = tanh(Wh * x + Uh * out(t-1) * r + bh) → Input activation
        Matrix h = currentWeightSet.Wh.dot(input).add(currentWeightSet.Uh.dot(previousOutput).multiply(f)).add(currentWeightSet.bh);
        h = h.apply(tanh);
        h.setName("h");

        // s = (1 - f) x h + f x out(t-1) → Internal state
        Matrix s = currentWeightSet.ones.subtract(f).multiply(h).add(f.multiply(previousOutput));
        s.setName("Output");

        previousOutput = s;

        MMatrix outputs = new MMatrix(1, "Output");
        outputs.put(0, s);
        return outputs;

    }

    /**
     * Returns matrices for which gradient is not calculated.
     *
     * @return matrices for which gradient is not calculated.
     */
    protected HashSet<Matrix> getStopGradients() {
        HashSet<Matrix> stopGradients = new HashSet<>();
        stopGradients.add(currentWeightSet.ones);
        return stopGradients;
    }

    /**
     * Returns constant matrices.
     *
     * @return constant matrices.
     */
    protected HashSet<Matrix> getConstantMatrices() {
        HashSet<Matrix> constantMatrices = new HashSet<>();
        constantMatrices.add(currentWeightSet.ones);
        return constantMatrices;
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

