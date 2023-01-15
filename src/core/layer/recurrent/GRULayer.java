/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
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
import java.util.TreeMap;

/**
 * Implements gated recurrent unit (GRU).<br>
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
     * Parameter name types for GRU layer.
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value true).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value false).<br>
     *
     */
    private final static String paramNameTypes = "(regulateDirectWeights:BOOLEAN), " +
            "(regulateRecurrentWeights:BOOLEAN)";

    /**
     * Implements weight set for layer.
     *
     */
    protected class GRUWeightSet implements WeightSet, Serializable {

        @Serial
        private static final long serialVersionUID = -2646665508373510779L;

        /**
         * Weights for update gate
         *
         */
        private final Matrix Wz;

        /**
         * Weights for reset gate
         *
         */
        private final Matrix Wr;

        /**
         * Weights for input activation
         *
         */
        private final Matrix Wh;

        /**
         * Weights for recurrent update gate
         *
         */
        private final Matrix Uz;

        /**
         * Weights for recurrent reset gate
         *
         */
        private final Matrix Ur;

        /**
         * Weights for current input activation
         *
         */
        private final Matrix Uh;

        /**
         * Bias for update gate
         *
         */
        private final Matrix bz;

        /**
         * Bias for reset gate
         *
         */
        private final Matrix br;

        /**
         * Bias for input activation
         *
         */
        private final Matrix bh;

        /**
         * Matrix of ones for calculation of z
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
        GRUWeightSet(Initialization initialization, int previousLayerWidth, int layerWidth, boolean regulateDirectWeights, boolean regulateRecurrentWeights) {
            Wz = new DMatrix(layerWidth, previousLayerWidth, initialization);
            Wz.setName("Wz");
            Wr = new DMatrix(layerWidth, previousLayerWidth, initialization);
            Wr.setName("Wr");
            Wh = new DMatrix(layerWidth, previousLayerWidth, initialization);
            Wh.setName("Wh");

            Uz = new DMatrix(layerWidth, layerWidth, initialization);
            Uz.setName("Uz");
            Ur = new DMatrix(layerWidth, layerWidth, initialization);
            Ur.setName("Ur");
            Uh = new DMatrix(layerWidth, layerWidth, initialization);
            Uh.setName("Uh");

            bz = new DMatrix(layerWidth, 1);
            bz.setName("bz");
            br = new DMatrix(layerWidth, 1);
            br.setName("br");
            bh = new DMatrix(layerWidth, 1);
            bh.setName("bh");

            weights.add(Wz);
            weights.add(Wr);
            weights.add(Wh);

            weights.add(Uz);
            weights.add(Ur);
            weights.add(Uh);

            weights.add(bz);
            weights.add(br);
            weights.add(bh);

            registerWeight(Wz, regulateDirectWeights, true);
            registerWeight(Wr, regulateDirectWeights, true);
            registerWeight(Wh, regulateDirectWeights, true);

            registerWeight(Uz, regulateRecurrentWeights, true);
            registerWeight(Ur, regulateRecurrentWeights, true);
            registerWeight(Uh, regulateRecurrentWeights, true);

            registerWeight(bz, false, false);
            registerWeight(br, false, false);
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
            Wz.initialize(initialization);
            Wr.initialize(initialization);
            Wh.initialize(initialization);

            Uz.initialize(initialization);
            Ur.initialize(initialization);
            Uh.initialize(initialization);

            bz.reset();
            br.reset();
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
    protected GRUWeightSet weightSet;

    /**
     * Current weight set.
     *
     */
    protected GRUWeightSet currentWeightSet;

    /**
     * Matrix to store previous output
     *
     */
    private Matrix previousOutput;

    /**
     * Tanh activation function needed for GRU
     *
     */
    private final ActivationFunction tanh;

    /**
     * Sigmoid activation function needed for GRU
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
     * Constructor for GRU layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for GRU layer.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public GRULayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
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
     * Returns parameters used for GRU layer.
     *
     * @return parameters used for GRU layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + GRULayer.paramNameTypes;
    }

    /**
     * Sets parameters used for GRU layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - regulateDirectWeights: true if direct weights are regulated otherwise false (default value true).<br>
     *     - regulateRecurrentWeights: true if recurrent weights are regulated otherwise false (default value false).<br>
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
     * Returns weight set.
     *
     * @return weight set.
     */
    protected WeightSet getWeightSet() {
        return weightSet;
    }

    /**
     * Initializes GRU layer.<br>
     * Initialize weights and bias and their gradients.<br>
     *
     */
    public void initializeWeights() {
        currentWeightSet = weightSet = new GRUWeightSet(initialization, getDefaultPreviousLayer().getLayerWidth(), getLayerWidth(), regulateDirectWeights, regulateRecurrentWeights);
    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public TreeMap<Integer, MMatrix> getInputMatrices(boolean resetPreviousInput) throws MatrixException {
        input = new DMatrix(getDefaultPreviousLayer().getLayerWidth(), 1, Initialization.ONE);
        input.setName("Input" + getDefaultPreviousLayer().getLayerIndex());
        if (resetPreviousInput) {
            previousOutput = new DMatrix(getLayerWidth(), 1);
        }
        return new TreeMap<>() {{ put(0, new MMatrix(input)); }};
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public MMatrix getForwardProcedure() throws MatrixException {
        previousOutput.setName("PreviousOutput");

        // z = sigmoid(Wz * x + Uz * out(t-1) + bz) → Update gate
        Matrix z = currentWeightSet.Wz.dot(input).add(currentWeightSet.Uz.dot(previousOutput)).add(currentWeightSet.bz);
        z = z.apply(sigmoid);
        z.setName("z");

        // r = sigmoid(Wr * x + Ur * out(t-1) + br) → Reset gate
        Matrix r = currentWeightSet.Wr.dot(input).add(currentWeightSet.Ur.dot(previousOutput)).add(currentWeightSet.br);
        r = r.apply(sigmoid);
        r.setName("r");

        // h = tanh(Wh * x + Uh * out(t-1) * r + bh) → Input activation
        Matrix h = currentWeightSet.Wh.dot(input).add(currentWeightSet.Uh.dot(previousOutput).multiply(r)).add(currentWeightSet.bh);
        h = h.apply(tanh);
        h.setName("h");

        // s = (1 - z) x h + z x out(t-1) → Internal state
        Matrix s = currentWeightSet.ones.subtract(z).multiply(h).add(z.multiply(previousOutput));
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
    public HashSet<Matrix> getStopGradients() {
        return new HashSet<>() {{ add(currentWeightSet.ones); }};
    }

    /**
     * Returns constant matrices.
     *
     * @return constant matrices.
     */
    public HashSet<Matrix> getConstantMatrices() {
        return new HashSet<>() {{ add(currentWeightSet.ones); }};
    }

}

