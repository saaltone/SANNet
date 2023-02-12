package core.layer.reinforcement;

import core.activation.ActivationFunction;
import core.layer.AbstractExecutionLayer;
import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashSet;
import java.util.TreeMap;

/**
 * Implements dueling layer for Deep Q Network.
 *
 */
public class DuelingLayer extends AbstractExecutionLayer {

    /**
     * Parameter name types for dueling layer.
     *     - regulateDirectWeights: true if (direct) weights are regulated otherwise false. Default value true.<br>
     *
     */
    private final static String paramNameTypes = "(regulateDirectWeights:BOOLEAN)";

    /**
     * Implements weight set for layer.
     *
     */
    protected class DuelingWeightSet implements WeightSet, Serializable {

        @Serial
        private static final long serialVersionUID = 7859757124635021579L;

        /**
         * Weight matrix for value.
         *
         */
        private final Matrix valueWeight;

        /**
         * Bias matrix for value.
         *
         */
        private final Matrix valueBias;

        /**
         * Weight matrix for action.
         *
         */
        private final Matrix actionWeight;

        /**
         * Bias matrix for bias.
         *
         */
        private final Matrix actionBias;

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
         */
        DuelingWeightSet(Initialization initialization, int previousLayerWidth, int layerWidth, boolean regulateDirectWeights) {
            valueWeight = new DMatrix(1, previousLayerWidth, initialization);
            valueWeight.setName("ValueWeight");
            valueBias = new DMatrix(layerWidth, 1);
            valueWeight.setName("ValueBias");
            actionWeight = new DMatrix(layerWidth, previousLayerWidth, initialization);
            valueWeight.setName("ActionWeight");
            actionBias = new DMatrix(layerWidth, 1);
            valueWeight.setName("ActionBias");

            weights.add(valueWeight);
            weights.add(valueBias);
            weights.add(actionWeight);
            weights.add(actionBias);

            registerWeight(valueWeight, regulateDirectWeights, true);
            registerWeight(valueBias, false, false);
            registerWeight(actionWeight, regulateDirectWeights, true);
            registerWeight(actionBias, false, false);
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
            valueWeight.initialize(initialization);
            valueBias.reset();
            actionWeight.initialize(initialization);
            actionBias.reset();
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
    protected DuelingWeightSet weightSet;

    /**
     * True if weights are regulated otherwise weights are not regulated.
     *
     */
    private boolean regulateDirectWeights;

    /**
     * Activation function for activation layer.
     *
     */
    protected final ActivationFunction activationFunction;

    /**
     * Input matrix for procedure construction.
     *
     */
    private TreeMap<Integer, MMatrix> inputs;

    /**
     * Constructor for dueling layer.
     *
     * @param layerIndex layer index
     * @param activationFunction activation function used.
     * @param initialization initialization function for weight.
     * @param params parameters for dueling layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public DuelingLayer(int layerIndex, ActivationFunction activationFunction, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, params);
        this.activationFunction = activationFunction != null ? activationFunction : new ActivationFunction(UnaryFunctionType.RELU);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        regulateDirectWeights = true;
    }

    /**
     * Returns parameters used for dueling layer.
     *
     * @return parameters used for dueling layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + DuelingLayer.paramNameTypes;
    }

    /**
     * Sets parameters used for dueling layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - regulateDirectWeights: true if (direct) weights are regulated otherwise false. Default value true.<br>
     *
     * @param params parameters used for dueling layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("regulateDirectWeights")) regulateDirectWeights = params.getValueAsBoolean("regulateDirectWeights");
    }

    /**
     * Checks if layer is recurrent layer type.
     *
     * @return always false.
     */
    public boolean isRecurrentLayer() { return false; }

    /**
     * Checks if layer works with recurrent layers.
     *
     * @return if true layer works with recurrent layers otherwise false.
     */
    public boolean worksWithRecurrentLayer() {
        return true;
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
        weightSet = new DuelingWeightSet(initialization, getDefaultPreviousLayer().getLayerWidth(), getLayerWidth(), regulateDirectWeights);
    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public TreeMap<Integer, MMatrix> getInputMatrices(boolean resetPreviousInput) throws MatrixException {
        inputs = new TreeMap<>();

        Matrix input = new DMatrix(getDefaultPreviousLayer().getLayerWidth(), 1, Initialization.ONE);
        input.setName("Input" + getDefaultPreviousLayer().getLayerIndex());
        inputs.put(0, new MMatrix(input));

        return inputs;
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public MMatrix getForwardProcedure() throws MatrixException {
        Matrix valueOutput = weightSet.valueWeight.dot(inputs.get(0).get(0));
        valueOutput = valueOutput.add(weightSet.valueBias);
        valueOutput = valueOutput.apply(activationFunction);

        Matrix actionOutput = weightSet.actionWeight.dot(inputs.get(0).get(0));
        actionOutput = actionOutput.add(weightSet.actionBias);
        actionOutput = actionOutput.apply(activationFunction);

        Matrix output = valueOutput.add(actionOutput.subtract(actionOutput.meanAsMatrix()));

        output.setName("Output");
        MMatrix outputs = new MMatrix(1, "Output");
        outputs.put(0, output);
        return outputs;

    }

    /**
     * Returns matrices for which gradient is not calculated.
     *
     * @return matrices for which gradient is not calculated.
     */
    public HashSet<Matrix> getStopGradients() {
        return new HashSet<>();
    }

    /**
     * Returns constant matrices.
     *
     * @return constant matrices.
     */
    public HashSet<Matrix> getConstantMatrices() {
        return new HashSet<>();
    }

    /**
     * Returns number of truncated steps for gradient calculation. -1 means no truncation.
     *
     * @return number of truncated steps.
     */
    protected int getTruncateSteps() {
        return -1;
    }

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        return "";
    }

}
