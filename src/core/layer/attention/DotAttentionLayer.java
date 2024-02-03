/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.layer.attention;

import core.layer.AbstractExecutionLayer;
import core.layer.NeuralNetworkLayer;
import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashSet;
import java.util.Map;
import java.util.TreeMap;

/**
 * Implements dot attention layer.
 *
 */
public class DotAttentionLayer extends AbstractExecutionLayer {

    /**
     * Parameter name types for dot attention layer.
     *     - scaled: If true applies scaled self attention otherwise pure self attention. Default true.<br>
     *
     */
    private final static String paramNameTypes = "(scaled:BOOLEAN)";

    /**
     * Implements weight set for layer.
     *
     */
    protected class SelfAttentionWeightSet implements WeightSet, Serializable {

        @Serial
        private static final long serialVersionUID = -555933709661682346L;

        /**
         * Query weight matrix.
         *
         */
        protected final Matrix queryWeight;

        /**
         * Key weight matrix.
         *
         */
        protected final Matrix keyWeight;

        /**
         * Value weight matrix.
         *
         */
        protected final Matrix valueWeight;

        /**
         * Set of weights.
         *
         */
        private final HashSet<Matrix> weights = new HashSet<>();

        /**
         * Constructor for weight set
         *
         * @param initialization weight initialization function.
         * @param previousLayers previous layers.
         * @throws MatrixException throws exception if layer dimensions are not matching.
         */
        SelfAttentionWeightSet(Initialization initialization, TreeMap<Integer, NeuralNetworkLayer> previousLayers) throws MatrixException {
            int previousLayerWidth = -1;
            int previousLayerHeight = -1;
            int previousLayerDepth = -1;
            for (NeuralNetworkLayer previousLayer : previousLayers.values()) {
                if (previousLayerWidth == -1) previousLayerWidth = previousLayer.getLayerWidth();
                else if (previousLayerWidth != previousLayer.getLayerWidth()) throw new MatrixException("All layers must have same width");
                if (previousLayerHeight == -1) previousLayerHeight = previousLayer.getLayerHeight();
                else if (previousLayerHeight != previousLayer.getLayerHeight()) throw new MatrixException("All layers must have same height");
                if (previousLayerDepth == -1) previousLayerDepth = previousLayer.getLayerDepth();
                else if (previousLayerDepth != previousLayer.getLayerDepth()) throw new MatrixException("All layers must have same depth");
            }
            queryWeight = new DMatrix(previousLayerWidth, previousLayerWidth, previousLayerDepth, initialization);
            queryWeight.setName("QueryWeight");
            keyWeight = new DMatrix(previousLayerWidth, previousLayerWidth, previousLayerDepth, initialization);
            keyWeight.setName("KeyWeight");
            valueWeight = new DMatrix(previousLayerWidth, previousLayerWidth, previousLayerDepth, initialization);
            valueWeight.setName("ValueWeight");

            weights.add(queryWeight);
            weights.add(keyWeight);
            weights.add(valueWeight);

            registerWeight(queryWeight, false, false);
            registerWeight(keyWeight, false, false);
            registerWeight(valueWeight, false, false);
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
            queryWeight.initialize(initialization);
            keyWeight.initialize(initialization);
            valueWeight.initialize(initialization);
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
    protected SelfAttentionWeightSet weightSet;

    /**
     * Transpose function.
     *
     */
    protected final UnaryFunction transposeFunction = new UnaryFunction(UnaryFunctionType.TRANSPOSE);

    /**
     * Softmax function.
     *
     */
    protected final UnaryFunction softmaxFunction = new UnaryFunction(UnaryFunctionType.SOFTMAX);

    /**
     * Input matrices for procedure construction.
     *
     */
    protected TreeMap<Integer, Matrix> inputs;

    /**
     * If true applies scaled dot attention otherwise dot attention.
     *
     */
    protected boolean scaled;

    /**
     * Scaling factor.
     *
     */
    protected Matrix scalingFactor;

    /**
     * Constructor for dot attention layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for dot attention layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public DotAttentionLayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, params);
    }

    /**
     * Returns parameters used for dot attention layer.
     *
     * @return parameters used for dot attention layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + DotAttentionLayer.paramNameTypes;
    }

    /**
     * Sets parameters used for dot attention layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - scaled: If true applies scaled dot attention otherwise dot attention. Default true.<br>
     *
     * @param params parameters used for dot attention layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("scaled")) scaled = params.getValueAsBoolean("scaled");
    }

    /**
     * Checks if layer can have multiple previous layers.
     *
     * @return  if true layer can have multiple previous layers otherwise false.
     */
    public boolean canHaveMultiplePreviousLayers() {
        return true;
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        scaled = true;
        scalingFactor = null;
    }

    /**
     * Initializes neural network layer dimensions.
     *
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     */
    public void initializeDimensions() throws NeuralNetworkException {
        super.initializeDimensions();
        setLayerHeight(getDefaultPreviousLayer().getLayerHeight() * getPreviousLayers().size());
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
     * @throws MatrixException throws exception if layer dimensions are not matching.
     */
    public void initializeWeights() throws MatrixException {
        weightSet = new SelfAttentionWeightSet(initialization, getPreviousLayers());
        if (scaled) {
            scalingFactor = new DMatrix(1.0 / Math.sqrt(getDefaultPreviousLayer().getLayerWidth()));
            scalingFactor.setName("ScalingFactor");
            registerConstantMatrix(scalingFactor);
            registerStopGradient(scalingFactor);
        }
    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public TreeMap<Integer, Matrix> getInputMatrices(boolean resetPreviousInput) throws MatrixException {
        inputs = new TreeMap<>();

        int inputIndex = 0;

        int layerWidth = -1;
        int layerHeight = -1;
        int layerDepth = -1;
        for (Map.Entry<Integer, NeuralNetworkLayer> entry : getPreviousLayers().entrySet()) {
            if (layerWidth == -1 || layerHeight == -1 || layerDepth == -1) {
                layerWidth = entry.getValue().getLayerWidth();
                layerHeight = entry.getValue().getLayerHeight();
                layerDepth = entry.getValue().getLayerDepth();
            }
            else if (layerWidth != entry.getValue().getLayerWidth() || layerHeight != entry.getValue().getLayerHeight() || layerDepth != entry.getValue().getLayerDepth()) throw new MatrixException("All inputs must have same size.");

            Matrix input = new DMatrix(layerWidth, layerHeight, layerDepth, Initialization.ONE);
            input.setName("Input" + entry.getValue().getLayerIndex());
            inputs.put(inputIndex++, input);
        }
        return inputs;
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix getForwardProcedure() throws MatrixException {
        Matrix joinedInput = null;
        for (Map.Entry<Integer, Matrix> entry : inputs.entrySet()) {
            joinedInput = joinedInput == null ? entry.getValue() : joinedInput.join(entry.getValue(), false);
        }
        assert joinedInput != null;
        joinedInput.setName("JoinedInput");

        Matrix transposedJoinedInput = joinedInput.apply(transposeFunction);
        transposedJoinedInput.setName("TransposedInput");
        Matrix query = transposedJoinedInput.dot(weightSet.queryWeight);
        query.setName("Query");
        Matrix key = transposedJoinedInput.dot(weightSet.keyWeight);
        key.setName("Key");
        Matrix attentionScores = query.dot(key.apply(transposeFunction));
        attentionScores.setName("AttentionScores");
        if (scaled) {
            attentionScores = attentionScores.multiply(scalingFactor);
            attentionScores.setName("ScaledAttentionScores");
        }

        Matrix attentionsWeights = attentionScores.apply(softmaxFunction);
        attentionsWeights.setName("AttentionWeights");

        Matrix value = weightSet.valueWeight.dot(joinedInput);
        value.setName("Value");
        Matrix output = value.dot(attentionsWeights);
        output.setName("Output");
        return output;

    }

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        return "Scaled: " + scaled;
    }

}
