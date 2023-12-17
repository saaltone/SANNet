/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer.attention;

import core.layer.NeuralNetworkLayer;
import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.Initialization;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashSet;
import java.util.Map;
import java.util.TreeMap;

/**
 * Implements location-based attention layer.
 *
 */
public class LocationBasedAttention extends DotAttentionLayer {

    /**
     * Implements weight set for layer.
     *
     */
    protected class LocationBasedAttentionWeightSet implements WeightSet, Serializable {

        @Serial
        private static final long serialVersionUID = 1788541533785360815L;

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
         * Position weight matrix.
         *
         */
        protected final Matrix positionWeight;

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
        LocationBasedAttentionWeightSet(Initialization initialization, TreeMap<Integer, NeuralNetworkLayer> previousLayers) throws MatrixException {
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
            positionWeight = new DMatrix(previousLayerWidth, previousLayerWidth, previousLayerDepth, initialization);
            positionWeight.setName("PositionWeight");

            weights.add(queryWeight);
            weights.add(keyWeight);
            weights.add(valueWeight);
            weights.add(positionWeight);

            registerWeight(queryWeight, false, false);
            registerWeight(keyWeight, false, false);
            registerWeight(valueWeight, false, false);
            registerWeight(positionWeight, false, false);
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
            positionWeight.initialize(initialization);
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
    protected LocationBasedAttentionWeightSet weightSet;

    /**
     * Positional encoding matrix.
     *
     */
    private Matrix positionalEncodingMatrix;

    /**
     * Constructor for location-based attention layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for location-based attention layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public LocationBasedAttention(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, params);
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
        super.initializeWeights();
        weightSet = new LocationBasedAttentionWeightSet(initialization, getPreviousLayers());
        int previousLayerWidth = getDefaultPreviousLayer().getLayerWidth();
        int previousLayerHeight = getDefaultPreviousLayer().getLayerHeight() * getPreviousLayers().size();
        int previousLayerDepth = getDefaultPreviousLayer().getLayerDepth();

        positionalEncodingMatrix = new DMatrix(previousLayerWidth, previousLayerHeight, previousLayerDepth);
        registerConstantMatrix(positionalEncodingMatrix);
        registerStopGradient(positionalEncodingMatrix);
    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public TreeMap<Integer, Matrix> getInputMatrices(boolean resetPreviousInput) throws MatrixException {
        int previousLayerWidth = getDefaultPreviousLayer().getLayerWidth();
        int previousLayerHeight = getDefaultPreviousLayer().getLayerHeight() * getPreviousLayers().size();
        int previousLayerDepth = getDefaultPreviousLayer().getLayerDepth();
        for (int depth = 0; depth < previousLayerDepth; depth++) {
            for (int column = 0; column < previousLayerHeight; column++) {
                for (int row = 0; row < previousLayerWidth; row++) {
                    double positionalCode = (double) column / Math.pow(previousLayerWidth, 2 * (double) row / (double) previousLayerWidth);
                    double positionalEncodingCode = (column % 2 == 0) ? Math.sin(positionalCode) : Math.cos(positionalCode);
                    positionalEncodingMatrix.setValue(row, column, depth, positionalEncodingCode);
                }
            }
        }
        positionalEncodingMatrix.setName("PositionalEncodingMatrix");
        return super.getInputMatrices(resetPreviousInput);
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
        Matrix value = weightSet.valueWeight.dot(joinedInput);
        value.setName("Value");
        Matrix position = weightSet.positionWeight.dot(positionalEncodingMatrix).apply(transposeFunction);
        position.setName("Position");

        Matrix output = query.add(key).add(position).apply(transposeFunction).multiply(value);
        output.setName("Output");
        return output;
    }

}
