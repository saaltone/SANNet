/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer.attention;

import core.layer.NeuralNetworkLayer;
import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashSet;
import java.util.TreeMap;

/**
 * Implements input based attention layer.
 *
 */
public class InputBasedAttention extends AbstractAttentionLayer {

    /**
     * Implements weight set for layer.
     *
     */
    protected class InputBasedAttentionWeightSet implements WeightSet, Serializable {

        @Serial
        private static final long serialVersionUID = -7854770677975563135L;

        /**
         * Input attention weight matrix.
         *
         */
        private final Matrix attentionWeight;

        /**
         * Set of weights.
         *
         */
        private final HashSet<Matrix> weights = new HashSet<>();

        /**
         * Constructor for weight set
         *
         * @param initialization weight initialization function.
         * @param previousLayers input layers.
         */
        InputBasedAttentionWeightSet(Initialization initialization, TreeMap<Integer, NeuralNetworkLayer> previousLayers) {
            int previousLayerWidth = previousLayers.get(previousLayers.firstKey()).getLayerWidth();
            attentionWeight = new DMatrix(1, previousLayerWidth, initialization);
            attentionWeight.setName("AttentionWeight");

            weights.add(attentionWeight);

            registerWeight(attentionWeight, false, false);
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
            attentionWeight.initialize(initialization);
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
    protected InputBasedAttentionWeightSet weightSet;

    /**
     * Constructor for input based attention layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for input based layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public InputBasedAttention(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
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
     */
    public void initializeWeights() {
        weightSet = new InputBasedAttentionWeightSet(initialization, getPreviousLayers());
    }

    /**
     * Return score matrix for attention.
     *
     * @param input input
     * @param inputIndex input index
     * @param previousOutput previous output
     * @return score matrix.
     */
    protected Matrix getScoreMatrix(Matrix input, int inputIndex, Matrix previousOutput) throws MatrixException {
        Matrix scoreMatrix = weightSet.attentionWeight.dot(input);
        scoreMatrix.setName("Score" + inputIndex);
        return scoreMatrix;
    }

}
