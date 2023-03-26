/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer.attention;

import core.activation.ActivationFunction;
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
 * Implements additive attention layer.
 *
 */
public class AdditiveAttentionLayer extends AbstractAttentionLayer {

    /**
     * Implements weight set for layer.
     *
     */
    protected class AdditiveAttentionWeightSet implements WeightSet, Serializable {

        @Serial
        private static final long serialVersionUID = -7854770677975563135L;

        /**
         * Input attention weight matrix.
         *
         */
        private final Matrix attentionWeight;

        /**
         * Attention bias matrix.
         *
         */
        private final Matrix attentionBias;

        /**
         * Attention v matrix.
         *
         */
        private final Matrix v;

        /**
         * Set of weights.
         *
         */
        private final HashSet<Matrix> weights = new HashSet<>();

        /**
         * Constructor for weight set
         *
         * @param initialization weight initialization function.
         * @param layerWidth width of current layer.
         * @param previousLayers input layers.
         */
        AdditiveAttentionWeightSet(Initialization initialization, int layerWidth, TreeMap<Integer, NeuralNetworkLayer> previousLayers) {
            int previousLayerWidth = previousLayers.get(previousLayers.firstKey()).getLayerWidth();
            attentionWeight = new DMatrix(layerWidth, 2 * previousLayerWidth, 1, initialization);
            attentionWeight.setName("AttentionWeight");
            attentionBias = new DMatrix(layerWidth, 1, 1);
            attentionBias.setName("AttentionBias");
            v = new DMatrix(1, layerWidth, 1);
            v.setName("vMatrix");

            weights.add(attentionWeight);
            weights.add(attentionBias);
            weights.add(v);

            registerWeight(attentionWeight, false, false);
            registerWeight(attentionBias, false, false);
            registerWeight(v, false, false);
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
            attentionBias.reset();
            v.initialize(initialization);
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
    protected AdditiveAttentionWeightSet weightSet;

    /**
     * Tanh activation function.
     *
     */
    protected final ActivationFunction tanhActivationFunction = new ActivationFunction(UnaryFunctionType.TANH);

    /**
     * Constructor for additive attention layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for additive attention layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public AdditiveAttentionLayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
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
        weightSet = new AdditiveAttentionWeightSet(initialization, getLayerWidth(), getPreviousLayers());
    }

    /**
     * Return score matrix for attention.
     *
     * @param input input
     * @param inputIndex input index
     * @param previousOutput previous output
     * @return score matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix getScoreMatrix(Matrix input, int inputIndex, Matrix previousOutput) throws MatrixException {
        Matrix scoreMatrix = weightSet.v.dot(weightSet.attentionWeight.dot(input.join(previousOutput, true)).add(weightSet.attentionBias).apply(tanhActivationFunction));
        scoreMatrix.setName("Score" + inputIndex);
        return scoreMatrix;
    }

}
