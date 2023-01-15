/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer.recurrent;

import core.activation.ActivationFunction;
import core.layer.AbstractExecutionLayer;
import core.layer.NeuralNetworkLayer;
import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashSet;
import java.util.Map;
import java.util.TreeMap;

/**
 * Implements attention layer.<br>
 *
 * Reference: https://machinelearningmastery.com/adding-a-custom-attention-layer-to-recurrent-neural-network-in-keras/<br>
 * Reference: https://www.analyticsvidhya.com/blog/2019/11/comprehensive-guide-attention-mechanism-deep-learning/
 */
public class AttentionLayer extends AbstractExecutionLayer {

    /**
     * Implements weight set for layer.
     *
     */
    protected class AttentionWeightSet implements WeightSet, Serializable {

        @Serial
        private static final long serialVersionUID = 2862320451826596230L;

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
        AttentionWeightSet(Initialization initialization, int layerWidth, TreeMap<Integer, NeuralNetworkLayer> previousLayers) {
            int previousLayerWidth = previousLayers.get(previousLayers.firstKey()).getLayerWidth();
            attentionWeight = new DMatrix(layerWidth, 2 * previousLayerWidth, initialization);
            attentionWeight.setName("AttentionWeight");
            attentionBias = new DMatrix(layerWidth, 1);
            attentionBias.setName("AttentionBias");
            v = new DMatrix(1, layerWidth);
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
    protected AttentionWeightSet weightSet;

    /**
     * Tanh activation function.
     *
     */
    protected final ActivationFunction tanhActivationFunction = new ActivationFunction(UnaryFunctionType.TANH);

    /**
     * Softmax activation function.
     *
     */
    protected final ActivationFunction softmaxActivationFunction = new ActivationFunction(UnaryFunctionType.SOFTMAX);

    /**
     * Input matrices for procedure construction.
     *
     */
    private TreeMap<Integer, MMatrix> inputs;

    /**
     * Matrix to store previous output.
     *
     */
    private Matrix previousOutput;

    /**
     * Constructor for attention layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for attention layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public AttentionLayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, params);
    }

    /**
     * Initializes neural network layer dimensions.
     *
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     */
    public void initializeDimensions() throws NeuralNetworkException {
        if (getLayerWidth() == -1) {
            if ((getDefaultPreviousLayer().getLayerWidth()) < 1) throw new NeuralNetworkException("Default previous layer width must be positive. Invalid value: " + (getDefaultPreviousLayer().getLayerWidth()));
            setLayerWidth(getDefaultPreviousLayer().getLayerWidth());
            setLayerHeight(1);
            setLayerDepth(1);
        }
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
        weightSet = new AttentionWeightSet(initialization, getLayerWidth(), getPreviousLayers());
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

        TreeMap<Integer, Matrix> inputMatrices = new TreeMap<>();
        int layerWidth = -1;
        for (Map.Entry<Integer, NeuralNetworkLayer> entry : getPreviousLayers().entrySet()) {
            if (layerWidth == -1) layerWidth = entry.getValue().getLayerWidth();
            else if (layerWidth != entry.getValue().getLayerWidth()) throw new MatrixException("All inputs must have same width.");
            Matrix input = new DMatrix(layerWidth, 1, Initialization.ONE);
            input.setName("Input" + entry.getValue().getLayerIndex());
            inputMatrices.put(entry.getKey(), input);
        }

        for (int inputIndex = 0; inputIndex < inputMatrices.size(); inputIndex++) {
            inputs.put(inputIndex, new MMatrix(inputMatrices.get(inputIndex)));
        }

        if (resetPreviousInput) {
            previousOutput = new DMatrix(getLayerWidth(), 1, Initialization.ONE);
        }

        return inputs;
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public MMatrix getForwardProcedure() throws MatrixException {
        previousOutput.setName("PreviousOutput");

        Matrix totalScoreMatrix = null;
        for (Map.Entry<Integer, MMatrix> entry : inputs.entrySet()) {
            Matrix inputMatrix = weightSet.attentionWeight.dot(entry.getValue().get(0).join(previousOutput, true)).add(weightSet.attentionBias);
            inputMatrix.setName("Input" + entry.getKey());

            Matrix scoreMatrix = weightSet.v.dot(inputMatrix.apply(tanhActivationFunction));
            scoreMatrix.setName("Score" + entry.getKey());

            totalScoreMatrix = totalScoreMatrix == null ? scoreMatrix : totalScoreMatrix.join(scoreMatrix, true);
        }

        assert totalScoreMatrix != null;
        Matrix weightMatrix = totalScoreMatrix.apply(softmaxActivationFunction);
        weightMatrix.setName("Weights");

        Matrix contextMatrix = null;
        for (Map.Entry<Integer, MMatrix> entry : inputs.descendingMap().entrySet()) {
            Matrix singleWeightMatrix =  weightMatrix.unjoin(entry.getKey(), 0, 1, 1);
            singleWeightMatrix.setName("Weight" + entry.getKey());
            contextMatrix = contextMatrix == null ? entry.getValue().get(0).multiply(singleWeightMatrix) : contextMatrix.add(entry.getValue().get(0).multiply(singleWeightMatrix));
        }

        previousOutput = contextMatrix;

        assert contextMatrix != null;
        contextMatrix.setName("Output");
        MMatrix outputs = new MMatrix(1, "Output");
        outputs.put(0, contextMatrix);
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
