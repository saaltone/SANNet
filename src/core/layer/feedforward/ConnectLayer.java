/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer.feedforward;

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
 * Implements layer that connects multiple inputs from previous layers.
 *
 */
public class ConnectLayer extends AbstractExecutionLayer {

    /**
     * Implements weight set for layer.
     *
     */
    protected class ConnectWeightSet implements WeightSet, Serializable {

        @Serial
        private static final long serialVersionUID = 2862320451826596230L;

        /**
         * Connect input weight matrices.
         *
         */
        private final TreeMap<Integer, Matrix> connectInputWeights = new TreeMap<>();

        /**
         * Set of weights.
         *
         */
        private final HashSet<Matrix> weights = new HashSet<>();

        /**
         * Constructor for weight set
         *
         * @param initialization weight initialization function.
         * @param layerWidth     width of current layer.
         * @param layerDepth     depth of current layer.
         * @param previousLayers input layers.
         */
        ConnectWeightSet(Initialization initialization, int layerWidth, int layerDepth, TreeMap<Integer, NeuralNetworkLayer> previousLayers) {
            for (Map.Entry<Integer, NeuralNetworkLayer> entry : previousLayers.entrySet()) {
                Matrix connectInputWeight = new DMatrix(layerWidth, entry.getValue().getLayerWidth(), layerDepth, initialization);
                connectInputWeight.setName("ConnectWeight" + entry.getValue().getLayerIndex());
                weights.add(connectInputWeight);
                registerWeight(connectInputWeight, false, false);
                connectInputWeights.put(entry.getKey(), connectInputWeight);
            }
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
            for (Matrix weight : weights) weight.initialize(initialization);
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
    protected ConnectWeightSet weightSet;

    /**
     * Input matrices for procedure construction.
     *
     */
    private TreeMap<Integer, Matrix> inputs;

    /**
     * Constructor for connector layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for connector layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public ConnectLayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException {
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
        weightSet = new ConnectWeightSet(initialization, getLayerWidth(), getLayerDepth(), getPreviousLayers());
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

        TreeMap<Integer, Matrix> inputMatrices = new TreeMap<>();
        int layerHeight = -1;
        int layerDepth = -1;
        for (Map.Entry<Integer, NeuralNetworkLayer> entry : getPreviousLayers().entrySet()) {
            if (layerHeight == -1 || layerDepth == -1) {
                layerHeight = entry.getValue().getLayerHeight();
                layerDepth = entry.getValue().getLayerDepth();
            }
            else if (layerHeight != entry.getValue().getLayerHeight() || layerDepth != entry.getValue().getLayerDepth()) throw new MatrixException("All inputs must have same size.");
            Matrix input = new DMatrix(entry.getValue().getLayerWidth(), layerHeight, layerDepth, Initialization.ONE);
            input.setName("Input" + entry.getValue().getLayerIndex());
            inputMatrices.put(entry.getKey(), input);
        }

        for (int inputIndex = 0; inputIndex < inputMatrices.size(); inputIndex++) {
            inputs.put(inputIndex, inputMatrices.get(inputIndex));
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
        Matrix output = null;
        for (Map.Entry<Integer, Matrix> entry : inputs.entrySet()) {
            output = output == null ? weightSet.connectInputWeights.get(entry.getKey()).dot(entry.getValue()) : output.add(weightSet.connectInputWeights.get(entry.getKey()).dot(entry.getValue()));
        }

        if (output != null) output.setName("Output");
        return output;

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
