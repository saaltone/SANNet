/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
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
 * Implements layer that joins multiple inputs from previous layers.
 *
 */
public class JoinLayer extends AbstractExecutionLayer {

    /**
     * Implements weight set for layer.
     *
     */
    protected class JoinWeightSet implements WeightSet, Serializable {

        @Serial
        private static final long serialVersionUID = -1035983607775966046L;

        /**
         * Join input weight matrices.
         *
         */
        private final TreeMap<Integer, Matrix> joinInputWeights = new TreeMap<>();

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
         */
        JoinWeightSet(Initialization initialization, int layerWidth, int layerDepth) {
            if (getLayerWidth() != getPreviousLayerTotalWidth()) {
                Matrix previousInputWeight = new DMatrix(layerWidth, getPreviousLayerTotalWidth(), layerDepth, initialization);
                previousInputWeight.setName("JoinedInputWeight");
                weights.add(previousInputWeight);
                registerWeight(previousInputWeight, false, false);
                joinInputWeights.put(0, previousInputWeight);
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
    protected JoinWeightSet weightSet;

    /**
     * Input matrices for procedure construction.
     *
     */
    private TreeMap<Integer, Matrix> inputs;

    /**
     * Equal function for joining matrices from previous layers.
     *
     */
    private final UnaryFunction equalFunction = new UnaryFunction(UnaryFunctionType.EQUAL);

    /**
     * Constructor for join layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for connector layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public JoinLayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, params);
    }

    /**
     * Returns total width of previous layers.
     *
     * @return total width of previous layers.
     */
    private int getPreviousLayerTotalWidth() {
        int totalPreviousLayerWidth = 0;
        for (NeuralNetworkLayer previousLayer : getPreviousLayers().values()) totalPreviousLayerWidth += previousLayer.getLayerWidth();
        return totalPreviousLayerWidth;
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
     * Returns true if input is joined otherwise returns false.
     *
     * @return true if input is joined otherwise returns false.
     */
    public boolean isJoinedInput() {
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
        weightSet = new JoinWeightSet(initialization, getLayerWidth(), getLayerDepth());
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

        int layerHeight = -1;
        int layerDepth = -1;
        for (Map.Entry<Integer, NeuralNetworkLayer> entry : getPreviousLayers().entrySet()) {
            if (layerHeight == -1 || layerDepth == -1) {
                layerHeight = entry.getValue().getLayerHeight();
                layerDepth = entry.getValue().getLayerDepth();
            }
            else if (layerHeight != entry.getValue().getLayerHeight() || layerDepth != entry.getValue().getLayerDepth()) throw new MatrixException("All inputs must have same height and depth.");
            Matrix input = new DMatrix(entry.getValue().getLayerWidth(), layerHeight, layerDepth, Initialization.ONE);
            input.setName("Input" + entry.getValue().getLayerIndex());
            inputs.put(inputIndex++, input);
        }

        if (inputs.size() > 1) {
            StringBuilder joinedInputName = new StringBuilder("JoinedInput[");
            for (int joinedInputIndex = 0; joinedInputIndex < inputs.size(); joinedInputIndex++) {
                joinedInputName.append(inputs.get(joinedInputIndex).getName()).append(joinedInputIndex < inputs.size() - 1 ? "," : "]");
            }
            Matrix joinedInput = new JMatrix(inputs, true);
            joinedInput.setName(joinedInputName.toString());
            inputs.clear();
            inputs.put(0, joinedInput);
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
        Matrix output = getLayerWidth() == getPreviousLayerTotalWidth() ? inputs.get(0).apply(equalFunction) : weightSet.joinInputWeights.get(0).dot(inputs.get(0));

        if (output != null) output.setName("Output");
        return output;

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
