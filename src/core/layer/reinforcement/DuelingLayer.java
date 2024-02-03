/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.layer.reinforcement;

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
        private static final long serialVersionUID = -8610809232828571461L;

        /**
         * Weight matrix for value.
         *
         */
        private final Matrix valueWeight;

        /**
         * Weight matrix for action.
         *
         */
        private final Matrix actionWeight;

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
            valueWeight = new DMatrix(1, previousLayerWidth, 1, initialization);
            valueWeight.setName("ValueWeight");
            actionWeight = new DMatrix(layerWidth, previousLayerWidth, 1, initialization);
            actionWeight.setName("ActionWeight");

            weights.add(valueWeight);
            weights.add(actionWeight);

            registerWeight(valueWeight, regulateDirectWeights, true);
            registerWeight(actionWeight, regulateDirectWeights, true);
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
            actionWeight.initialize(initialization);
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
     * Input matrix for procedure construction.
     *
     */
    private TreeMap<Integer, Matrix> inputs;

    /**
     * Constructor for dueling layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for dueling layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public DuelingLayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException {
        super (layerIndex, initialization, params);
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
     */
    public TreeMap<Integer, Matrix> getInputMatrices(boolean resetPreviousInput) {
        inputs = new TreeMap<>();

        Matrix input = new DMatrix(getDefaultPreviousLayer().getLayerWidth(), 1, 1, Initialization.ONE);
        input.setName("Input" + getDefaultPreviousLayer().getLayerIndex());
        inputs.put(0, input);

        return inputs;
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix getForwardProcedure() throws MatrixException {
        Matrix valueOutput = weightSet.valueWeight.dot(inputs.get(0));

        Matrix actionOutput = weightSet.actionWeight.dot(inputs.get(0));
        actionOutput = actionOutput.subtract(actionOutput.meanAsMatrix(0));

        Matrix output = valueOutput.add(actionOutput);

        output.setName("Output");
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
