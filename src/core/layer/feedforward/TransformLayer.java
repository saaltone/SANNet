/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer.feedforward;

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
 * Implements transform layer that transforms dimensions of input to output dimensions
 *
 */
public class TransformLayer extends AbstractExecutionLayer {

    /**
     * Parameter name types for transform layer.
     *     - regulateDirectWeights: true if (direct) weights are regulated otherwise false. Default value true.<br>
     *
     */
    private final static String paramNameTypes = "(regulateDirectWeights:BOOLEAN)";

    /**
     * Implements weight set for layer.
     *
     */
    protected class TransformWeightSet implements WeightSet, Serializable {

        @Serial
        private static final long serialVersionUID = -6105877308342892549L;

        /**
         * Weight 0 matrix.
         *
         */
        private final Matrix weight0;

        /**
         * Weight 1 matrix.
         *
         */
        private final Matrix weight1;

        /**
         * Set of weights.
         *
         */
        private final HashSet<Matrix> weights = new HashSet<>();

        /**
         * Constructor for weight set
         *
         * @param initialization        weight initialization function.
         * @param previousLayerWidth    width of previous layer.
         * @param previousLayerHeight   height of previous layer.
         * @param layerWidth            width of current layer.
         * @param layerHeight           height of current layer.
         * @param previousLayerDepth    depth of previous layer.
         * @param regulateDirectWeights if true direct weights are regulated.
         */
        TransformWeightSet(Initialization initialization, int previousLayerWidth, int previousLayerHeight, int layerWidth, int layerHeight, int previousLayerDepth, boolean regulateDirectWeights) {
            weight0 = new DMatrix(previousLayerHeight, layerHeight, previousLayerDepth, initialization);
            weight0.setName("Weight0");
            weight1 = new DMatrix(layerWidth, previousLayerWidth, previousLayerDepth);
            weight1.setName("Weight0");

            weights.add(weight0);
            weights.add(weight1);

            registerWeight(weight0, regulateDirectWeights, true);
            registerWeight(weight1, regulateDirectWeights, true);
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
            weight0.initialize(initialization);
            weight1.initialize(initialization);
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
    protected TransformWeightSet weightSet;

    /**
     * True if weights are regulated otherwise weights are not regulated.
     *
     */
    private boolean regulateDirectWeights;

    /**
     * Input matrix for procedure construction.
     *
     */
    private Matrix input;

    /**
     * Constructor for transform layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for transform layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public TransformLayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException {
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
     * Returns parameters used for transform layer.
     *
     * @return parameters used for transform layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + TransformLayer.paramNameTypes;
    }

    /**
     * Sets parameters used for transform layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - regulateDirectWeights: true if (direct) weights are regulated otherwise false. Default value true.<br>
     *
     * @param params parameters used for transform layer.
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
        weightSet = new TransformWeightSet(initialization, getDefaultPreviousLayer().getLayerWidth(), getDefaultPreviousLayer().getLayerHeight(), getLayerWidth(), getLayerHeight(), getDefaultPreviousLayer().getLayerDepth(), regulateDirectWeights);
    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     */
    public TreeMap<Integer, Matrix> getInputMatrices(boolean resetPreviousInput) {
        input = new DMatrix(getDefaultPreviousLayer().getLayerWidth(), getDefaultPreviousLayer().getLayerHeight(), getDefaultPreviousLayer().getLayerDepth(), Initialization.ONE);
        input.setName("Input" + getDefaultPreviousLayer().getLayerIndex());
        return new TreeMap<>() {{ put(0, input); }};
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix getForwardProcedure() throws MatrixException {
        Matrix output0 = input.dot(weightSet.weight0);
        output0.setName("Output0");
        Matrix output = weightSet.weight1.dot(output0);
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
