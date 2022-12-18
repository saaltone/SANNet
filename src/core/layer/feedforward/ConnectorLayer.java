/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.layer.feedforward;

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
 * Implements layer that connects or joins multiple inputs from previous layers.
 *
 */
public class ConnectorLayer extends AbstractExecutionLayer {

    /**
     * Parameter name types for connector layer.
     *     - joinInputs: if true join outputs of previous layers otherwise connects via weights and summation. Default value false.<br>
     *
     */
    private final static String paramNameTypes = "(joinInputs:BOOLEAN)";

    /**
     * Implements weight set for layer.
     *
     */
    protected class ConnectorWeightSet implements WeightSet, Serializable {

        @Serial
        private static final long serialVersionUID = 2862320451826596230L;

        /**
         * Other input weight matrices.
         *
         */
        private final TreeMap<Integer, Matrix> previousInputWeights = new TreeMap<>();

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
         * @param joinInputs if true input and previous connect input are joined otherwise previous connect layer input is added through dedicated weight.
         */
        ConnectorWeightSet(Initialization initialization, int layerWidth, TreeMap<Integer, NeuralNetworkLayer> previousLayers, boolean joinInputs) {
            if (isJoinedInput() && getLayerWidth() != getPreviousLayerTotalWidth()) {
                Matrix previousInputWeight = new DMatrix(layerWidth, getPreviousLayerTotalWidth(), initialization);
                previousInputWeight.setName("JoinedInputWeight");
                weights.add(previousInputWeight);
                registerWeight(previousInputWeight, false, false);
                previousInputWeights.put(0, previousInputWeight);
            }
            if (!joinInputs) {
                for (Map.Entry<Integer, NeuralNetworkLayer> entry : previousLayers.entrySet()) {
                    Matrix previousInputWeight = new DMatrix(layerWidth, entry.getValue().getLayerWidth(), initialization);
                    previousInputWeight.setName("InputWeight" + entry.getValue().getLayerIndex());
                    weights.add(previousInputWeight);
                    registerWeight(previousInputWeight, false, false);
                    previousInputWeights.put(entry.getKey(), previousInputWeight);
                }
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
    protected ConnectorWeightSet weightSet;

    /**
     * If true previous layer inputs are joined otherwise previous layer inputs are added through dedicated weights.
     *
     */
    private boolean joinInputs;

    /**
     * Input matrix for procedure construction.
     *
     */
    private TreeMap<Integer, MMatrix> inputs;

    /**
     * Equal function for joining matrices from previous layers.
     *
     */
    private final UnaryFunction equalFunction = new UnaryFunction(UnaryFunctionType.EQUAL);

    /**
     * Constructor for connector layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for connector layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public ConnectorLayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, params);
    }

    /**
     * Initializes neural network layer dimensions.
     *
     */
    public void initializeDimensions() {
        if (getLayerWidth() == -1) {
            setLayerWidth(isJoinedInput() ? getPreviousLayerTotalWidth() : getDefaultPreviousLayer().getLayerWidth());
            setLayerHeight(1);
            setLayerDepth(1);
        }
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
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        joinInputs = false;
    }

    /**
     * Returns parameters used for connector layer.
     *
     * @return parameters used for connector layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + ConnectorLayer.paramNameTypes;
    }

    /**
     * Sets parameters used for connector layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - joinInputs: if true join inputs of previous layers otherwise connects via weight and summation. Default value false.<br>
     *
     * @param params parameters used for connector layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("joinInputs")) {
            joinInputs = params.getValueAsBoolean("joinInputs");
        }
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
     * Returns true if input is joined otherwise returns false.
     *
     * @return true if input is joined otherwise returns false.
     */
    protected boolean isJoinedInput() {
        return joinInputs;
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
        weightSet = new ConnectorWeightSet(initialization, getLayerWidth(), getPreviousLayers(), joinInputs);
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
        for (Map.Entry<Integer, NeuralNetworkLayer> entry : getPreviousLayers().entrySet()) {
            Matrix input = new DMatrix(entry.getValue().getLayerWidth(), 1, Initialization.ONE);
            input.setName("Input" + entry.getValue().getLayerIndex());
            inputMatrices.put(entry.getKey(), input);
        }

        if (isJoinedInput()) {
            if (inputMatrices.size() == 1) inputs.put(0, new MMatrix(inputMatrices.get(0)));
            else {
                inputs.put(0, new MMatrix(new JMatrix(inputMatrices, true)));
                Matrix joinedInput = inputs.get(0).get(0);
                StringBuilder joinedInputName = new StringBuilder("JoinedInput[");
                for (int inputIndex = 0; inputIndex < inputMatrices.size(); inputIndex++) {
                    joinedInputName.append(inputMatrices.get(inputIndex).getName()).append(inputIndex < inputMatrices.size() - 1 ? "," : "]");
                }
                joinedInput.setName(joinedInputName.toString());
            }
        }
        else {
            for (int inputIndex = 0; inputIndex < inputMatrices.size(); inputIndex++) {
                inputs.put(inputIndex, new MMatrix(inputMatrices.get(inputIndex)));
            }
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
        Matrix output = null;
        if (isJoinedInput()) {
            output = getLayerWidth() == getPreviousLayerTotalWidth() ? inputs.get(0).get(0).apply(equalFunction) : weightSet.previousInputWeights.get(0).dot(inputs.get(0).get(0));
        }
        else {
            for (Map.Entry<Integer, MMatrix> entry : inputs.entrySet()) {
                output = output == null ? weightSet.previousInputWeights.get(entry.getKey()).dot(entry.getValue().get(0)) : output.add(weightSet.previousInputWeights.get(entry.getKey()).dot(entry.getValue().get(0)));
            }
        }

        if (output != null) output.setName("Output");
        MMatrix outputs = new MMatrix(1, "Output");
        outputs.put(0, output);
        return outputs;

    }

    /**
     * Returns matrices for which gradient is not calculated.
     *
     * @return matrices for which gradient is not calculated.
     */
    protected HashSet<Matrix> getStopGradients() {
        return new HashSet<>();
    }

    /**
     * Returns constant matrices.
     *
     * @return constant matrices.
     */
    protected HashSet<Matrix> getConstantMatrices() {
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
