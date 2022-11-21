package core.layer.feedforward;

import core.layer.AbstractExecutionLayer;
import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.*;
import utils.procedure.Procedure;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
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
     *     - inputLayers: list of connected previous layers.<br>
     *     - joinPreviousLayerInputs: if true join outputs of previous layers otherwise connects via weights and summation. Default value false.<br>
     *
     */
    private final static String paramNameTypes = "(inputLayers:LIST), " +
            "(joinPreviousLayerInputs:BOOLEAN)";

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
        private final ArrayList<Matrix> otherInputWeights = new ArrayList<>();

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
         * @param inputLayerWidths width of previous connection layer.
         * @param joinPreviousLayerInputs if true input and previous connect input are joined otherwise previous connect layer input is added through dedicated weight.
         */
        ConnectorWeightSet(Initialization initialization, int layerWidth, ArrayList<Integer> inputLayerWidths, boolean joinPreviousLayerInputs) {
            if (!inputLayerWidths.isEmpty() && !joinPreviousLayerInputs) {
                for (Integer inputLayerWidth : inputLayerWidths) {
                    Matrix otherInputWeight = new DMatrix(layerWidth, inputLayerWidth, initialization);
                    otherInputWeight.setName("PreviousConnectWeight");
                    weights.add(otherInputWeight);
                    registerWeight(otherInputWeight, false, false);
                    otherInputWeights.add(otherInputWeight);
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
            for (Matrix otherInputWeight : otherInputWeights) otherInputWeight.initialize(initialization);
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
     * Indices of other input layers.
     *
     */
    private ArrayList<Integer> inputLayerList;

    /**
     * If true previous layer inputs are joined otherwise previous layer inputs are added through dedicated weights.
     *
     */
    private boolean joinPreviousLayerInputs;

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
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        inputLayerList = new ArrayList<>();
        joinPreviousLayerInputs = false;
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
     *     - inputLayers: list of connected previous layers.<br>
     *     - joinPreviousLayerInputs: if true join inputs of previous layers otherwise connects via weight and summation. Default value false.<br>
     *
     * @param params parameters used for connector layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("inputLayers")) {
            String[] inputLayers = params.getValueAsList("inputLayers");
            for (String inputLayerIndexString : inputLayers) {
                int inputLayerIndex = Integer.parseInt(inputLayerIndexString);
                if (inputLayerIndex < 0 || inputLayerIndex > getLayerIndex() - 2) throw new DynamicParamException("Previous connect layer index must be positive value and connection must be created from a layer having index at least 2 smaller than this layer: " + getLayerIndex());
                inputLayerList.add(inputLayerIndex);
            }
        }
        if (params.hasParam("joinPreviousLayerInputs")) {
            joinPreviousLayerInputs = params.getValueAsBoolean("joinPreviousLayerInputs");
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
     * Returns reversed procedure.
     *
     * @return reversed procedure.
     */
    protected Procedure getReverseProcedure() {
        return null;
    }

    /**
     * Returns true if input is joined otherwise returns false.
     *
     * @return true if input is joined otherwise returns false.
     */
    protected boolean isJoinedInput() {
        return joinPreviousLayerInputs;
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
     * Returns width of neural network layer.
     *
     * @return width of neural network layer.
     */
    public int getLayerWidth()  {
        if (!isJoinedInput()) return getPreviousLayerWidth();
        else {
            int otherLayerWidth = 0;
            for (Integer otherLayer : inputLayerList) otherLayerWidth += getPreviousLayerWidth(otherLayer);
            return getPreviousLayerWidth() + otherLayerWidth;
        }
    }

    /**
     * Initializes neural network layer weights.
     *
     */
    public void initializeWeights() {
        weightSet = new ConnectorWeightSet(initialization, getLayerWidth(), getOtherLayerInputWidths(), joinPreviousLayerInputs);
    }

    /**
     * Returns other layer input widths.
     *
     * @return other layer input widths.
     */
    private ArrayList<Integer> getOtherLayerInputWidths() {
        ArrayList<Integer> otherLayerInputWidths = new ArrayList<>();
        for (Integer otherLayerIndex : inputLayerList) otherLayerInputWidths.add(getPreviousLayerWidth(otherLayerIndex));
        return otherLayerInputWidths;
    }

    /**
     * Adds other input layers.
     *
     */
    protected void addOtherInputLayers() {
        for (Integer otherLayerIndex : inputLayerList) addInputSequence(otherLayerIndex);
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

        ArrayList<Matrix> inputMatrices = new ArrayList<>();
        for (Integer inputLayerIndex : getInputLayerIndices()) {
            Matrix input = new DMatrix(getPreviousLayerWidth(inputLayerIndex), 1, Initialization.ONE);
            input = handleBidirectionalInput(input, inputLayerIndex);
            input.setName("Input" + inputLayerIndex);
            inputMatrices.add(input);
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
        if (!inputLayerList.isEmpty()) {
            if (isJoinedInput()) {
                output = inputs.get(0).get(0).apply(equalFunction);
            }
            else {
                for (Map.Entry<Integer, MMatrix> entry : inputs.entrySet()) {
                    if (entry.getKey() == 0) output = entry.getValue().get(0);
                    else output = output == null ? entry.getValue().get(0) : output.add(weightSet.otherInputWeights.get(entry.getKey() - 1).dot(entry.getValue().get(0)));
                }
            }
        }
        else {
            output = inputs.get(0).get(0).apply(equalFunction);
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
        return "Connect from previous layers: " + (!inputLayerList.isEmpty() ? inputLayerList : "N/A") + ", Join previous layer inputs: " + (isJoinedInput() ? "Yes" : "No");
    }


}
