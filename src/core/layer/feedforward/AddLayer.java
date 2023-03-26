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

import java.util.HashSet;
import java.util.Map;
import java.util.TreeMap;

/**
 * Implements layer that adds multiple inputs.
 *
 */
public class AddLayer extends AbstractExecutionLayer {

    /**
     * Input matrices for procedure construction.
     *
     */
    private TreeMap<Integer, Matrix> inputs;

    /**
     * Constructor for add layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for add layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AddLayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException {
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
        return null;
    }

    /**
     * Initializes neural network layer weights.
     *
     */
    public void initializeWeights() {
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
        int layerWidth = -1;
        int layerHeight = -1;
        int layerDepth = -1;
        for (Map.Entry<Integer, NeuralNetworkLayer> entry : getPreviousLayers().entrySet()) {
            if (layerWidth == -1 || layerHeight == -1 || layerDepth == -1) {
                layerWidth = entry.getValue().getLayerWidth();
                layerHeight = entry.getValue().getLayerHeight();
                layerDepth = entry.getValue().getLayerDepth();
            }
            else if (layerWidth != entry.getValue().getLayerWidth() || layerHeight != entry.getValue().getLayerHeight() || layerDepth != entry.getValue().getLayerDepth()) throw new MatrixException("All inputs must have same size.");
            Matrix input = new DMatrix(layerWidth, layerHeight, layerDepth, Initialization.ONE);
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
        for (Matrix matrix : inputs.values()) {
            output = output == null ? matrix : output.add(matrix);
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
