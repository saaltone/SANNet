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

import java.util.Map;
import java.util.TreeMap;

/**
 * Implements abstract multi-input layer.
 *
 */
public abstract class AbstractMultiInputLayer extends AbstractExecutionLayer {

    /**
     * Input matrices for procedure construction.
     *
     */
    private TreeMap<Integer, Matrix> inputs;

    /**
     * Equal function for handling matrices from previous layers.
     *
     */
    private final UnaryFunction equalFunction = new UnaryFunction(UnaryFunctionType.EQUAL);

    /**
     * Constructor for add layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for add layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public AbstractMultiInputLayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, params);
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

        int inputIndex = 0;

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
            inputs.put(inputIndex++, input);
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
        for (Matrix input : inputs.values()) {
            output = output == null ? input.apply(equalFunction) : executeOperation(input, output);
        }

        if (output != null) output.setName("Output");
        return output;

    }

    /**
     * Executes operation
     *
     * @param input current input.
     * @param output current output
     * @return result.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract Matrix executeOperation(Matrix input, Matrix output) throws MatrixException;

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        return "";
    }

}
