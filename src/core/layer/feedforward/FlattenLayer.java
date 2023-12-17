/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer.feedforward;

import core.layer.AbstractExecutionLayer;
import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.util.TreeMap;

/**
 * Implements flattening layer.<br>
 * Flattens in forward direction inputs from width x height x depth to width * height * depth x 1 x 1.<br>
 * Unflattens in backward direction gradients from width * height * depth x 1 x 1 to width x height x depth.<br>
 *
 */
public class FlattenLayer extends AbstractExecutionLayer {

    /**
     * Input matrix for procedure construction.
     *
     */
    private Matrix input;

    /**
     * Constructor for flatten layer.
     *
     * @param layerIndex layer index
     * @param params parameters for flatten layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public FlattenLayer(int layerIndex, String params) throws NeuralNetworkException, DynamicParamException {
        super (layerIndex, null, params);
    }

    /**
     * Initializes neural network layer dimensions.
     *
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     */
    public void initializeDimensions() throws NeuralNetworkException {
        if (getDefaultPreviousLayer().getLayerWidth() * getDefaultPreviousLayer().getLayerHeight() * getDefaultPreviousLayer().getLayerDepth() < 1) throw new NeuralNetworkException("Default previous layer width must be positive. Invalid dimensions: " + getDefaultPreviousLayer().getLayerWidth() + "x" + getDefaultPreviousLayer().getLayerHeight() + "x" + getDefaultPreviousLayer().getLayerDepth());
        setLayerWidth(getDefaultPreviousLayer().getLayerWidth() * getDefaultPreviousLayer().getLayerHeight() * getDefaultPreviousLayer().getLayerDepth());
        setLayerHeight(1);
        setLayerDepth(1);
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
        Matrix output = input.flatten();

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
