/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.layer.feedforward;

import core.activation.ActivationFunction;
import core.activation.ActivationFunctionType;
import core.layer.AbstractExecutionLayer;
import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.util.TreeMap;

/**
 * Implements activation layer.<br>
 *
 */
public class ActivationLayer extends AbstractExecutionLayer {

    /**
     * Activation function for activation layer.
     *
     */
    protected final ActivationFunction activationFunction;

    /**
     * Input matrices for procedure construction.
     *
     */
    private TreeMap<Integer, Matrix> inputs;

    /**
     * Constructor for activation layer.
     *
     * @param layerIndex layer index
     * @param activationFunction activation function used.
     * @param params parameters for activation layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public ActivationLayer(int layerIndex, ActivationFunction activationFunction, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, null, params);
        this.activationFunction = activationFunction != null ? activationFunction : new ActivationFunction(ActivationFunctionType.RELU);
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
        inputs = new TreeMap<>();
        Matrix input = new DMatrix(getDefaultPreviousLayer().getLayerWidth(), getDefaultPreviousLayer().getLayerHeight(), getDefaultPreviousLayer().getLayerDepth(), Initialization.ONE);
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
        Matrix output = inputs.get(0);
        output = output.apply(activationFunction);
        output.setName("Output");

        return output;
    }

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        return "Activation function: " + activationFunction.getName();
    }

}
