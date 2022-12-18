/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.layer.feedforward;

import core.activation.ActivationFunction;
import core.layer.AbstractExecutionLayer;
import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.util.HashSet;
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
    private TreeMap<Integer, MMatrix> inputs;

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
        this.activationFunction = activationFunction != null ? activationFunction : new ActivationFunction(UnaryFunctionType.RELU);
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
    public TreeMap<Integer, MMatrix> getInputMatrices(boolean resetPreviousInput) throws MatrixException {
        int layerDepth = getLayerDepth();
        inputs = new TreeMap<>();
        for (int index = 0; index < layerDepth; index++) {
            Matrix input = new DMatrix(getLayerWidth(), getLayerHeight(), Initialization.ONE);
            input.setName("Input" + getDefaultPreviousLayer().getLayerIndex() + (layerDepth > 1 ? "{" + index + "}" : ""));
            inputs.put(index, new MMatrix(input));
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
        int layerDepth = getLayerDepth();
        MMatrix outputs = new MMatrix(layerDepth, "Output");

        for (int depthIndex = 0; depthIndex < layerDepth; depthIndex++) {
            Matrix output = inputs.get(depthIndex).get(0);
            output = output.apply(activationFunction);
            output.setName("Output" + (layerDepth > 1 ? depthIndex : ""));
            outputs.put(depthIndex, output);
        }

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
        return "Activation function: " + activationFunction.getName();
    }

}
