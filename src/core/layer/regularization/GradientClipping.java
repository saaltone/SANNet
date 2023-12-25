/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer.regularization;

import core.layer.AbstractExecutionLayer;
import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.Initialization;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.TreeMap;

/**
 * Implements gradient clipping.<br>
 * Gradient clipping cuts gradient when certain threshold is reached to prevent then from growing too big i.e. exploding.<br>
 * <br>
 * Reference: <a href="https://hackernoon.com/gradient-clipping-57f04f0adae">...</a><br>
 *
 */
public class GradientClipping extends AbstractExecutionLayer {

    /**
     * Parameter name types for gradient clipping.
     *     - threshold: threshold for clipping gradients. Default value 2.<br>
     *
     */
    private final static String paramNameTypes = "(threshold:DOUBLE)";

    /**
     * Threshold for gradient clipping.
     *
     */
    private double threshold;

    /**
     * Input matrices for procedure construction.
     *
     */
    private TreeMap<Integer, Matrix> inputs;

    /**
     * Constructor for gradient clipping layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for feedforward layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public GradientClipping(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException {
        super (layerIndex, initialization, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        threshold = 2;
    }

    /**
     * Returns parameters used for gradient clipping layer.
     *
     * @return parameters used for gradient clipping layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + GradientClipping.paramNameTypes;
    }

    /**
     * Sets parameters used for gradient clipping.<br>
     * <br>
     * Supported parameters are:<br>
     *     - threshold: threshold for clipping gradients. Default value 2.<br>
     *
     * @param params parameters used for gradient clipping.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("threshold")) threshold = params.getValueAsDouble("threshold");
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
        Matrix input = new DMatrix(getLayerWidth(), getLayerHeight(), getLayerDepth(), Initialization.ONE);
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
        output = output.gradientClip(threshold);
        output.setName("Output");

        return output;
    }

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        return "Threshold: " + threshold;
    }

}
