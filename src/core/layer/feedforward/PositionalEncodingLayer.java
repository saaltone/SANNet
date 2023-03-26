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

import java.util.HashSet;
import java.util.TreeMap;

/**
 * Implements positional encoding layer.<br>
 *
 * Reference: <a href="https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/">...</a> <br>
 */
public class PositionalEncodingLayer extends AbstractExecutionLayer {

    /**
     * Parameter name types for positional encoding layer.
     *     - positionIndex: Position index for positional encoding.<br>
     *
     */
    private final static String paramNameTypes = "(positionIndex:INT)";

    /**
     * Position (token) index of layer.
     *
     */
    private int positionIndex;

    /**
     * Constant for positional encoding.
     *
     */
    private static final int n = 10000;

    /**
     * Positional encoding matrix.
     *
     */
    private Matrix positionalEncodingMatrix;

    /**
     * Input matrix for procedure construction.
     *
     */
    private Matrix input;

    /**
     * Constructor for positional encoding layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for positional encoding layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public PositionalEncodingLayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException {
        super (layerIndex, initialization, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        positionIndex = -1;
    }

    /**
     * Returns parameters used for positional encoding layer.
     *
     * @return parameters used for positional encoding layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + PositionalEncodingLayer.paramNameTypes;
    }

    /**
     * Sets parameters used for positional encoding layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - positionIndex: Position index for positional encoding.<br>
     *
     * @param params parameters used for positional encoding layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("positionIndex")) positionIndex = params.getValueAsInteger("positionIndex");
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
        int previousLayerWidth = getDefaultPreviousLayer().getLayerWidth();
        int previousLayerHeight = getDefaultPreviousLayer().getLayerHeight();
        int previousLayerDepth = getDefaultPreviousLayer().getLayerDepth();
        positionalEncodingMatrix = new DMatrix(previousLayerWidth, previousLayerHeight, previousLayerDepth);
        for (int index = 0; index < previousLayerWidth; index++) {
            double positionalCode = (double)positionIndex / Math.pow(n, 2 * (double)index / (double)previousLayerWidth);
            double positionalEncodingCode = (positionIndex % 2 == 0) ? Math.sin(positionalCode) : Math.cos(positionalCode);
            positionalEncodingMatrix.setValue(index, 0, 0, positionalEncodingCode);
        }
        positionalEncodingMatrix.setName("PositionalEncodingMatrix" + positionIndex);
    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     */
    public TreeMap<Integer, Matrix> getInputMatrices(boolean resetPreviousInput) {
        input = new DMatrix(getDefaultPreviousLayer().getLayerWidth(), 1, 1, Initialization.ONE);
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
        if (positionIndex < 0) throw new MatrixException("Position index must have positive value.");
        Matrix output = input.add(positionalEncodingMatrix);

        output.setName("Output");
        return output;
    }

    /**
     * Returns matrices for which gradient is not calculated.
     *
     * @return matrices for which gradient is not calculated.
     */
    public HashSet<Matrix> getStopGradients() {
        return new HashSet<>() {{ add(positionalEncodingMatrix); }};
    }

    /**
     * Returns constant matrices.
     *
     * @return constant matrices.
     */
    public HashSet<Matrix> getConstantMatrices() {
        return new HashSet<>() {{ add(positionalEncodingMatrix); }};
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
        return "Positional index: " + positionIndex;
    }


}
