/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer.attention;

import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.util.HashSet;

/**
 * Implements dot attention layer.
 *
 */
public class DotAttentionLayer extends AbstractAttentionLayer {

    /**
     * Parameter name types for dot attention layer.
     *     - scaled: If true applies scaled dot attention otherwise dot attention. Default false.<br>
     *
     */
    private final static String paramNameTypes = "(scaled:BOOLEAN)";

    /**
     * If true applies scaled dot attention otherwise dot attention.
     *
     */
    private boolean scaled;

    /**
     * Input width.
     *
     */
    private Matrix inputWidthMatrix;

    /**
     * Transpose function.
     *
     */
    private final UnaryFunction transposeFunction = new UnaryFunction(UnaryFunctionType.TRANSPOSE);

    /**
     * Constructor for dot attention layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for dot attention layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public DotAttentionLayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        scaled = false;
        inputWidthMatrix = null;
    }

    /**
     * Returns parameters used for dot attention layer.
     *
     * @return parameters used for dot attention layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + DotAttentionLayer.paramNameTypes;
    }

    /**
     * Sets parameters used for dot attention layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - scaled: If true applies scaled dot attention otherwise dot attention. Default false.<br>
     *
     * @param params parameters used for dot attention layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("scaled")) scaled = params.getValueAsBoolean("scaled");
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
        if (scaled) {
            inputWidthMatrix = new DMatrix(getDefaultPreviousLayer().getLayerWidth());
            inputWidthMatrix.setName("inputWidth");
        }
    }

    /**
     * Return score matrix for attention.
     *
     * @param input input
     * @param inputIndex input index
     * @param previousOutput previous output
     * @return score matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix getScoreMatrix(Matrix input, int inputIndex, Matrix previousOutput) throws MatrixException {
        Matrix scoreMatrix = scaled ? input.apply(transposeFunction).dot(previousOutput).divide(inputWidthMatrix) : input.apply(transposeFunction).dot(previousOutput);
        scoreMatrix.setName("Score" + inputIndex);
        return scoreMatrix;
    }

    /**
     * Returns matrices for which gradient is not calculated.
     *
     * @return matrices for which gradient is not calculated.
     */
    public HashSet<Matrix> getStopGradients() {
        return scaled ? new HashSet<>() {{ add(inputWidthMatrix); }} : new HashSet<>();
    }

    /**
     * Returns constant matrices.
     *
     * @return constant matrices.
     */
    public HashSet<Matrix> getConstantMatrices() {
        return scaled ? new HashSet<>() {{ add(inputWidthMatrix); }} : new HashSet<>();
    }

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        return "Scaled: " + scaled;
    }

}
