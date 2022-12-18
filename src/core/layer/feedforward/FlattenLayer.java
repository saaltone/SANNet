/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.layer.feedforward;

import core.layer.AbstractExecutionLayer;
import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.util.HashSet;
import java.util.TreeMap;

/**
 * Implements flattening layer.<br>
 * Flattens in forward direction inputs from width x height x depth to width * height * depth x 1 x 1.<br>
 * Unflattens in backward direction gradients from width * height * depth x 1 x 1 to width x height x depth.<br>
 *
 */
public class FlattenLayer extends AbstractExecutionLayer {

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
     */
    public void initializeDimensions() {
        setLayerWidth(getDefaultPreviousLayer().getLayerWidth() * getDefaultPreviousLayer().getLayerHeight() * getDefaultPreviousLayer().getLayerDepth());
        setLayerHeight(1);
        setLayerDepth(1);
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
     * Defines layer procedure for forward and backward calculation (automatic gradient) by applying procedure factory.<br>
     *
     */
    protected void defineProcedure() {
    }

    /**
     * Reinitializes neural network layer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void reinitialize() throws MatrixException {
        this.reset();
    }

    /**
     * Takes single forward processing step to process layer input(s).<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forwardProcess() throws MatrixException {
        this.reset();
        setLayerOutputs(getDefaultLayerInput().flatten());
    }

    /**
     * Takes single backward processing step to process layer output gradient(s) towards input.<br>
     * Applies automated backward (automatic gradient) procedure when relevant to layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backwardProcess() throws MatrixException {
        setLayerOutputGradients(getLayerOutputGradients().unflatten(getDefaultPreviousLayer().getLayerWidth(), getDefaultPreviousLayer().getLayerHeight(), getDefaultPreviousLayer().getLayerDepth()));
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
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     */
    public TreeMap<Integer, MMatrix> getInputMatrices(boolean resetPreviousInput) {
        return null;
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     */
    public MMatrix getForwardProcedure() {
        return null;
    }

    /**
     * Executes weight updates with regularizers and optimizer.
     *
     */
    public void optimize() {
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
