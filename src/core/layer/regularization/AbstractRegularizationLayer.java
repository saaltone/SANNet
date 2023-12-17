/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer.regularization;

import core.layer.AbstractExecutionLayer;
import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.Initialization;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.TreeMap;

/**
 * Implements abstract regularization layer for common regularization functions.
 *
 */
public abstract class AbstractRegularizationLayer extends AbstractExecutionLayer {

    /**
     * Constructor for abstract regularization layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for feedforward layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractRegularizationLayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException {
        super (layerIndex, initialization, params);
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
        return null;
    }

    /**
     * Defines layer procedure for forward and backward calculation (automatic gradient) by applying procedure factory.<br>
     *
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     */
    protected void defineProcedure() throws NeuralNetworkException {
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     */
    public Matrix getForwardProcedure() {
        return null;
    }

    /**
     * Takes single forward processing step to process layer input(s).<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void forwardProcess() throws MatrixException, DynamicParamException {
        passLayerOutputs();
    }

    /**
     * Takes single backward processing step to process layer output gradient(s) towards input.<br>
     * Applies automated backward (automatic gradient) procedure when relevant to layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void backwardProcess() throws MatrixException, DynamicParamException {
        passLayerOutputGradients();
    }

    /**
     * Executes weight updates with regularizers and optimizer.
     *
     */
    public void optimize() {
    }

    /**
     * Prints forward expression chains of layer.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    public void printExpressions() throws NeuralNetworkException {
        System.out.println(getLayerName() + ": ");
        System.out.println("No explicit procedure.");
        System.out.println();
    }

    /**
     * Prints backward gradient chains of layer.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    public void printGradients() throws NeuralNetworkException {
        System.out.println(getLayerName() + ": ");
        System.out.println("No explicit procedure.");
        System.out.println();
    }

}
