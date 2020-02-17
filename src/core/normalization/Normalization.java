/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.normalization;

import core.optimization.Optimizer;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.procedure.Node;

/**
 * Interface for normalization functions.
 *
 */
public interface Normalization {

    /**
     * Sets parameters for normalizer.
     *
     * @param params parameters for normalizer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void setParams(DynamicParam params) throws DynamicParamException;

    /**
     * Resets normalizer state.
     *
     */
    void reset();

    /**
     * Indicates to normalizer if neural network is in training mode.
     *
     * @param isTraining if true neural network is in state otherwise false.
     */
    void setTraining(boolean isTraining);

    /**
     * Sets optimizer for normalizer.
     *
     * @param optimizer optimizer
     */
    void setOptimizer(Optimizer optimizer);

    /**
     * Executes forward step for normalization.<br>
     * Used typically for batch normalization.<br>
     *
     * @param node node for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void forward(Node node) throws MatrixException;

    /**
     * Executes backward step for normalization.<br>
     * Used typically for batch normalization.<br>
     *
     * @param node node for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void backward(Node node) throws MatrixException;

    /**
     * Executes forward step for normalization.<br>
     * Used typically for layer normalization.<br>
     *
     * @param node node for normalization.
     * @param inputIndex input index for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void forward(Node node, int inputIndex) throws MatrixException;

    /**
     * Executes backward step for normalization.<br>
     * Used typically for layer normalization.<br>
     *
     * @param node node for normalization.
     * @param outputIndex input index for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void backward(Node node, int outputIndex) throws MatrixException;

    /**
     * Executes forward step for normalization.<br>
     * Used typically for weight normalization.<br>
     *
     * @param W weight for normalization.
     * @return normalized weight.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix forward(Matrix W) throws MatrixException;

    /**
     * Executes backward step for normalization.<br>
     * Used typically for weight normalization.<br>
     *
     * @param W weight for backward's normalization.
     * @param dW gradient of weight for backward normalization.
     * @return input weight gradients for backward normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix backward(Matrix W, Matrix dW) throws MatrixException;

}
