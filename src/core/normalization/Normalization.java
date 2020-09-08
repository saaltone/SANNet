/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

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
     * Initializes normalization.
     *
     * @param node node for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void initialize(Node node) throws MatrixException;

    /**
     * Initializes normalization.
     *
     * @param weight weight for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void initialize(Matrix weight) throws MatrixException;

    /**
     * Executes forward step for normalization.<br>
     * Used typically for batch and layer normalization.<br>
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
     * Executes forward step for normalization.<br>
     * Used typically for weight normalization.<br>
     *
     * @param weight weight for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void forward(Matrix weight) throws MatrixException;

    /**
     * Finalizes forward step for normalization.<br>
     * Used typically for weight normalization.<br>
     *
     * @param weight weight for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void forwardFinalize(Matrix weight) throws MatrixException;

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
     * Executes backward step for normalization.<br>
     * Used typically for weight normalization.<br>
     *
     * @param weight weight for backward normalization.
     * @param weightGradient gradient of weight for backward normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void backward(Matrix weight, Matrix weightGradient) throws MatrixException;

    /**
     * Executes optimizer step for normalizer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void optimize() throws MatrixException;

    /**
     * Returns name of normalization.
     *
     * @return name of normalization.
     */
    String getName();

    /**
     * Prints expression chains of normalization.
     *
     */
    void printExpressions();

    /**
     * Prints gradient chains of normalization.
     *
     */
    void printGradients();

}
