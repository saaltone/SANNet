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
import utils.procedure.node.Node;

/**
 * Interface for normalization functions.<br>
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
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    void reset() throws MatrixException;

    /**
     * Reinitializes normalizer.
     *
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    void reinitialize() throws MatrixException;

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
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void initialize(Node node) throws MatrixException, DynamicParamException;

    /**
     * Initializes normalization.
     *
     * @param weight weight for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void initialize(Matrix weight) throws MatrixException, DynamicParamException;

    /**
     * Executes forward step for normalization.<br>
     * Used typically for batch and layer normalization.<br>
     *
     * @param node node for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void forward(Node node) throws MatrixException, DynamicParamException;

    /**
     * Executes backward step for normalization.<br>
     * Used typically for batch normalization.<br>
     *
     * @param node node for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void backward(Node node) throws MatrixException, DynamicParamException;

    /**
     * Executes forward step for normalization.<br>
     * Used typically for layer normalization.<br>
     *
     * @param node node for normalization.
     * @param inputIndex input index for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void forward(Node node, int inputIndex) throws MatrixException, DynamicParamException;

    /**
     * Executes forward step for normalization.<br>
     * Used typically for weight normalization.<br>
     *
     * @param weight weight for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void forward(Matrix weight) throws MatrixException, DynamicParamException;

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
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void backward(Node node, int outputIndex) throws MatrixException, DynamicParamException;

    /**
     * Executes backward step for normalization.<br>
     * Used typically for weight normalization.<br>
     *
     * @param weight weight for backward normalization.
     * @param weightGradient gradient of weight for backward normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void backward(Matrix weight, Matrix weightGradient) throws MatrixException, DynamicParamException;

    /**
     * Executes optimizer step for normalizer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void optimize() throws MatrixException, DynamicParamException;

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
