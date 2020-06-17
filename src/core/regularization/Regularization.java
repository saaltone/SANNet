/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.regularization;

import utils.DynamicParam;
import utils.DynamicParamException;
import utils.Sequence;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Interface for regularization functions.
 *
 */
public interface Regularization {

    /**
     * Sets parameters for regularizer.
     *
     * @param params parameters for regularizer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void setParams(DynamicParam params) throws DynamicParamException;

    /**
     * Indicates to regularizer if neural network is in training mode.
     *
     * @param isTraining if true neural network is in state otherwise false.
     */
    void setTraining(boolean isTraining);

    /**
     * Executes regularization method for forward step.<br>
     *
     * @param sequence input sequence.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void forward(Sequence sequence) throws MatrixException;

    /**
     * Executes regularization method for forward step.<br>
     *
     * @param inputs inputs.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void forward(MMatrix inputs) throws MatrixException;

    /**
     * Cumulates error from regularization. Mainly from L1 / L2 / Lp regularization.
     *
     * @param weight weight matrix.
     * @return cumulated error from regularization.
     */
    double error(Matrix weight);

    /**
     * Executes regularization method for backward phase at pre in.
     *
     * @param weight weight matrix.
     * @param weightGradientSum gradient sum of weight.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void backward(Matrix weight, Matrix weightGradientSum) throws MatrixException;

    /**
     * Returns name of regularization.
     *
     * @return name of regularization.
     */
    String getName();

}

