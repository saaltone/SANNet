/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.regularization;

import utils.DynamicParam;
import utils.DynamicParamException;
import utils.Matrix;
import utils.MatrixException;

import java.util.TreeMap;

/**
 * Defines interface for regularization functions.
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
     * Resets regularizer state.
     *
     */
    void reset();

    /**
     * Indicates to regularizer if neural network is in training mode.
     *
     * @param isTraining if true neural network is in state otherwise false.
     */
    void setTraining(boolean isTraining);

    /**
     * Executes regularization method for forward step.<br>
     * This operation assumes execution at step start.<br>
     *
     * @param ins input samples for forward step.
     * @param index if index is zero or positive value operation is executed for this sample. if index is -1 operation is executed for all samples.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void forwardPre(TreeMap<Integer, Matrix> ins, int index) throws MatrixException;

    /**
     * Executes regularization method for forward step.<br>
     * This operation assumes execution post activation.<br>
     *
     * @param outs input samples for forward step.
     */
    void forwardPost(TreeMap<Integer, Matrix> outs);

    /**
     * Cumulates error from regularization. Mainly from L1 / L2 / Lp regularization.
     *
     * @return cumulated error from regularization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    double error() throws MatrixException;

    /**
     * Executes regularization method for backward phase at pre in.
     *
     * @param index if index is zero or positive value operation is executed for this sample. if index is -1 operation is executed for all samples.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void backward(int index) throws MatrixException;

    /**
     * Executes regularization method prior neural network update step.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void update() throws MatrixException;

}

