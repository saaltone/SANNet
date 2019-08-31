/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.normalization;

import utils.DynamicParam;
import utils.DynamicParamException;
import utils.Matrix;
import utils.MatrixException;

import java.util.TreeMap;

/**
 * Defines interface for normalization functions.
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
     * Executes normalization method for forward step.<br>
     * This operation assumes execution at step start.<br>
     *
     * @param ins input samples for forward step.
     * @param channels number of channels of a convolutional layer. Only relevant for convolutional layer.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void forwardPre(TreeMap<Integer, Matrix> ins, int channels) throws MatrixException;

    /**
     * Executes normalization method for forward step.<br>
     * This operation assumes execution post activation.<br>
     *
     * @param outs input samples for forward step.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void forwardPost(TreeMap<Integer, Matrix> outs) throws MatrixException;

    /**
     * Executes normalization method for backward phase.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void backward() throws MatrixException;

}
