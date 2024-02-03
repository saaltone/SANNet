/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.metrics;

import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.sampling.Sequence;
import utils.matrix.MatrixException;

/**
 * Defines interface for metric.
 *
 */
public interface Metric {

    /**
     * Returns reference metric.
     *
     * @return reference metric.
     */
    Metric reference();

    /**
     * Resets current metric.
     *
     */
    void reset();

    /**
     * Reinitializes current metric.
     *
     */
    void reinitialize();

    /**
     * Returns last error.
     *
     * @return last error.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    double getLastError() throws MatrixException, DynamicParamException;

    /**
     * Reports errors and handles them as either regression or classification errors depending on metrics initialization.
     *
     * @param predicted    predicted errors.
     * @param actual       actual (true) error.
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if reporting of errors fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     */
    void report(Sequence predicted, Sequence actual) throws MatrixException, NeuralNetworkException, DynamicParamException;

    /**
     * Prints classification report.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void printReport() throws MatrixException, DynamicParamException;

}
