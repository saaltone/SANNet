/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.metrics;

import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.sampling.Sequence;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.LinkedHashMap;

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
     * Returns last error.
     *
     * @return last error.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    double getLastError() throws MatrixException, DynamicParamException;

    /**
     * Reports error.
     *
     * @param predicted predicted sample.
     * @param actual actual (true) sample.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void report(Matrix predicted, Matrix actual) throws MatrixException, DynamicParamException;

    /**
     * Reports errors and handles them as either regression or classification errors depending on metrics initialization.
     *
     * @param predicted predicted errors.
     * @param actual actual (true) error.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if reporting of errors fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void report(LinkedHashMap<Integer, Matrix> predicted, LinkedHashMap<Integer, Matrix> actual) throws MatrixException, NeuralNetworkException, DynamicParamException;

    /**
     * Reports errors and handles them as either regression or classification errors depending on metrics initialization.
     *
     * @param predicted predicted errors.
     * @param actual actual (true) error.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if reporting of errors fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void report(MMatrix predicted, MMatrix actual) throws MatrixException, NeuralNetworkException, DynamicParamException;

    /**
     * Reports errors and handles them as either regression or classification errors depending on metrics initialization.
     *
     * @param predicted predicted errors.
     * @param actual actual (true) error.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if reporting of errors fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void report(Sequence predicted, Sequence actual) throws MatrixException, NeuralNetworkException, DynamicParamException;

    /**
     * Reports single error value.
     *
     * @param error single error value to be reported.
     */
    void report(double error);

    /**
     * Prints classification report.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void printReport() throws MatrixException, DynamicParamException;

}
