/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.metrics;

import utils.sampling.Sequence;

import java.io.Serial;
import java.io.Serializable;
import java.util.TreeMap;

/**
 * Implements single regression metric with scalar values.
 *
 */
public class SingleRegressionMetric implements Metric, Serializable {

    @Serial
    private static final long serialVersionUID = -1319495528844759912L;

    /**
     * Error count;
     *
     */
    private int errorCount = 0;

    /**
     * Absolute errors.
     *
     */
    private final TreeMap<Integer, Double> absoluteErrors = new TreeMap<>();

    /**
     * Cumulative absolute error.
     *
     */
    private double cumulativeAbsoluteError = 0;

    /**
     * Squared errors.
     *
     */
    private final TreeMap<Integer, Double> squaredErrors = new TreeMap<>();

    /**
     * Cumulative squared error.
     *
     */
    private double cumulativeSquaredError = 0;

    /**
     * Default constructor for single regression metric.
     *
     */
    public SingleRegressionMetric() {
    }

    /**
     * Returns reference metric.
     *
     * @return reference metric.
     */
    public Metric reference() {
        return new SingleRegressionMetric();
    }

    /**
     * Returns number of error samples cumulated.
     *
     * @return number of error samples cumulated.
     */
    public int getErrorCount() {
        return errorCount;
    }

    /**
     * Returns mean absolute error.
     *
     * @return mean absolute error.
     */
    public double getMeanAbsoluteError() {
        return getMeanAbsoluteErrorMatrix();
    }

    /**
     * Returns mean absolute error matrix.
     *
     * @return mean absolute error matrix.
     */
    public double getMeanAbsoluteErrorMatrix() {
        return cumulativeAbsoluteError / (double)getErrorCount();
    }

    /**
     * Returns mean absolute error.
     *
     * @param lastN calculate for last N errors.
     * @return mean absolute error.
     */
    public double getMeanAbsoluteError(int lastN) {
        return getMeanAbsoluteErrorMatrix(lastN);
    }

    /**
     * Returns mean absolute error matrix.
     *
     * @param lastN calculate for last N errors.
     * @return mean absolute error matrix.
     */
    public double getMeanAbsoluteErrorMatrix(int lastN) {
        if (lastN < 1) return 0;
        double lastNCumulativeAbsoluteError = 0;
        int lastNCount = 0;
        for (Integer index : absoluteErrors.descendingKeySet()) {
            lastNCumulativeAbsoluteError += absoluteErrors.get(index);
            if (++lastNCount == lastN) break;
        }
        return lastNCumulativeAbsoluteError / (double)lastN;
    }

    /**
     * Returns mean squared error.
     *
     * @return mean squared error.
     */
    public double getMeanSquaredError() {
        return getMeanSquaredErrorMatrix();
    }

    /**
     * Returns mean squared error matrix.
     *
     * @return mean squared error matrix.
     */
    public double getMeanSquaredErrorMatrix() {
        return Math.sqrt(cumulativeSquaredError) / (double)(getErrorCount());
    }

    /**
     * Returns mean squared error.
     *
     * @param lastN calculate for last N errors.
     * @return mean absolute error.
     */
    public double getMeanSquaredError(int lastN) {
        return getMeanSquaredErrorMatrix(lastN);
    }

    /**
     * Returns mean squared error matrix.
     *
     * @param lastN calculate for last N errors.
     * @return mean squared error matrix.
     */
    public double getMeanSquaredErrorMatrix(int lastN) {
        if (lastN < 1) return 0;
        double lastNCumulativeSquaredError = 0;
        int lastNCount = 0;
        for (Integer index : squaredErrors.descendingKeySet()) {
            lastNCumulativeSquaredError += squaredErrors.get(index);
            if (++lastNCount == lastN) break;
        }
        return lastNCumulativeSquaredError / (double)(lastN);
    }

    /**
     * Returns last error.
     *
     * @return last error.
     */
    public double getLastError() {
        return Math.sqrt(squaredErrors.get(squaredErrors.lastKey()));
    }

    /**
     * Resets cumulative error and sample count.
     *
     */
    public void reset() {
        absoluteErrors.clear();
        cumulativeAbsoluteError = 0;
        squaredErrors.clear();
        cumulativeSquaredError = 0;
        errorCount = 0;
    }

    /**
     * Reports errors and handles them as either regression or classification errors depending on metrics initialization.
     *
     * @param predicted predicted errors.
     * @param actual actual (true) error.
     */
    public void report(Sequence predicted, Sequence actual) {
    }

    /**
     * Reports single error value.
     *
     * @param error single error value to be reported.
     */
    public void report(double error) {
        cumulativeAbsoluteError += error;
        absoluteErrors.put(errorCount, error);
        cumulativeSquaredError += Math.pow(error, 2);
        squaredErrors.put(errorCount, Math.pow(error, 2));
        errorCount++;
    }

    /**
     * Prints classification report.
     *
     */
    public void printReport() {
        System.out.println("Mean absolute error: " + getMeanAbsoluteError());
        System.out.println("Mean squared error: " +getMeanSquaredError());
    }


}
