/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.metrics;

import utils.configurable.DynamicParamException;
import utils.matrix.*;
import utils.sampling.Sequence;

import java.io.Serial;
import java.io.Serializable;
import java.util.Map;
import java.util.TreeMap;

/**
 * Implements functionality for calculation of regression error.
 *
 */
public class RegressionMetric implements Metric, Serializable {

    @Serial
    private static final long serialVersionUID = 1961561346099108396L;

    /**
     * Error count;
     *
     */
    private int errorCount = 0;

    /**
     * Absolute errors.
     *
     */
    private final TreeMap<Integer, Matrix> absoluteErrors = new TreeMap<>();

    /**
     * Cumulative absolute error.
     *
     */
    private Matrix cumulativeAbsoluteError = null;

    /**
     * Squared errors.
     *
     */
    private final TreeMap<Integer, Matrix> squaredErrors = new TreeMap<>();

    /**
     * Cumulative squared error.
     *
     */
    private Matrix cumulativeSquaredError = null;

    /**
     * Predictions for R2 calculation.
     *
     */
    private final TreeMap<Integer, Matrix> predictions = new TreeMap<>();

    /**
     * Actuals for R2 calculation.
     *
     */
    private final TreeMap<Integer, Matrix> actuals = new TreeMap<>();

    /**
     * If true uses R2 as last error otherwise uses MSE.
     *
     */
    private final boolean useR2AsLastError;

    /**
     * Reference to metrics chart.
     *
     */
    private final TrendMetricChart trendMetricChart;

    /**
     * Default constructor for regression metric.
     *
     * @param showMetric if true shows metric otherwise not.
     */
    public RegressionMetric(boolean showMetric) {
        this(true, showMetric);
    }

    /**
     * Constructor for regression metric.
     *
     * @param useR2AsLastError if true uses R2 as last error otherwise uses MSE.
     * @param showMetric if true shows metric otherwise not.
     */
    public RegressionMetric(boolean useR2AsLastError, boolean showMetric) {
        this.useR2AsLastError = useR2AsLastError;
        trendMetricChart = showMetric ? new TrendMetricChart("Neural Network Validation Error", "Step", "Error") : null;
    }

    /**
     * Returns reference metric.
     *
     * @return reference metric.
     */
    public Metric reference() {
        return new RegressionMetric(useR2AsLastError, trendMetricChart != null);
    }

    /**
     * Returns squared error for predicted / actual sample pair.
     *
     * @param predicted predicted sample.
     * @param actual actual (true) sample.
     * @return squared error.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private Matrix getAbsoluteError(Matrix predicted, Matrix actual) throws MatrixException, DynamicParamException {
        return actual.subtract(predicted).apply(UnaryFunctionType.ABS);
    }

    /**
     * Returns squared error for predicted / actual sample pair.
     *
     * @param absoluteError absolute error.
     * @return squared error.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private Matrix getSquaredError(Matrix absoluteError) throws MatrixException, DynamicParamException {
        return absoluteError.power(2);
    }

    /**
     * Reports errors and handles them as either regression or classification errors depending on metrics initialization.
     *
     * @param predicted    predicted errors.
     * @param actual       actual (true) error.
     * @throws MatrixException       throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void report(Sequence predicted, Sequence actual) throws MatrixException, DynamicParamException {
        actuals.clear();
        predictions.clear();
        Matrix currentCumulativeAbsoluteError = null;
        Matrix currentCumulativeSquaredError = null;
        int entryCount = 0;
        for (Map.Entry<Integer, Matrix> entry : predicted.entrySet()) {
            Matrix singleActual = actual.get(entry.getKey());
            actuals.put(++entryCount, singleActual);

            Matrix singlePredicted = entry.getValue();
            predictions.put(entryCount, singlePredicted);

            Matrix absoluteError = getAbsoluteError(singlePredicted, singleActual);
            currentCumulativeAbsoluteError = cumulateError(absoluteError, currentCumulativeAbsoluteError);

            Matrix squaredError = getSquaredError(absoluteError);
            currentCumulativeSquaredError = cumulateError(squaredError, currentCumulativeSquaredError);
        }

        assert currentCumulativeAbsoluteError != null;
        Matrix meanAbsoluteError = currentCumulativeAbsoluteError.divide(predicted.sampleSize());
        Matrix meanSquaredError = currentCumulativeSquaredError.divide(predicted.sampleSize());

        errorCount++;

        absoluteErrors.put(errorCount, meanAbsoluteError);
        cumulativeAbsoluteError = cumulateError(meanAbsoluteError, cumulativeAbsoluteError);

        squaredErrors.put(errorCount, meanSquaredError);
        cumulativeSquaredError = cumulateError(meanSquaredError, cumulativeSquaredError);

        if (trendMetricChart != null) trendMetricChart.addErrorData(errorCount, meanAbsoluteError.mean());
    }

    /**
     * Cumulates error.
     *
     * @param error error.
     * @param cumulativeError cumulative error.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return cumulative error.
     */
    private Matrix cumulateError(Matrix error, Matrix cumulativeError) throws MatrixException {
        if (cumulativeError == null) cumulativeError = error.getNewMatrix();
        cumulativeError.addBy(error);
        return cumulativeError;
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
     * Returns last absolute error.
     *
     * @return last absolute error.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double getLastAbsoluteError() throws MatrixException {
        return absoluteErrors.get(absoluteErrors.lastKey()).mean();
    }

    /**
     * Returns mean absolute error.
     *
     * @return mean absolute error.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double getMeanAbsoluteError() throws MatrixException {
        return getMeanAbsoluteErrorMatrix().mean();
    }

    /**
     * Returns mean absolute error matrix.
     *
     * @return mean absolute error matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix getMeanAbsoluteErrorMatrix() throws MatrixException {
        return cumulativeAbsoluteError.divide(getErrorCount());
    }

    /**
     * Returns mean absolute error.
     *
     * @param lastN calculate for last N errors.
     * @return mean absolute error.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double getMeanAbsoluteError(int lastN) throws MatrixException {
        return getCumulativeErrorMatrix(lastN, absoluteErrors).mean();
    }

    /**
     * Returns last squared error.
     *
     * @return last squared error.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double getLastSquaredError() throws MatrixException {
        return squaredErrors.get(squaredErrors.lastKey()).mean();
    }

    /**
     * Returns mean squared error.
     *
     * @return mean squared error.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double getMeanSquaredError() throws MatrixException {
        return getMeanSquaredErrorMatrix().mean();
    }

    /**
     * Returns mean squared error matrix.
     *
     * @return mean squared error matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix getMeanSquaredErrorMatrix() throws MatrixException {
        return cumulativeSquaredError.divide(getErrorCount());
    }

    /**
     * Returns mean squared error.
     *
     * @param lastN calculate for last N errors.
     * @return mean absolute error.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double getMeanSquaredError(int lastN) throws MatrixException {
        return getCumulativeErrorMatrix(lastN, squaredErrors).mean();
    }

    /**
     * Returns mean cumulative error matrix.
     *
     * @param lastN calculate for last N errors.
     * @param errors errors.
     * @return mean absolute error matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix getCumulativeErrorMatrix(int lastN, TreeMap<Integer, Matrix> errors) throws MatrixException {
        if (lastN < 1) throw new MatrixException("Last N samples cannot be less than 1.");
        Matrix lastNCumulativeError = null;
        int lastNCount = 0;
        for (Integer index : errors.descendingKeySet()) {
            Matrix error = errors.get(index);
            if (lastNCumulativeError == null) lastNCumulativeError = error.getNewMatrix();
            lastNCumulativeError.addBy(error);

            if (++lastNCount >= lastN) break;
        }
        return lastNCumulativeError == null ? null : lastNCumulativeError.divide(lastN);
    }

    /**
     * Returns R2.
     *
     * @return R2.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public double getR2() throws MatrixException, DynamicParamException {
        return errorCount == 0 ? 0 : getR2Matrix().mean();
    }

    /**
     * Returns R2 in matrix form.
     *
     * @return R2 in matrix form.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix getR2Matrix() throws MatrixException, DynamicParamException {
        if (predictions.isEmpty()) return null;
        Matrix meanActualValue = getMeanActualValue();
        Matrix SSRes = null;
        Matrix SSTot = null;
        for (Map.Entry<Integer, Matrix> entry : actuals.entrySet()) {
            int index = entry.getKey();
            Matrix actual = entry.getValue();
            Matrix prediction = predictions.get(index);
            Matrix SSResEntry = actual.subtract(prediction).power(2);
            if (SSRes == null) SSRes = SSResEntry.getNewMatrix();
            SSRes.addBy(SSResEntry);
            Matrix SSTotEntry = actual.subtract(meanActualValue).power(2);
            if (SSTot == null) SSTot = SSTotEntry.getNewMatrix();
            SSTot.addBy(SSTotEntry);
        }
        Matrix ones = new DMatrix(meanActualValue.getRows(), meanActualValue.getColumns(), meanActualValue.getDepth());
        ones.initializeToValue(1);
        return SSTot == null ? null : ones.subtract(SSRes.divide(SSTot));
    }

    /**
     * Return mean of actual values.
     *
     * @return mean of actual values.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Matrix getMeanActualValue() throws MatrixException {
        Matrix meanActualValue = null;
        for (Matrix actual : actuals.values()) {
            if (meanActualValue == null) meanActualValue = actual.getNewMatrix();
            meanActualValue.addBy(actual);
        }
        if (meanActualValue != null) meanActualValue.divideBy(actuals.size());
        return meanActualValue;
    }

    /**
     * Returns last error.
     *
     * @return last error.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public double getLastError() throws MatrixException, DynamicParamException {
        return useR2AsLastError ? getR2() : 1 - absoluteErrors.get(absoluteErrors.lastKey()).mean();
    }

    /**
     * Resets reported sample sets.
     *
     */
    public void reset() {
        actuals.clear();
        predictions.clear();
    }

    /**
     * Reinitialized metric.
     *
     */
    public void reinitialize() {
        reset();
        absoluteErrors.clear();
        cumulativeAbsoluteError = null;
        squaredErrors.clear();
        cumulativeSquaredError = null;
        errorCount = 0;
    }

    /**
     * Prints classification report.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void printReport() throws MatrixException, DynamicParamException {
        System.out.println("Mean absolute error: " + getMeanAbsoluteError());
        System.out.println("Mean squared error: " +getMeanSquaredError());
        System.out.println("R2: " +getR2());
    }

}
