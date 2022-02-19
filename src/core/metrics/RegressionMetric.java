/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.metrics;

import utils.configurable.DynamicParamException;
import utils.matrix.*;
import utils.sampling.Sequence;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashMap;
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
    private final HashMap<Integer, Matrix> absoluteErrors = new HashMap<>();

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
     * Default constructor for regression metric.
     *
     */
    public RegressionMetric() {
        useR2AsLastError = true;
    }

    /**
     * Constructor for regression metric.
     *
     * @param useR2AsLastError if true uses R2 as last error otherwise uses MSE.
     */
    public RegressionMetric(boolean useR2AsLastError) {
        this.useR2AsLastError = useR2AsLastError;
    }

    /**
     * Returns reference metric.
     *
     * @return reference metric.
     */
    public Metric reference() {
        return new RegressionMetric(useR2AsLastError);
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
        return actual.subtract(predicted).power(2);
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
     * Updates regression error for single predicted / actual sample pair.<br>
     *
     * @param predicted predicted sample.
     * @param actual actual (true) sample.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void update(Matrix predicted, Matrix actual) throws MatrixException, DynamicParamException {
        predictions.put(errorCount, predicted);
        actuals.put(errorCount, actual);
        Matrix absoluteError = getAbsoluteError(predicted, actual);
        absoluteErrors.put(errorCount, absoluteError);
        if (cumulativeAbsoluteError == null) cumulativeAbsoluteError = absoluteError;
        else cumulativeAbsoluteError.add(absoluteError, cumulativeAbsoluteError);
        Matrix squaredError = getSquaredError(absoluteError);
        squaredErrors.put(errorCount, squaredError);
        if (cumulativeSquaredError == null) cumulativeSquaredError = squaredError;
        else cumulativeAbsoluteError.add(squaredError, cumulativeSquaredError);
        errorCount++;
    }

    /**
     * Updates regression accuracy for multiple predicted / actual sample pairs.
     *
     * @param predicted predicted samples.
     * @param actual actual (true) samples.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void update(Sequence predicted, Sequence actual) throws MatrixException, DynamicParamException {
        for (Map.Entry<Integer, MMatrix> entry : predicted.entrySet()) {
            int sampleIndex = entry.getKey();
            MMatrix predictedSample = entry.getValue();
            MMatrix actualSample = actual.get(sampleIndex);
            int depth = predictedSample.getDepth();
            for (int depthIndex = 0; depthIndex < depth; depthIndex++) {
                update(predictedSample.get(depthIndex), actualSample.get(depthIndex));
            }
        }
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
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public double getMeanAbsoluteError() throws MatrixException, DynamicParamException {
        return getMeanAbsoluteErrorMatrix().mean();
    }

    /**
     * Returns mean absolute error matrix.
     *
     * @return mean absolute error matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix getMeanAbsoluteErrorMatrix() throws MatrixException, DynamicParamException {
        return cumulativeAbsoluteError.apply(UnaryFunctionType.SQRT).divide(getErrorCount());
    }

    /**
     * Returns mean absolute error.
     *
     * @param lastN calculate for last N errors.
     * @return mean absolute error.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public double getMeanAbsoluteError(int lastN) throws MatrixException, DynamicParamException {
        return getMeanAbsoluteErrorMatrix(lastN).mean();
    }

    /**
     * Returns mean absolute error matrix.
     *
     * @param lastN calculate for last N errors.
     * @return mean absolute error matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix getMeanAbsoluteErrorMatrix(int lastN) throws MatrixException, DynamicParamException {
        if (lastN < 1) return null;
        Matrix lastNCumulativeAbsoluteError = null;
        int lastNCount = 0;
        for (Integer index : actuals.descendingKeySet()) {
            if (lastNCumulativeAbsoluteError == null) lastNCumulativeAbsoluteError = actuals.get(index);
            else lastNCumulativeAbsoluteError.add(absoluteErrors.get(index), lastNCumulativeAbsoluteError);
            if (++lastNCount == lastN) break;
        }
        return lastNCumulativeAbsoluteError == null ? null : lastNCumulativeAbsoluteError.apply(UnaryFunctionType.SQRT).divide(lastN);
    }

    /**
     * Returns mean squared error.
     *
     * @return mean squared error.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public double getMeanSquaredError() throws MatrixException, DynamicParamException {
        return getMeanSquaredErrorMatrix().mean();
    }

    /**
     * Returns mean squared error matrix.
     *
     * @return mean squared error matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix getMeanSquaredErrorMatrix() throws MatrixException, DynamicParamException {
        return cumulativeSquaredError.apply(UnaryFunctionType.SQRT).divide(getErrorCount());
    }

    /**
     * Returns mean squared error.
     *
     * @param lastN calculate for last N errors.
     * @return mean absolute error.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public double getMeanSquaredError(int lastN) throws MatrixException, DynamicParamException {
        return getMeanSquaredErrorMatrix(lastN).mean();
    }

    /**
     * Returns mean squared error matrix.
     *
     * @param lastN calculate for last N errors.
     * @return mean squared error matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix getMeanSquaredErrorMatrix(int lastN) throws MatrixException, DynamicParamException {
        if (lastN < 1) return null;
        Matrix lastNCumulativeSquaredError = null;
        int lastNCount = 0;
        for (Integer index : actuals.descendingKeySet()) {
            if (lastNCumulativeSquaredError == null) lastNCumulativeSquaredError = actuals.get(index);
            else lastNCumulativeSquaredError.add(squaredErrors.get(index), lastNCumulativeSquaredError);
            if (++lastNCount == lastN) break;
        }
        return lastNCumulativeSquaredError == null ? null : lastNCumulativeSquaredError.apply(UnaryFunctionType.SQRT).divide(lastN);
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
        if (errorCount == 0) return null;
        Matrix meanActualValue = getMeanActualValue();
        Matrix SSRes = null;
        Matrix SSTot = null;
        for (Map.Entry<Integer, Matrix> entry : actuals.entrySet()) {
            int index = entry.getKey();
            Matrix actual = entry.getValue();
            Matrix prediction = predictions.get(index);
            Matrix SSResEntry = actual.subtract(prediction).power(2);
            if (SSRes == null) SSRes = SSResEntry;
            else SSRes.add(SSResEntry, SSRes);
            Matrix SSTotEntry = actual.subtract(meanActualValue).power(2);
            if (SSTot == null) SSTot = SSTotEntry;
            else SSTot.add(SSTotEntry, SSTot);
        }
        Matrix ones = new DMatrix(meanActualValue.getRows(), meanActualValue.getColumns());
        ones.initializeToValue(1);
        return SSTot == null || SSRes == null ? null : ones.subtract(SSRes.divide(SSTot));
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
            if (meanActualValue == null) meanActualValue = actual;
            else meanActualValue.add(actual, meanActualValue);
        }
        if (meanActualValue != null) meanActualValue.divide(actuals.size(), meanActualValue);
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
        return useR2AsLastError ? getR2() : Math.sqrt(squaredErrors.get(squaredErrors.lastKey()).mean());
    }

    /**
     * Resets cumulative error and sample count.
     *
     */
    public void reset() {
        absoluteErrors.clear();
        cumulativeAbsoluteError = null;
        squaredErrors.clear();
        cumulativeSquaredError = null;
        actuals.clear();
        predictions.clear();
        errorCount = 0;
    }

    /**
     * Reports errors and handles them as either regression or classification errors depending on metrics initialization.
     *
     * @param predicted predicted errors.
     * @param actual actual (true) error.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void report(Sequence predicted, Sequence actual) throws MatrixException, DynamicParamException {
        update(predicted, actual);
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
