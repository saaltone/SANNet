/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.metrics;

import core.network.NeuralNetworkException;
import utils.DynamicParamException;
import utils.Sequence;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.TreeMap;

/**
 * Implements class that defines metrics for classification and regression.<br>
 * <br>
 * Reference: https://en.wikipedia.org/wiki/Precision_and_recall and https://en.wikipedia.org/wiki/Sensitivity_and_specificity <br>
 *
 */
public class Metrics {

    /**
     * Averaging type for classification.
     *
     */
    public enum AverageType {

        /**
         * Micro average
         *
         */
        MICRO,

        /**
         * Macro average
         *
         */
        MACRO

    }

    /**
     * Class that handles calculation of regression error.
     *
     */
    public static class Regression {

        /**
         * Number of error samples cumulated.
         *
         */
        private int count;

        /**
         * Cumulative error.
         *
         */
        private double cumulativeError;

        /**
         * Predictions for R2 calculation.
         *
         */
        private final LinkedHashMap<Integer, Matrix> predictions = new LinkedHashMap<>();

        /**
         * Actuals for R2 calculation.
         *
         */
        private final LinkedHashMap<Integer, Matrix> actuals = new LinkedHashMap<>();

        /**
         * Returns regression accuracy for predicted / actual sample pair.
         *
         * @param predicted predicted sample.
         * @param actual actual (true) sample.
         * @return regression accuracy.
         * @throws MatrixException throws exception if matrix operation fails.
         * @throws DynamicParamException throws exception if parameter (params) setting fails.
         */
        private double getRegressionAccuracy(Matrix predicted, Matrix actual) throws MatrixException, DynamicParamException {
            return 1 - Math.sqrt(actual.subtract(predicted).power(2).sum());
        }

        /**
         * Updates regression accuracy for single predicted / actual sample pair.<br>
         *
         * @param predicted predicted sample.
         * @param actual actual (true) sample.
         * @throws MatrixException throws exception if matrix operation fails.
         * @throws DynamicParamException throws exception if parameter (params) setting fails.
         */
        public void update(Matrix predicted, Matrix actual) throws MatrixException, DynamicParamException {
            update(1- getRegressionAccuracy(predicted, actual));
        }

        /**
         * Updates regression accuracy for multiple predicted / actual sample pairs.
         *
         * @param predicted predicted samples.
         * @param actual actual (true) samples.
         * @throws MatrixException throws exception if matrix operation fails.
         * @throws DynamicParamException throws exception if parameter (params) setting fails.
         */
        public void update(LinkedHashMap<Integer, Matrix> predicted, LinkedHashMap<Integer, Matrix> actual) throws MatrixException, DynamicParamException {
            double error = 0;
            int size = actual.size();
            for (int sample = 0; sample < size; sample++) {
                error += getRegressionAccuracy(predicted.get(sample), actual.get(sample));
            }
            update(1- error / size);
        }

        /**
         * Updates regression accuracy for multiple predicted / actual sample pairs.
         *
         * @param predicted predicted samples.
         * @param actual actual (true) samples.
         * @throws MatrixException throws exception if matrix operation fails.
         * @throws DynamicParamException throws exception if parameter (params) setting fails.
         */
        public void update(MMatrix predicted, MMatrix actual) throws MatrixException, DynamicParamException {
            double error = 0;
            for (int sample = 0; sample < actual.size(); sample++) {
                error += getRegressionAccuracy(predicted.get(sample), actual.get(sample));
            }
            update(1- error / actual.size());
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
            double error = 0;
            for (Integer sampleIndex : predicted.keySet()) {
                for (Integer matrixIndex : predicted.sampleKeySet()) {
                    error += getRegressionAccuracy(predicted.get(sampleIndex, matrixIndex), actual.get(sampleIndex, matrixIndex));
                }
            }
            update(1- error / actual.totalSize());
            cumulateR2Values(predicted, actual);
        }

        /**
         * Updates cumulative error and increments sample count.
         *
         * @param error error to be cumulated.
         */
        private void update(double error) {
            cumulativeError += error;
            count++;
        }

        /**
         * Cumulates predictions and actuals for R2 values calculation.
         *
         * @param predicted predicted samples.
         * @param actual actual (true) samples.
         */
        private void cumulateR2Values(Sequence predicted, Sequence actual) {
            for (Integer sampleIndex : actual.keySet()) {
                for (Integer matrixIndex : actual.sampleKeySet()) {
                    predictions.put(predictions.size(), predicted.get(sampleIndex, matrixIndex));
                    actuals.put(actuals.size(), actual.get(sampleIndex, matrixIndex));
                }
            }
        }

        /**
         * Returns cumulative error.
         *
         * @return cumulative error.
         */
        public double getCumulativeError() {
            return cumulativeError;
        }

        /**
         * Returns number of error samples cumulated.
         *
         * @return number of error samples cumulated.
         */
        public int getCount() {
            return count;
        }

        /**
         * Returns average error.
         *
         * @return average error.
         */
        public double getAverageError() {
            return count == 0 ? 0 : cumulativeError / (double)count;
        }

        /**
         * Returns and calculates average R2 values.
         *
         * @return average R2 values.
         * @throws MatrixException throws exception if matrix operation fails.
         * @throws DynamicParamException throws exception if parameter (params) setting fails.
         */
        public Matrix getAverageR2Values() throws MatrixException, DynamicParamException {
            if (predictions.isEmpty() || actuals.isEmpty()) return null;
            // Calculate mean of actual values.
            Matrix actualMean = null;
            for (Integer index : actuals.keySet()) {
                Matrix actual = actuals.get(index);
                actualMean = actualMean == null ? actual : actualMean.add(actual);
            }
            if (actualMean == null) return null;
            actualMean.divide(actuals.size(), actualMean);

            // Calculate total sum of squares and sum of squares of residuals.
            Matrix totalSumOfSquares = null;
            Matrix totalSumOfResiduals = null;
            for (Integer index : actuals.keySet()) {
                Matrix actual = actuals.get(index);
                Matrix prediction = predictions.get(index);
                Matrix totalSumOfSquare = actual.subtract(actualMean).power(2);
                totalSumOfSquares = totalSumOfSquares == null ? totalSumOfSquare : totalSumOfSquares.add(totalSumOfSquare);
                Matrix totalSumOfResidual = actual.subtract(prediction).power(2);
                totalSumOfResiduals = totalSumOfResiduals == null ? totalSumOfResidual : totalSumOfResiduals.add(totalSumOfResidual);
            }
            if (totalSumOfSquares == null || totalSumOfResiduals == null) return null;
            for (int row = 0; row < totalSumOfSquares.getRows(); row++) {
                for (int col = 0; col < totalSumOfSquares.getColumns(); col++) {
                    if (totalSumOfSquares.getValue(row, col) == 0) return null;
                }
            }
            predictions.clear();
            actuals.clear();
            return totalSumOfResiduals.divide(totalSumOfSquares).multiply(-1).add(1);
        }

        /**
         * Resets cumulative error and sample count.
         *
         */
        public void reset() {
            cumulativeError = 0;
            count = 0;
        }

    }

    /**
     * Class that handles calculation of classification error.
     *
     */
    public static class Classification {

        /**
         * Features classified.
         *
         */
        private HashSet<Integer> features = new HashSet<>();

        /**
         * True positive counts for each feature.
         *
         */
        private HashMap<Integer, Integer> TP = new HashMap<>();

        /**
         * False positive counts for each feature.
         *
         */
        private HashMap<Integer, Integer> FP = new HashMap<>();

        /**
         * True negative counts for each feature.
         *
         */
        private HashMap<Integer, Integer> TN = new HashMap<>();

        /**
         * False negative counts for each feature.
         *
         */
        private HashMap<Integer, Integer> FN = new HashMap<>();

        /**
         * Total true positive count over all features.
         *
         */
        private int TPTotal;

        /**
         * Total false positive count over all features.
         *
         */
        private int FPTotal;

        /**
         * Total true negative count over all features.
         *
         */
        private int TNTotal;

        /**
         * Total False negative count over all features.
         *
         */
        private int FNTotal;

        /**
         * Confusion matrix.
         *
         */
        HashMap<Integer, HashMap<Integer, Integer>> confusion;

        /**
         * Default constructor for classification class.
         *
         */
        Classification() {}

        /**
         * Updates classification statistics and confusion matrix for a predicted / actual (true) sample pair.
         *
         * @param predicted predicted sample.
         * @param actual actual (true) sample.
         */
        public void update(Matrix predicted, Matrix actual) {
            if (confusion == null) reset();
            for (int predictedRow = 0; predictedRow < predicted.getRows(); predictedRow++) {
                features.add(predictedRow);
                for (int actualRow = 0; actualRow < actual.getRows(); actualRow++) {
                    double actualValue = actual.getValue(predictedRow, 0);
                    double predictedValue = predicted.getValue(actualRow, 0);
                    if (actualValue == 1 && predictedValue == 1) incrementConfusion(predictedRow, actualRow);
                    if (predictedRow == actualRow) {
                        if (actualValue == 1 && predictedValue == 1) incrementTP(predictedRow);
                        if (actualValue == 0 && predictedValue == 0) incrementTN(predictedRow);
                        if (actualValue == 1 && predictedValue == 0) incrementFN(predictedRow);
                        if (actualValue == 0 && predictedValue == 1) incrementFP(predictedRow);
                    }
                }
            }
        }

        /**
         * Increments confusion matrix.
         *
         * @param predictedRow predicted row
         * @param actualRow actual row
         */
        private void incrementConfusion(int predictedRow, int actualRow) {
            if (confusion == null) confusion = new HashMap<>();
            HashMap<Integer, Integer> actuals;
            if (confusion.containsKey(predictedRow)) actuals = confusion.get(predictedRow);
            else {
                actuals = new HashMap<>();
                confusion.put(predictedRow, actuals);
            }
            if (!actuals.containsKey(actualRow)) actuals.put(actualRow, 1);
            else actuals.put(actualRow, actuals.get(actualRow) + 1);
        }

        /**
         * Increments true positive count.
         *
         * @param row row
         */
        private void incrementTP(int row) {
            if (!TP.containsKey(row)) TP.put(row, 1);
            else TP.put(row, TP.get(row) + 1);
            TPTotal++;
        }

        /**
         * Increments true negative count.
         *
         * @param row row
         */
        private void incrementTN(int row) {
            if (!TN.containsKey(row)) TN.put(row, 1);
            else TN.put(row, TN.get(row) + 1);
            TNTotal++;
        }

        /**
         * Increments false negative count.
         *
         * @param row row
         */
        private void incrementFN(int row) {
            if (!FN.containsKey(row)) FN.put(row, 1);
            else FN.put(row, FN.get(row) + 1);
            FNTotal++;
        }

        /**
         * Increments false positive count.
         *
         * @param row row
         */
        private void incrementFP(int row) {
            if (!FP.containsKey(row)) FP.put(row, 1);
            else FP.put(row, FP.get(row) + 1);
            FPTotal++;
        }

        /**
         * Updates classification statistics and confusion matrix for multiple samples.<br>
         *
         * @param predicted predicted samples.
         * @param actual actual (true) samples.
         */
        public void update(MMatrix predicted, MMatrix actual) {
            for (int sample = 0; sample < actual.size(); sample++) {
                update(predicted.get(sample), actual.get(sample));
            }
        }

        /**
         * Updates classification statistics and confusion matrix for multiple samples.
         *
         * @param predicted predicted samples.
         * @param actual actual (true) samples.
         */
        public void update(Sequence predicted, Sequence actual) {
            for (Integer sampleIndex : predicted.keySet()) {
                for (Integer matrixIndex : predicted.sampleKeySet()) {
                    update(predicted.get(sampleIndex, matrixIndex), actual.get(sampleIndex, matrixIndex));
                }
            }
        }

        /**
         * Updates classification statistics and confusion matrix for multiple samples.
         *
         * @param predicted predicted samples.
         * @param actual actual (true) samples.
         */
        public void update(LinkedHashMap<Integer, Matrix> predicted, LinkedHashMap<Integer, Matrix> actual) {
            for (int sample = 0; sample < actual.size(); sample++) {
                update(predicted.get(sample), actual.get(sample));
            }
        }

        /**
         * Resets classification statistics.
         *
         */
        public void reset() {
            features = new HashSet<>();
            TP = new HashMap<>();
            FP = new HashMap<>();
            TN = new HashMap<>();
            FN = new HashMap<>();
            TPTotal = 0;
            FPTotal = 0;
            TNTotal = 0;
            FNTotal = 0;
            confusion = new HashMap<>();
        }

        /**
         * Returns classified features.
         *
         * @return classified features.
         */
        public HashSet<Integer> getFeatures() {
            return features;
        }

        /**
         * Returns true positive statistics.
         *
         * @param feature feature.
         * @return true positive statistics.
         */
        public int getTP(int feature) {
            return TP.getOrDefault(feature, 0);
        }

        /**
         * Returns false positive statistics.
         *
         * @param feature feature.
         * @return false positive statistics.
         */
        public int getFP(int feature) {
            return FP.getOrDefault(feature, 0);
        }

        /**
         * Returns true negative statistics.
         *
         * @param feature feature.
         * @return true negative statistics.
         */
        public int getTN(int feature) {
            return TN.getOrDefault(feature, 0);
        }

        /**
         * Returns false negative statistics.
         *
         * @param feature feature.
         * @return false negative statistics.
         */
        public int getFN(int feature) {
            return FN.getOrDefault(feature, 0);
        }

        /**
         * Returns total true positive count over all features.
         *
         * @return total true positive count.
         */
        public int getTPTotal() {
            return TPTotal;
        }

        /**
         * Returns total false positive count over all features.
         *
         * @return total false positive count.
         */
        public int getFPTotal() {
            return FPTotal;
        }

        /**
         * Returns total true negative count over all features.
         *
         * @return total true negative count.
         */
        public int getTNTotal() {
            return TNTotal;
        }

        /**
         * Returns total false negative count over all features.
         *
         * @return total false negative count.
         */
        public int getFNTotal() {
            return FNTotal;
        }

        /**
         * Returns confusion matrix.
         *
         * @return confusion matrix.
         */
        public HashMap<Integer, HashMap<Integer, Integer>> getConfusion() {
            return confusion;
        }

        /**
         * Returns specific value in confusion matrix.
         *
         * @param predictedRow predicted row.
         * @param actualRow actual row.
         * @return specific value in confusion matrix.
         */
        public int getConfusionValue(int predictedRow, int actualRow) {
            return confusion.get(predictedRow) == null ? 0 : confusion.get(predictedRow).getOrDefault(actualRow, 0);
        }

    }

    /**
     * Metrics type: classification or regression.
     *
     */
    private final MetricsType metricsType;

    /**
     * Reference to classification statistics.
     *
     */
    private Classification classification;

    /**
     * Average type for classification: macro or micro.
     *
     */
    private AverageType averageType = AverageType.MACRO;

    /**
     * If true assumes multi label classification otherwise assumes single label classification.<br>
     * Single label assumes that only one label is true (1) and others false (0). Assumes that max output value takes true value.<br>
     * Multi label assumes that any value above threshold is true (1) otherwise false (0).<br>
     *
     */
    private boolean multiLabel;

    /**
     * Defines threshold value for multi label classification. If value of label is below threshold it is classified as negative (0) otherwise classified as positive (1).
     *
     */
    private double multiLabelThreshold = 0.5;

    /**
     * Reference to regression statistics.
     *
     */
    private Regression regression;

    /**
     * List of recorded errors as error history.
     *
     */
    private TreeMap<Integer, Double> errors = new TreeMap<>();

    /**
     * Average R2 values.
     *
     */
    private Matrix averageR2Values;

    /**
     * Number of errors recorded into error history.
     *
     */
    private int errorHistorySize = 1000;

    /**
     * Constructor for metrics class.
     *
     * @param metricsType metrics type as classification or regression.
     * @throws NeuralNetworkException throws neural network exception if metrics type is not defined.
     */
    public Metrics(MetricsType metricsType) throws NeuralNetworkException {
        if (metricsType == null) throw new NeuralNetworkException("Metrics type must be defined.");
        this.metricsType = metricsType;
        if (metricsType == MetricsType.CLASSIFICATION) classification = new Classification();
        else regression = new Regression();
    }

    /**
     * Constructor for metrics class.
     *
     * @param metricsType metrics type as classification or regression.
     * @param multiLabel if true assumes multi label classification otherwise assumes single label.
     * @throws NeuralNetworkException throws neural network exception if metrics type is not defined.
     */
    public Metrics(MetricsType metricsType, boolean multiLabel) throws NeuralNetworkException {
        this(metricsType);
        this.multiLabel = multiLabel;
    }

    /**
     * Constructor for metrics class.
     *
     * @param metricsType metrics type as classification or regression.
     * @param multiLabel if true assumes multi label classification otherwise assumes single class.
     * @param multiLabelThreshold if class probability is below threshold is it classified as negative (0) otherwise as positive (1).
     * @throws NeuralNetworkException throws neural network exception if metrics type is not defined.
     */
    public Metrics(MetricsType metricsType, boolean multiLabel, double multiLabelThreshold) throws NeuralNetworkException {
        this(metricsType);
        this.multiLabel = multiLabel;
        if (multiLabelThreshold < 0 || multiLabelThreshold > 1) throw new NeuralNetworkException("Multi label threshold must be between 0 ad 1.");
        this.multiLabelThreshold = multiLabelThreshold;
    }

    /**
     * Sets classification average type: macro or micro.
     *
     * @param averageType average type for classification.
     */
    public void setClassificationAverageType(AverageType averageType) {
        this.averageType = averageType;
    }

    /**
     * Resets current error and history.
     *
     */
    public void resetAll() {
        errors = new TreeMap<>();
        resetError();
    }

    /**
     * Resets current error.
     *
     */
    public void resetError() {
        if (metricsType == MetricsType.CLASSIFICATION) {
            regression = null;
            classification = new Classification();
        }
        else {
            regression = new Regression();
            classification = null;
        }
    }

    /**
     * Reports error and handles it as either regression or classification error depending on metrics initialization.
     *
     * @param predicted predicted sample.
     * @param actual actual (true) sample.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void report(Matrix predicted, Matrix actual) throws MatrixException, DynamicParamException {
        if (metricsType == MetricsType.REGRESSION) regression.update(predicted, actual);
        else updateConfusion(predicted, actual);
    }

    /**
     * Reports errors and handles them as either regression or classification errors depending on metrics initialization.
     *
     * @param predicted predicted errors.
     * @param actual actual (true) error.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if reporting of errors fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void report(LinkedHashMap<Integer, Matrix> predicted, LinkedHashMap<Integer, Matrix> actual) throws MatrixException, NeuralNetworkException, DynamicParamException {
        if (metricsType == MetricsType.REGRESSION) regression.update(predicted, actual);
        else updateConfusion(predicted, actual);
    }

    /**
     * Reports errors and handles them as either regression or classification errors depending on metrics initialization.
     *
     * @param predicted predicted errors.
     * @param actual actual (true) error.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if reporting of errors fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void report(MMatrix predicted, MMatrix actual) throws MatrixException, NeuralNetworkException, DynamicParamException {
        if (metricsType == MetricsType.REGRESSION) regression.update(predicted, actual);
        else updateConfusion(predicted, actual);
    }

    /**
     * Reports errors and handles them as either regression or classification errors depending on metrics initialization.
     *
     * @param predicted predicted errors.
     * @param actual actual (true) error.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if reporting of errors fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void report(Sequence predicted, Sequence actual) throws MatrixException, NeuralNetworkException, DynamicParamException {
        if (metricsType == MetricsType.REGRESSION) regression.update(predicted, actual);
        else updateConfusion(predicted, actual);
    }

    /**
     * Reports single error value.
     *
     * @param error single error value to be reported.
     */
    public void report(double error) {
        if (metricsType == MetricsType.REGRESSION) regression.update(error);
    }

    /**
     * Sets size of error history.
     *
     * @param errorHistorySize size of error history.
     */
    public void setErrorHistorySize(int errorHistorySize) {
        this.errorHistorySize = errorHistorySize;
    }

    /**
     * Stores current error to error history.<br>
     * Does not reset current error.<br>
     *
     * @param iteration iteration index for error history.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if calculation of classification accuracy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void store(int iteration) throws MatrixException, NeuralNetworkException, DynamicParamException {
        errors.remove(iteration - errorHistorySize);
        if (metricsType == MetricsType.REGRESSION) {
            errors.put(iteration, regression.getAverageError());
            Matrix currentAverageR2Values = regression.getAverageR2Values();
            if (currentAverageR2Values != null) averageR2Values = averageR2Values == null ? currentAverageR2Values : averageR2Values.multiply(0.9).add(currentAverageR2Values.multiply(0.1));
        }
        else errors.put(iteration, 1 - classificationAccuracy());
    }

    /**
     * Stores current error to error history.
     *
     * @param iteration iteration index for error history.
     * @param reset if true resets current error otherwise not.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if calculation of classification accuracy fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void store(int iteration, boolean reset) throws MatrixException, NeuralNetworkException, DynamicParamException {
        store(iteration);
        if (reset) resetError();
    }

    /**
     * Returns error history.
     *
     * @return error history.
     */
    public TreeMap<Integer, Double> getErrors() {
        return errors;
    }

    /**
     * Returns and calculates moving average of last N iterations in error history.
     *
     * @param lastNIterations number of last iterations.
     * @return moving average of N last iterations.
     */
    public TreeMap<Integer, Double> getMovingAverage(int lastNIterations) {
        TreeMap<Integer, Double> movingAverage = new TreeMap<>();
        if (errors.size() == 0) return new TreeMap<>();
        double currentAverage = 0;
        int start = Integer.MIN_VALUE;
        for (Integer index : errors.descendingKeySet()) {
            if (start == Integer.MIN_VALUE) start = index;
            currentAverage = start == index ? errors.get(index) : 0.9 * currentAverage + 0.1 * errors.get(index);
            movingAverage.put(index, currentAverage);
            if (index <= start - lastNIterations) break;
        }
        return movingAverage;
    }

    /**
     * Checks if specific iteration index exists in error history.
     *
     * @param iteration iteration index.
     * @return true if iteration index exists in error history otherwise false.
     */
    public boolean valueExists(int iteration) {
        return errors.containsKey(iteration);
    }

    /**
     * Returns last absolute error value in error history.
     *
     * @return last absolute error value in error history.
     */
    public double getAbsolute() {
        return errors.lastEntry() != null ? errors.lastEntry().getValue() : 0;
    }

    /**
     * Returns absolute error value of specific iteration in error history.<br>
     * Returns 0 if iteration index does not exist in error history.<br>
     *
     * @param iteration iteration index.
     * @return absolute error value of specific iteration in error history.
     */
    public double getAbsolute(int iteration) {
        return valueExists(iteration) ? errors.get(iteration) : 0;
    }

    /**
     * Get average error of last N iterations.
     *
     * @param lastNIterations number of iterations to be included into average error.
     * @return average error of last N iteration.
     */
    public double getAverage(int lastNIterations) {
        if (errors.size() == 0) return 0;
        double currentAverage = 0;
        int count = 0;
        int start = Integer.MIN_VALUE;
        for (Integer index : errors.descendingKeySet()) {
            if (start == Integer.MIN_VALUE) start = index;
            currentAverage += errors.get(index);
            count++;
            if (index <= start - lastNIterations && lastNIterations != -1) break;
        }
        return currentAverage / (double)count;
    }

    /**
     * Returns minimum error in error history.
     *
     * @return minimum error in error history.
     */
    public double getMin() {
        return getMin(errors.size());
    }

    /**
     * Returns minimum error over last N iterations in error history.
     *
     * @param lastNIterations number of last N iterations to be included into minimum counting.
     * @return minimum error over last N iterations in error history.
     */
    public double getMin(int lastNIterations) {
        if (errors.size() == 0) return 0;
        double min = Double.POSITIVE_INFINITY;
        int start = Integer.MIN_VALUE;
        for (Integer index : errors.descendingKeySet()) {
            if (start == Integer.MIN_VALUE) start = index;
            min = Math.min(min, errors.get(index));
            if (index <= start - lastNIterations) break;
        }
        return min;
    }

    /**
     * Returns maximum error in error history.
     *
     * @return maximum error in error history.
     */
    public double getMax() {
        return getMax(errors.size());
    }

    /**
     * Returns maximum error over last N iterations in error history.
     *
     * @param lastNIterations number of last N iterations to be included into maximum counting.
     * @return maximum error over last N iterations in error history.
     */
    public double getMax(int lastNIterations) {
        if (errors.size() == 0) return 0;
        double max = Double.NEGATIVE_INFINITY;
        int start = Integer.MIN_VALUE;
        for (Integer index : errors.descendingKeySet()) {
            if (start == Integer.MIN_VALUE) start = index;
            max = Math.max(max, errors.get(index));
            if (index <= start - lastNIterations) break;
        }
        return max;
    }
    /**
     * Returns average R2 values.
     *
     * @return average R2 values.
     */
    public Matrix getAverageR2Values() {
        return averageR2Values;
    }

    /**
     * Returns classification for (predicted) sample.<br>
     * Takes into consideration if single label or multi label classification for metrics is defined.<br>
     *
     * @param predicted predicted sample.
     * @return classification for predicted sample.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Matrix getClassification(Matrix predicted) throws MatrixException {
        if (!multiLabel) {
            double maxValue = predicted.max();
            Matrix.MatrixUnaryOperation classification = (value) -> value != maxValue ? 0 : 1;
            return predicted.apply(classification);
        }
        else {
            Matrix.MatrixUnaryOperation classification = (value) -> value < multiLabelThreshold ? 0 : 1;
            return predicted.apply(classification);
        }
    }

    /**
     * Returns classification for (predicted) multiple samples.<br>
     * Takes into consideration if single label or multi label classification for metrics is defined.<br>
     *
     * @param predicted predicted samples.
     * @return classification for predicted samples.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private MMatrix getClassification(MMatrix predicted) throws MatrixException {
        MMatrix classified = new MMatrix();
        int index = 0;
        for (Matrix sample: predicted.values()) classified.put(index++, getClassification(sample));
        return classified;
    }

    /**
     * Returns classification for (predicted) multiple samples.<br>
     * Takes into consideration if single label or multi label classification for metrics is defined.<br>
     *
     * @param predicted predicted samples.
     * @return classification for predicted samples.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private LinkedHashMap<Integer, Matrix> getClassification(LinkedHashMap<Integer, Matrix> predicted) throws MatrixException {
        LinkedHashMap<Integer, Matrix> classified = new LinkedHashMap<>();
        int index = 0;
        for (Matrix sample: predicted.values()) classified.put(index++, getClassification(sample));
        return classified;
    }

    /**
     * Returns classification for (predicted) multiple samples.<br>
     * Takes into consideration if single label or multi label classification for metrics is defined.<br>
     *
     * @param predicted predicted samples.
     * @return classification for predicted samples.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Sequence getClassification(Sequence predicted) throws MatrixException {
        Sequence classified = new Sequence(predicted.getDepth());
        for (Integer sampleIndex : predicted.keySet()) {
            for (Integer matrixIndex : predicted.sampleKeySet()) {
                classified.put(sampleIndex, matrixIndex, getClassification(predicted.get(sampleIndex, matrixIndex)));
            }
        }
        return classified;
    }

    /**
     * Updates confusion and classification statistics by including new predicted / actual (true) sample pair.
     *
     * @param predicted predicted sample.
     * @param actual actual (true) sample.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private void updateConfusion(Matrix predicted, Matrix actual) throws MatrixException {
        predicted = getClassification(predicted);
        classification.update(predicted, actual);
    }

    /**
     * Updates confusion and classification statistics by including multiple new predicted / actual (true) sample pairs.<br>
     *
     * @param predicted predicted samples.
     * @param actual actual (true) samples.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if classification statistics update fails.
     */
    private void updateConfusion(MMatrix predicted, MMatrix actual) throws MatrixException, NeuralNetworkException {
        if (actual.size() == 0) throw new NeuralNetworkException("Nothing to classify");
        predicted = getClassification(predicted);
        classification.update(predicted, actual);
    }

    /**
     * Updates confusion and classification statistics by including multiple new predicted / actual (true) sample pairs.<br>
     *
     * @param predicted predicted samples.
     * @param actual actual (true) samples.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if classification statistics update fails.
     */
    private void updateConfusion(Sequence predicted, Sequence actual) throws MatrixException, NeuralNetworkException {
        if (actual.sampleSize() == 0) throw new NeuralNetworkException("Nothing to classify");
        predicted = getClassification(predicted);
        classification.update(predicted, actual);
    }

    /**
     * Updates confusion and classification statistics by including multiple new predicted / actual (true) sample pairs.<br>
     *
     * @param predicted predicted samples.
     * @param actual actual (true) samples.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if classification statistics update fails.
     */
    private void updateConfusion(LinkedHashMap<Integer, Matrix> predicted, LinkedHashMap<Integer, Matrix> actual) throws MatrixException, NeuralNetworkException {
        if (actual.size() == 0) throw new NeuralNetworkException("Nothing to classify");
        predicted = getClassification(predicted);
        classification.update(predicted, actual);
    }

    /**
     * Returns classification metrics.
     *
     * @return classification metrics.
     * @throws NeuralNetworkException throws exception is metrics is not defined as classification type.
     */
    public Classification getClassificationMetrics() throws NeuralNetworkException {
        if (metricsType != MetricsType.CLASSIFICATION) throw new NeuralNetworkException("Not classification metric.");
        return classification;
    }

    /**
     * Returns classification accuracy.<br>
     * Accuracy is calculated as (TP + TN) / (TP + FP + TN + FN).<br>
     * Takes into consideration if statistics is defined as macro or micro average.<br>
     *
     * @return classification accuracy.
     * @throws NeuralNetworkException throws exception if metrics is not defined as classification type.
     */
    public double classificationAccuracy() throws NeuralNetworkException {
        if (metricsType != MetricsType.CLASSIFICATION) throw new NeuralNetworkException("Not classification metric.");
        if (averageType == AverageType.MACRO) {
            double average = 0;
            int averageCount = 0;
            for (Integer feature : classification.getFeatures()) {
                double TP = classification.getTP(feature);
                double FP = classification.getFP(feature);
                double TN = classification.getTN(feature);
                double FN = classification.getFN(feature);
                if (FP + TN + FP + FN > 0) {
                    average += (TP + TN) / (TP + FP + TN + FN);
                    averageCount++;
                }
            }
            return averageCount == 0 ? 0 : average / (double)averageCount;
        }
        else {
            return (double)(classification.getTPTotal() + classification.getTNTotal()) / (double)(classification.getTPTotal() + classification.getFPTotal() + classification.getTNTotal() + classification.getFNTotal());
        }
    }

    /**
     * Returns classification error rate.<br>
     * Accuracy is calculated as (FP + FN) / (TP + FP + TN + FN).<br>
     * Takes into consideration if statistics is defined as macro or micro average.<br>
     *
     * @return classification error rate.
     * @throws NeuralNetworkException throws exception if metrics is not defined as classification type.
     */
    public double classificationErrorRate() throws NeuralNetworkException {
        if (metricsType != MetricsType.CLASSIFICATION) throw new NeuralNetworkException("Not classification metric.");
        if (averageType == AverageType.MACRO) {
            double average = 0;
            int averageCount = 0;
            for (Integer feature : classification.getFeatures()) {
                double TP = classification.getTP(feature);
                double FP = classification.getFP(feature);
                double TN = classification.getTN(feature);
                double FN = classification.getFN(feature);
                if (FP + TN + FP + FN > 0) {
                    average += (FP + FN) / (TP + FP + TN + FN);
                    averageCount++;
                }
            }
            return averageCount == 0 ? 0 : average / (double)averageCount;
        }
        else {
            return (double)(classification.getFPTotal() + classification.getFNTotal()) / (double)(classification.getTPTotal() + classification.getFPTotal() + classification.getTNTotal() + classification.getFNTotal());
        }
    }

    /**
     * Returns classification precision (positive predictive value).<br>
     * Precision is calculated as TP / (TP + FP).<br>
     * Measures share of correctly classified positive samples out of all samples predicted as positive.<br>
     * Takes into consideration if statistics is defined as macro or micro average.<br>
     *
     * @return classification precision.
     * @throws NeuralNetworkException throws exception if metrics is not defined as classification type.
     */
    public double classificationPrecision() throws NeuralNetworkException {
        if (metricsType != MetricsType.CLASSIFICATION) throw new NeuralNetworkException("Not classification metric.");
        if (averageType == AverageType.MACRO) {
            double average = 0;
            int averageCount = 0;
            for (Integer feature : classification.getFeatures()) {
                double TP = classification.getTP(feature);
                double FP = classification.getFP(feature);
                if (TP + FP > 0) {
                    average += TP / (TP + FP);
                    averageCount++;
                }
            }
            return averageCount == 0 ? 0 : average / (double)averageCount;
        }
        else {
            return (double)(classification.getTPTotal()) / (double)(classification.getTPTotal() + classification.getFPTotal());
        }
    }

    /**
     * Returns classification recall (sensitivity, hit rate, true positive rate).<br>
     * Recall is calculated as TP / (TP + FN).<br>
     * Measures share of correctly classified positive samples out of all samples actually positive.<br>
     * Takes into consideration if statistics is defined as macro or micro average.<br>
     *
     * @return classification recall.
     * @throws NeuralNetworkException throws exception if metrics is not defined as classification type.
     */
    public double classificationRecall() throws NeuralNetworkException {
        if (metricsType != MetricsType.CLASSIFICATION) throw new NeuralNetworkException("Not classification metric.");
        if (averageType == AverageType.MACRO) {
            double average = 0;
            int averageCount = 0;
            for (Integer feature : classification.getFeatures()) {
                double TP = classification.getTP(feature);
                double FN = classification.getFN(feature);
                if (TP + FN > 0) {
                    average += TP / (TP + FN);
                    averageCount++;
                }
            }
            return averageCount == 0 ? 0 : average / (double)averageCount;
        }
        else {
            return (double)(classification.getTPTotal()) / (double)(classification.getTPTotal() + classification.getFNTotal());
        }
    }

    /**
     * Returns classification specificity (selectivity, true negative rate).<br>
     * Specificity is calculated as TN / (TN + FP).<br>
     * Measures share of correctly classified negative samples out of all samples actually negative.<br>
     * Takes into consideration if statistics is defined as macro or micro average.<br>
     *
     * @return classification specificity.
     * @throws NeuralNetworkException throws exception if metrics is not defined as classification type.
     */
    public double classificationSpecificity() throws NeuralNetworkException {
        if (metricsType != MetricsType.CLASSIFICATION) throw new NeuralNetworkException("Not classification metric.");
        if (averageType == AverageType.MACRO) {
            double average = 0;
            int averageCount = 0;
            for (Integer feature : classification.getFeatures()) {
                double FP = classification.getFP(feature);
                double TN = classification.getTN(feature);
                if (TN + FP > 0) {
                    average += TN / (TN + FP);
                    averageCount++;
                }
            }
            return averageCount == 0 ? 0 : average / (double)averageCount;
        }
        else {
            return (double)(classification.getTNTotal()) / (double)(classification.getTNTotal() + classification.getFPTotal());
        }
    }

    /**
     * Returns classification F1 score.<br>
     * F1 is calculated as 2 * precision * recall / (precision + recall).<br>
     * Takes into consideration if statistics is defined as macro or micro average.<br>
     *
     * @return classification F1 score.
     * @throws NeuralNetworkException throws exception if metrics is not defined as classification type.
     */
    public double classificationF1Score() throws NeuralNetworkException {
        if (metricsType != MetricsType.CLASSIFICATION) throw new NeuralNetworkException("Not classification metric.");
        if (averageType == AverageType.MACRO) {
            double average = 0;
            int averageCount = 0;
            for (Integer feature : classification.getFeatures()) {
                double TP = classification.getTP(feature);
                double FP = classification.getFP(feature);
                double FN = classification.getFN(feature);
                double precision = TP / (TP + FP);
                double recall =  TP / (TP + FN);
                if (precision + recall > 0) {
                    average += 2 * precision * recall / (precision + recall);
                    averageCount++;
                }
            }
            return averageCount == 0 ? 0 : average / (double)averageCount;
        }
        else {
            double TP = classification.getTPTotal();
            double FP = classification.getFPTotal();
            double FN = classification.getFNTotal();
            double precision = TP / (TP + FP);
            double recall =  TP / (TP + FN);
            return precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
        }
    }

    /**
     * Returns confusion matrix.
     *
     * @return confusion matrix.
     * @throws NeuralNetworkException throws exception if metrics is not defined as classification type.
     */
    public HashMap<Integer, HashMap<Integer, Integer>> confusionMatrix() throws NeuralNetworkException {
        if (metricsType != MetricsType.CLASSIFICATION) throw new NeuralNetworkException("Not classification metric.");
        return classification.getConfusion();
    }

    /**
     * Prints classification report.
     *
     * @throws NeuralNetworkException throws exception if metrics is not defined as classification type.
     */
    public void printReport() throws NeuralNetworkException {
        if (metricsType == MetricsType.CLASSIFICATION) {
            System.out.println("Classification report:");
            System.out.println("  Accuracy: " + classificationAccuracy());
            System.out.println("  Precision: " + classificationPrecision());
            System.out.println("  Recall: " + classificationRecall());
            System.out.println("  Specificity: " + classificationSpecificity());
            System.out.println("  F1 Score: " + classificationF1Score());
            printConfusionMatrix();
        }
        else {
            System.out.println("Regression accuracy: " + (1 - getAverage(-1)));
        }
    }

    /**
     * Prints confusion matrix.
     *
     */
    public void printConfusionMatrix() {
        if (metricsType != MetricsType.CLASSIFICATION) {
            System.out.println("Not classification metric.");
            return;
        }
        System.out.println("Confusion matrix (actual value as rows, predicted value as columns):");
        for (Integer predictedRow : classification.getFeatures()) {
            System.out.print("[");
            int index = 0;
            for (Integer actualRow : classification.getFeatures()) {
                System.out.print(classification.getConfusionValue(predictedRow, actualRow));
                if (index++ < classification.getFeatures().size() - 1) System.out.print(" ");
            }
            System.out.println("]");
        }
    }

}
