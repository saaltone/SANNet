/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.metrics;

import core.NeuralNetworkException;
import utils.Sequence;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.LinkedHashMap;
import java.util.TreeMap;

/**
 * Implements class that defines basic metrics for classification and regression.<br>
 * <br>
 * Reference: https://en.wikipedia.org/wiki/Precision_and_recall and https://en.wikipedia.org/wiki/Sensitivity_and_specificity<br>
 *
 */
public class Metrics {

    /**
     * Averaging type for classification as micro or macro.
     *
     */
    public enum AverageType {
        MICRO,
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
         */
        private double getRegressionAccuracy(Matrix predicted, Matrix actual) throws MatrixException {
            return 1 - Math.sqrt(actual.subtract(predicted).power(2).sum());
        }

        /**
         * Updates regression accuracy for single predicted / actual sample pair.<br>
         *
         * @param predicted predicted sample.
         * @param actual actual (true) sample.
         * @throws MatrixException throws exception if matrix operation fails.
         */
        public void update(Matrix predicted, Matrix actual) throws MatrixException {
            update(1- getRegressionAccuracy(predicted, actual));
        }

        /**
         * Updates regression accuracy for multiple predicted / actual sample pairs.<br>
         * Assumes hash map structure for samples.<br>
         *
         * @param predicted predicted samples.
         * @param actual actual (true) samples.
         * @throws MatrixException throws exception if matrix operation fails.
         */
        public void update(LinkedHashMap<Integer, Matrix> predicted, LinkedHashMap<Integer, Matrix> actual) throws MatrixException {
            double error = 0;
            for (int sample = 0; sample < actual.size(); sample++) {
                error += getRegressionAccuracy(predicted.get(sample), actual.get(sample));
            }
            update(1- error / actual.size());
        }

        /**
         * Updates regression accuracy for multiple predicted / actual sample pairs.<br>
         * Assumes tree map structure for samples.<br>
         *
         * @param predicted predicted samples.
         * @param actual actual (true) samples.
         * @throws MatrixException throws exception if matrix operation fails.
         */
        public void update(MMatrix predicted, MMatrix actual) throws MatrixException {
            double error = 0;
            for (int sample = 0; sample < actual.size(); sample++) {
                error += getRegressionAccuracy(predicted.get(sample), actual.get(sample));
            }
            update(1- error / actual.size());
        }

        /**
         * Updates regression accuracy for multiple predicted / actual sample pairs.<br>
         * Assumes sequence for samples.<br>
         *
         * @param predicted predicted samples.
         * @param actual actual (true) samples.
         * @throws MatrixException throws exception if matrix operation fails.
         */
        public void update(Sequence predicted, Sequence actual) throws MatrixException {
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
         * Cumulates predictions and actuals R2 values calculation.
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
         */
        public Matrix getAverageR2Values() throws MatrixException {
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
         * True positive counts for each feature.
         *
         */
        private int[] TP;

        /**
         * False positive counts for each feature.
         *
         */
        private int[] FP;

        /**
         * True negative counts for each feature.
         *
         */
        private int[] TN;

        /**
         * False negative counts for each feature.
         *
         */
        private int[] FN;

        /**
         * Total true positive count over all features.
         *
         */
        private int TPTot;

        /**
         * Total false positive count over all features.
         *
         */
        private int FPTot;

        /**
         * Total true negative count over all features.
         *
         */
        private int TNTot;

        /**
         * Total False negative count over all features.
         *
         */
        private int FNTot;

        /**
         * Confusion matrix.
         *
         */
        int[][] confusion;

        /**
         * Default constructor for classification class.
         *
         */
        Classification() {}

        /**
         * Updates classification statistics and confusion matrix for a sample.
         *
         * @param predicted predicted sample.
         * @param actual actual (true) sample.
         */
        public void update(Matrix predicted, Matrix actual) {
            if (confusion == null) reset (actual.getRows());
            for (int predictedRow = 0; predictedRow < predicted.getRows(); predictedRow++) {
                for (int actualRow = 0; actualRow < actual.getRows(); actualRow++) {
                    double actualValue = actual.getValue(predictedRow, 0);
                    double predictedValue = predicted.getValue(actualRow, 0);
                    if (actualValue == 1 && predictedValue == 1) confusion[predictedRow][actualRow]++;
                    if (predictedRow == actualRow) {
                        if (actualValue == 1 && predictedValue == 1) { TP[predictedRow]++; TPTot++; }
                        if (actualValue == 0 && predictedValue == 0) { TN[predictedRow]++; TNTot++; }
                        if (actualValue == 1 && predictedValue == 0) { FN[predictedRow]++; FNTot++; }
                        if (actualValue == 0 && predictedValue == 1) { FP[predictedRow]++; FPTot++; }
                    }
                }
            }
        }

        /**
         * Updates classification statistics and confusion matrix for multiple samples.<br>
         * Assumes tree map structure for samples.<br>
         *
         * @param predicted predicted samples.
         * @param actual actual (true) samples.
         * @throws MatrixException throws exception if matrix operation fails.
         */
        public void update(MMatrix predicted, MMatrix actual) throws MatrixException {
            if (confusion == null) reset (actual.values().toArray(new Matrix[0])[0].getRows());
            else if (confusion.length != actual.values().toArray(new Matrix[0])[0].getRows()) throw new MatrixException("Classification and sample feature amounts do not match");
            for (int sample = 0; sample < actual.size(); sample++) {
                update(predicted.get(sample), actual.get(sample));
            }
        }

        /**
         * Updates classification statistics and confusion matrix for multiple samples.<br>
         * Assumes sequence for samples.<br>
         *
         * @param predicted predicted samples.
         * @param actual actual (true) samples.
         * @throws MatrixException throws exception if matrix operation fails.
         */
        public void update(Sequence predicted, Sequence actual) throws MatrixException {
            if (confusion == null) reset (actual.firstValue().get(0).getRows());
            else if (confusion.length != actual.firstValue().get(0).getRows()) throw new MatrixException("Classification and sample feature amounts do not match");
            for (Integer sampleIndex : predicted.keySet()) {
                for (Integer matrixIndex : predicted.sampleKeySet()) {
                    update(predicted.get(sampleIndex, matrixIndex), actual.get(sampleIndex, matrixIndex));
                }
            }
        }

        /**
         * Updates classification statistics and confusion matrix for multiple samples.<br>
         * Assumes hash map structure for samples.<br>
         *
         * @param predicted predicted samples.
         * @param actual actual (true) samples.
         * @throws MatrixException throws exception if matrix operation fails.
         */
        public void update(LinkedHashMap<Integer, Matrix> predicted, LinkedHashMap<Integer, Matrix> actual) throws MatrixException {
            if (confusion == null) reset (actual.values().toArray(new Matrix[0])[0].getRows());
            else if (confusion.length != actual.values().toArray(new Matrix[0])[0].getRows()) throw new MatrixException("Classification and sample feature amounts do not match");
            for (int sample = 0; sample < actual.size(); sample++) {
                update(predicted.get(sample), actual.get(sample));
            }
        }

        /**
         * Resets classification statistics.
         *
         * @param featureAmount number of samples for statistics after reset.
         */
        public void reset(int featureAmount) {
            TP = new int[featureAmount];
            FP = new int[featureAmount];
            TN = new int[featureAmount];
            FN = new int[featureAmount];
            TPTot = 0;
            FPTot = 0;
            TNTot = 0;
            FNTot = 0;
            confusion = new int[featureAmount][featureAmount];
        }

        /**
         * Returns true positive statistics.
         *
         * @return true positive statistics.
         */
        public int[] getTP() {
            return TP;
        }

        /**
         * Returns false positive statistics.
         *
         * @return false positive statistics.
         */
        public int[] getFP() {
            return FP;
        }

        /**
         * Returns true negative statistics.
         *
         * @return true negative statistics.
         */
        public int[] getTN() {
            return TN;
        }

        /**
         * Returns false negative statistics.
         *
         * @return false negative statistics.
         */
        public int[] getFN() {
            return FN;
        }

        /**
         * Returns total true positive count over all features.
         *
         * @return total true positive count.
         */
        public int getTPTot() {
            return TPTot;
        }

        /**
         * Returns total false positive count over all features.
         *
         * @return total false positive count.
         */
        public int getFPTot() {
            return FPTot;
        }

        /**
         * Returns total true negative count over all features.
         *
         * @return total true negative count.
         */
        public int getTNTot() {
            return TNTot;
        }

        /**
         * Returns total false negative count over all features.
         *
         * @return total false negative count.
         */
        public int getFNTot() {
            return FNTot;
        }

        /**
         * Returns confusion matrix.
         *
         * @return confusion matrix.
         */
        public int[][] getConfusion() {
            return confusion;
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
     * If true assumes multi class classification otherwise assumes single class classification.<br>
     * Single class assumes that only one class is true (1) and others false (0). Assumes that max output value takes true value.<br>
     * Multi class assumes that any value above 0.5 is true (1) otherwise false (0).<br>
     *
     */
    private boolean multiClass;

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
     * @param multiClass if true assumes multi class classification otherwise assumes single class.
     * @throws NeuralNetworkException throws neural network exception if metrics type is not defined.
     */
    public Metrics(MetricsType metricsType, boolean multiClass) throws NeuralNetworkException {
        this(metricsType);
        this.multiClass = multiClass;
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
     */
    public void report(Matrix predicted, Matrix actual) throws MatrixException {
        if (metricsType == MetricsType.REGRESSION) regression.update(predicted, actual);
        else updateConfusion(predicted, actual);
    }

    /**
     * Reports errors and handles them as either regression or classification errors depending on metrics initialization.<br>
     * Assumes hash map structure for samples.<br>
     *
     * @param predicted predicted errors.
     * @param actual actual (true) error.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if reporting of errors fails.
     */
    public void report(LinkedHashMap<Integer, Matrix> predicted, LinkedHashMap<Integer, Matrix> actual) throws MatrixException, NeuralNetworkException {
        if (metricsType == MetricsType.REGRESSION) regression.update(predicted, actual);
        else updateConfusion(predicted, actual);
    }

    /**
     * Reports errors and handles them as either regression or classification errors depending on metrics initialization.<br>
     * Assumes tree map structure for samples.<br>
     *
     * @param predicted predicted errors.
     * @param actual actual (true) error.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if reporting of errors fails.
     */
    public void report(MMatrix predicted, MMatrix actual) throws MatrixException, NeuralNetworkException {
        if (metricsType == MetricsType.REGRESSION) regression.update(predicted, actual);
        else updateConfusion(predicted, actual);
    }

    /**
     * Reports errors and handles them as either regression or classification errors depending on metrics initialization.<br>
     * Assumes sequence for samples.<br>
     *
     * @param predicted predicted errors.
     * @param actual actual (true) error.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if reporting of errors fails.
     */
    public void report(Sequence predicted, Sequence actual) throws MatrixException, NeuralNetworkException {
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
     */
    public void store(int iteration) throws MatrixException, NeuralNetworkException {
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
     */
    public void store(int iteration, boolean reset) throws MatrixException, NeuralNetworkException {
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
        double curAvg = 0;
        int start = Integer.MIN_VALUE;
        for (Integer index : errors.descendingKeySet()) {
            if (start == Integer.MIN_VALUE) start = index;
            curAvg = start == index ? errors.get(index) : 0.9 * curAvg + 0.1 * errors.get(index);
            movingAverage.put(index, curAvg);
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
     * Get average error of last N iteration.
     *
     * @param lastNIterations number of iterations to be included into average error.
     * @return average error of last N iteration.
     */
    public double getAverage(int lastNIterations) {
        if (errors.size() == 0) return 0;
        double curAvg = 0;
        int count = 0;
        int start = Integer.MIN_VALUE;
        for (Integer index : errors.descendingKeySet()) {
            if (start == Integer.MIN_VALUE) start = index;
            curAvg += errors.get(index);
            count++;
            if (index <= start - lastNIterations && lastNIterations != -1) break;
        }
        return curAvg / (double)count;
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
        double min = Double.MAX_VALUE;
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
        double max = Double.MIN_VALUE;
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
     * Takes into consideration if single class or multi class classification for metrics is defined.<br>
     *
     * @param predicted predicted sample.
     * @return classification for predicted sample.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Matrix getClassification(Matrix predicted) throws MatrixException {
        if (!multiClass) {
            double maxValue = predicted.max();
            Matrix.MatrixUnaryOperation classification = (value) -> value != maxValue ? 0 : 1;
            return predicted.apply(classification);
        }
        else {
            Matrix.MatrixUnaryOperation classification = (value) -> value < 0.5 ? 0 : 1;
            return predicted.apply(classification);
        }
    }

    /**
     * Returns classification for (predicted) multiple samples.<br>
     * Takes into consideration if single class or multi class classification for metrics is defined.<br>
     * Assumes tree map structure for samples.<br>
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
     * Takes into consideration if single class or multi class classification for metrics is defined.<br>
     * Assumes hash map structure for samples.<br>
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
     * Takes into consideration if single class or multi class classification for metrics is defined.<br>
     * Assumes sequence for samples.<br>
     *
     * @param predicted predicted samples.
     * @return classification for predicted samples.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Sequence getClassification(Sequence predicted) throws MatrixException {
        Sequence classified = new Sequence(predicted.getDepth());
        int index = 0;
        for (Integer sampleIndex : predicted.keySet()) {
            for (Integer matrixIndex : predicted.sampleKeySet()) {
                classified.put(sampleIndex, matrixIndex, getClassification(predicted.get(sampleIndex, matrixIndex)));
            }
        }
        return classified;
    }

    /**
     * Updates confusion and classification statistics by including new predicted / actual sample pair.
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
     * Updates confusion and classification statistics by including multiple new predicted / actual sample pairs.<br>
     * Assumes tree map structure for samples.<br>
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
     * Updates confusion and classification statistics by including multiple new predicted / actual sample pairs.<br>
     * Assumes sequence for samples.<br>
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
     * Updates confusion and classification statistics by including multiple new predicted / actual sample pairs.<br>
     * Assumes hash map structure for samples.<br>
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
            double avg = 0;
            for (int item = 0; item < classification.getTP().length; item++) {
                if ((double)(classification.getTP()[item] + classification.getFP()[item] + classification.getTN()[item] + classification.getFN()[item]) != 0) {
                    avg += (double)(classification.getTP()[item] + classification.getTN()[item]) / (double)(classification.getTP()[item] + classification.getFP()[item] + classification.getTN()[item] + classification.getFN()[item]);
                }
            }
            return avg / (double)classification.getTP().length;
        }
        else {
            return (double)(classification.getTPTot() + classification.getTNTot()) / (double)(classification.getTPTot() + classification.getFPTot() + classification.getTNTot() + classification.getFNTot());
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
            double avg = 0;
            for (int item = 0; item < classification.getTP().length; item++) {
                if ((double)(classification.getTP()[item] + classification.getFP()[item] + classification.getTN()[item] + classification.getFN()[item]) != 0) {
                    avg += (double)(classification.getFP()[item] + classification.getFN()[item]) / (double)(classification.getTP()[item] + classification.getFP()[item] + classification.getTN()[item] + classification.getFN()[item]);
                }
            }
            return avg / (double)classification.getTP().length;
        }
        else {
            return (double)(classification.getFPTot() + classification.getFNTot()) / (double)(classification.getTPTot() + classification.getFPTot() + classification.getTNTot() + classification.getFNTot());
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
            double avg = 0;
            for (int item = 0; item < classification.getTP().length; item++) {
                if ((double)(classification.getTP()[item] + classification.getFP()[item]) != 0) {
                    avg += (double)(classification.getTP()[item]) / (double)(classification.getTP()[item] + classification.getFP()[item]);
                }
            }
            return avg / (double)classification.getTP().length;
        }
        else {
            return (double)(classification.getTPTot()) / (double)(classification.getTPTot() + classification.getFPTot());
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
            double avg = 0;
            for (int item = 0; item < classification.getTP().length; item++) {
                if ((double)(classification.getTP()[item] + classification.getFN()[item]) != 0) {
                    avg += (double)(classification.getTP()[item]) / (double)(classification.getTP()[item] + classification.getFN()[item]);
                }
            }
            return avg / (double)classification.getTP().length;
        }
        else {
            return (double)(classification.getTPTot()) / (double)(classification.getTPTot() + classification.getFNTot());
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
            double avg = 0;
            for (int item = 0; item < classification.getTN().length; item++) {
                if ((double)(classification.getTN()[item] + classification.getFP()[item]) != 0) {
                    avg += (double)(classification.getTN()[item]) / (double)(classification.getTN()[item] + classification.getFP()[item]);
                }
            }
            return avg / (double)classification.getTN().length;
        }
        else {
            return (double)(classification.getTNTot()) / (double)(classification.getTNTot() + classification.getFPTot());
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
    public double classificationF1score() throws NeuralNetworkException {
        if (metricsType != MetricsType.CLASSIFICATION) throw new NeuralNetworkException("Not classification metric.");
        double precision = classificationPrecision();
        double recall = classificationRecall();
        return 2 * precision * recall / (precision + recall);
    }

    /**
     * Returns confusion matrix.
     *
     * @return confusion matrix.
     * @throws NeuralNetworkException throws exception if metrics is not defined as classification type.
     */
    public int[][] confusionMatrix() throws NeuralNetworkException {
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
            System.out.println("  F1 Score: " + classificationF1score());
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
        int[][] confusionMatrix = classification.getConfusion();
        for (int[] matrix : confusionMatrix) {
            System.out.print("[");
            for (int j = 0; j < confusionMatrix[0].length; j++) {
                System.out.print(matrix[j]);
                if (j < confusionMatrix[0].length - 1) System.out.print(" ");
            }
            System.out.println("]");
        }
    }

}
