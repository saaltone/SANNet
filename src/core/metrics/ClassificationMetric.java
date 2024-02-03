/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.metrics;

import core.network.NeuralNetworkException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.UnaryFunction;
import utils.sampling.Sequence;

import java.io.Serial;
import java.io.Serializable;
import java.util.*;

/**
 * Implements functionality for calculation of classification error.
 *
 */
public class ClassificationMetric implements Metric, Serializable {

    @Serial
    private static final long serialVersionUID = 7333866575698397961L;

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
     * Average type for classification: macro or micro.
     *
     */
    private final AverageType averageType;

    /**
     * If true assumes multi label classification otherwise assumes single label classification.<br>
     * Single label assumes that only one label is true (1) and others false (0). Assumes that max output value takes true value.<br>
     * Multi label assumes that any value above threshold is true (1) otherwise false (0).<br>
     *
     */
    private final boolean multiLabel;

    /**
     * Defines threshold value for multi label classification. If value of label is below threshold it is classified as negative (0) otherwise classified as positive (1).
     *
     */
    private final double multiLabelThreshold;

    /**
     * Features classified.
     *
     */
    private final TreeSet<Integer> features = new TreeSet<>();

    /**
     * True positive counts for each feature.
     *
     */
    private final TreeMap<Integer, Integer> TP = new TreeMap<>();

    /**
     * False positive counts for each feature.
     *
     */
    private final TreeMap<Integer, Integer> FP = new TreeMap<>();

    /**
     * True negative counts for each feature.
     *
     */
    private final TreeMap<Integer, Integer> TN = new TreeMap<>();

    /**
     * False negative counts for each feature.
     *
     */
    private final TreeMap<Integer, Integer> FN = new TreeMap<>();

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
    private final TreeMap<Integer, TreeMap<Integer, Integer>> confusionMatrix = new TreeMap<>();

    /**
     * If true prints confusion matrix along other classification metrics.
     *
     */
    private boolean printConfusionMatrix;

    /**
     * If true shows confusion matrix along other classification metrics.
     *
     */
    private boolean showConfusionMatrix;

    /**
     * Reference to confusion matrix chart.
     *
     */
    private ConfusionMatrixChart confusionMatrixChart = null;

    /**
     * Reference to metrics chart.
     *
     */
    private final TrendMetricChart trendMetricChart;

    /**
     * Report count.
     *
     */
    private int reportCount = 0;

    /**
     * Default constructor for classification class.
     *
     */
    public ClassificationMetric() {
        this(false);
    }

    /**
     * Default constructor for classification class.
     *
     * @param showMetric if true shows metric otherwise not.
     */
    public ClassificationMetric(boolean showMetric) {
        this(AverageType.MACRO, showMetric);
    }

    /**
     * Default constructor for classification class.
     *
     * @param multiLabel if true assumes multi label classification otherwise assumes single label.
     * @param showMetric if true shows metric otherwise not.
     */
    public ClassificationMetric(boolean multiLabel, boolean showMetric) {
        this(AverageType.MACRO, multiLabel, showMetric);
    }

    /**
     * Constructor for classification.
     *
     * @param averageType average type.
     * @param showMetric if true shows metric otherwise not.
     */
    public ClassificationMetric(AverageType averageType, boolean showMetric) {
        this(averageType, false, showMetric);
    }

    /**
     * Constructor for classification metric.
     *
     * @param averageType average type.
     * @param multiLabel if true assumes multi label classification otherwise assumes single label.
     * @param showMetric if true shows metric otherwise not.
     */
    public ClassificationMetric(AverageType averageType, boolean multiLabel, boolean showMetric) {
        this(averageType, multiLabel, 0.5, showMetric);
    }

    /**
     * Constructor for classification metric.
     *
     * @param averageType average type.
     * @param multiLabel if true assumes multi label classification otherwise assumes single label.
     * @param multiLabelThreshold if class probability is below threshold is it classified as negative (0) otherwise as positive (1).
     * @param showMetric if true shows metric otherwise not.
     */
    public ClassificationMetric(AverageType averageType, boolean multiLabel, double multiLabelThreshold, boolean showMetric) {
        this(averageType, multiLabel, multiLabelThreshold, true, true, showMetric);
    }

    /**
     * Constructor for classification metric.
     *
     * @param averageType          average type.
     * @param multiLabel           if true assumes multi label classification otherwise assumes single label.
     * @param multiLabelThreshold  if class probability is below threshold is it classified as negative (0) otherwise as positive (1).
     * @param printConfusionMatrix if true prints confusion matrix otherwise not.
     * @param showConfusionMatrix  if true shows confusion matrix otherwise not.
     * @param showMetric           if true shows metric otherwise not.
     */
    public ClassificationMetric(AverageType averageType, boolean multiLabel, double multiLabelThreshold, boolean printConfusionMatrix, boolean showConfusionMatrix, boolean showMetric) {
        this.averageType = averageType;
        this.multiLabel = multiLabel;
        this.multiLabelThreshold = multiLabelThreshold;
        this.printConfusionMatrix = printConfusionMatrix;
        this.showConfusionMatrix = showConfusionMatrix;
        trendMetricChart = showMetric ? new TrendMetricChart("Neural Network Classification Accuracy", "Step", "F1 Score") : null;
    }

    /**
     * Sets if confusion matrix is printed along other classification metrics.
     *
     * @param printConfusionMatrix if true confusion matrix is printed along other classification metrics.
     */
    public void setPrintConfusionMatrix(boolean printConfusionMatrix) {
        this.printConfusionMatrix = printConfusionMatrix;
    }

    /**
     * Returns if confusion matrix is printed along other classification metrics.
     *
     * @return if true confusion matrix is printed along other classification metrics.
     */
    public boolean getPrintConfusionMatrix() {
        return printConfusionMatrix;
    }

    /**
     * Sets if confusion matrix is shown along other classification metrics.
     *
     * @param showConfusionMatrix if true confusion matrix is shown along other classification metrics.
     */
    public void setShowConfusionMatrix(boolean showConfusionMatrix) {
        this.showConfusionMatrix = showConfusionMatrix;
    }

    /**
     * Returns if confusion matrix is shown along other classification metrics.
     *
     * @return if true confusion matrix is shown along other classification metrics.
     */
    public boolean getShowConfusionMatrix() {
        return showConfusionMatrix;
    }

    /**
     * Returns reference metric.
     *
     * @return reference metric.
     */
    public Metric reference() {
        return new ClassificationMetric(averageType, multiLabel, multiLabelThreshold, getPrintConfusionMatrix(), getShowConfusionMatrix(), trendMetricChart != null);
    }

    /**
     * Reports errors and handles them as either regression or classification errors depending on metrics initialization.
     *
     * @param predicted    predicted errors.
     * @param actual       actual (true) error.
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if reporting of errors fails.
     */
    public void report(Sequence predicted, Sequence actual) throws MatrixException, NeuralNetworkException {
        if (actual.sampleSize() == 0) throw new NeuralNetworkException("Nothing to classify");
        update(getClassification(predicted), actual);
        if (trendMetricChart != null) trendMetricChart.addErrorData(++reportCount, classificationF1Score());
    }

    /**
     * Updates classification statistics and confusion matrix for a predicted / actual (true) sample pair.
     *
     * @param predicted predicted sample.
     * @param actual actual (true) sample.
     */
    private void update(Matrix predicted, Matrix actual) {
        int actualRows = actual.getRows();
        int predictedRows = predicted.getRows();
        for (int predictedRow = 0; predictedRow < predictedRows; predictedRow++) {
            features.add(predictedRow);
            double actualValue = actual.getValue(predictedRow, 0, 0);
            for (int actualRow = 0; actualRow < actualRows; actualRow++) {
                double predictedValue = predicted.getValue(actualRow, 0, 0);
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
        TreeMap<Integer, Integer> actuals = confusionMatrix.computeIfAbsent(predictedRow, k -> new TreeMap<>());
        actuals.put(actualRow, actuals.getOrDefault(actualRow, 0) + 1);
    }

    /**
     * Increments true positive count.
     *
     * @param row row
     */
    private void incrementTP(int row) {
        TP.put(row, TP.getOrDefault(row, 0) + 1);
        TPTotal++;
    }

    /**
     * Increments true negative count.
     *
     * @param row row
     */
    private void incrementTN(int row) {
        TN.put(row, TN.getOrDefault(row, 0) + 1);
        TNTotal++;
    }

    /**
     * Increments false negative count.
     *
     * @param row row
     */
    private void incrementFN(int row) {
        FN.put(row, FN.getOrDefault(row, 0) + 1);
        FNTotal++;
    }

    /**
     * Increments false positive count.
     *
     * @param row row
     */
    private void incrementFP(int row) {
        FP.put(row, FP.getOrDefault(row, 0) + 1);
        FPTotal++;
    }

    /**
     * Updates classification statistics and confusion matrix for multiple samples.
     *
     * @param predicted predicted samples.
     * @param actual actual (true) samples.
     */
    public void update(Sequence predicted, Sequence actual) {
        for (Map.Entry<Integer, Matrix> entry : predicted.entrySet()) {
            update(entry.getValue(), actual.get(entry.getKey()));
        }
    }

    /**
     * Returns last error.
     *
     * @return last error.
     */
    public double getLastError() {
        return classificationF1Score();
    }

    /**
     * Resets classification statistics.
     *
     */
    public void reset() {
        features.clear();
        TP.clear();
        FP.clear();
        TN.clear();
        FN.clear();
        TPTotal = 0;
        FPTotal = 0;
        TNTotal = 0;
        FNTotal = 0;
        confusionMatrix.clear();
    }

    /**
     * Reinitializes metric.
     *
     */
    public void reinitialize() {
        reset();
    }

    /**
     * Returns classified features.
     *
     * @return classified features.
     */
    public TreeSet<Integer> getFeatures() {
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
    public TreeMap<Integer, TreeMap<Integer, Integer>> getConfusionMatrix() {
        return confusionMatrix;
    }

    /**
     * Returns specific value in confusion matrix.
     *
     * @param predictedRow predicted row.
     * @param actualRow actual row.
     * @return specific value in confusion matrix.
     */
    public int getConfusionValue(int predictedRow, int actualRow) {
        return getConfusionMatrix().get(predictedRow) == null ? 0 : getConfusionMatrix().get(predictedRow).getOrDefault(actualRow, 0);
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
            return predicted.apply(new UnaryFunction(value -> value != maxValue ? 0 : 1));
        }
        else {
            return predicted.apply(new UnaryFunction(value -> value < multiLabelThreshold ? 0 : 1));
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
    private Sequence getClassification(Sequence predicted) throws MatrixException {
        Sequence classified = new Sequence();
        for (Map.Entry<Integer, Matrix> entry : predicted.entrySet()) {
            classified.put(entry.getKey(), getClassification(entry.getValue()));
        }
        return classified;
    }

    /**
     * Returns classification accuracy.<br>
     * Accuracy is calculated as (TP + TN) / (TP + FP + TN + FN).<br>
     * Takes into consideration if statistics is defined as macro or micro average.<br>
     *
     * @return classification accuracy.
     */
    public double classificationAccuracy() {
        if (averageType == AverageType.MACRO) {
            double average = 0;
            int averageCount = 0;
            for (Integer feature : getFeatures()) {
                double TP = getTP(feature);
                double FP = getFP(feature);
                double TN = getTN(feature);
                double FN = getFN(feature);
                if (FP + TN + FP + FN > 0) {
                    average += (TP + TN) / (TP + FP + TN + FN);
                    averageCount++;
                }
            }
            return averageCount == 0 ? 0 : average / (double)averageCount;
        }
        else {
            return (double)(getTPTotal() + getTNTotal()) / (double)(getTPTotal() + getFPTotal() + getTNTotal() + getFNTotal());
        }
    }

    /**
     * Returns classification error rate.<br>
     * Accuracy is calculated as (FP + FN) / (TP + FP + TN + FN).<br>
     * Takes into consideration if statistics is defined as macro or micro average.<br>
     *
     * @return classification error rate.
     */
    public double classificationErrorRate() {
        if (averageType == AverageType.MACRO) {
            double average = 0;
            int averageCount = 0;
            for (Integer feature : getFeatures()) {
                double TP = getTP(feature);
                double FP = getFP(feature);
                double TN = getTN(feature);
                double FN = getFN(feature);
                if (FP + TN + FP + FN > 0) {
                    average += (FP + FN) / (TP + FP + TN + FN);
                    averageCount++;
                }
            }
            return averageCount == 0 ? 0 : average / (double)averageCount;
        }
        else {
            return (double)(getFPTotal() + getFNTotal()) / (double)(getTPTotal() + getFPTotal() + getTNTotal() + getFNTotal());
        }
    }

    /**
     * Returns classification precision (positive predictive value).<br>
     * Precision is calculated as TP / (TP + FP).<br>
     * Measures share of correctly classified positive samples out of all samples predicted as positive.<br>
     * Takes into consideration if statistics is defined as macro or micro average.<br>
     *
     * @return classification precision.
     */
    public double classificationPrecision() {
        if (averageType == AverageType.MACRO) {
            double average = 0;
            int averageCount = 0;
            for (Integer feature : getFeatures()) {
                double TP = getTP(feature);
                double FP = getFP(feature);
                if (TP + FP > 0) {
                    average += TP / (TP + FP);
                    averageCount++;
                }
            }
            return averageCount == 0 ? 0 : average / (double)averageCount;
        }
        else {
            return (double)(getTPTotal()) / (double)(getTPTotal() + getFPTotal());
        }
    }

    /**
     * Returns classification recall (sensitivity, hit rate, true positive rate).<br>
     * Recall is calculated as TP / (TP + FN).<br>
     * Measures share of correctly classified positive samples out of all samples actually positive.<br>
     * Takes into consideration if statistics is defined as macro or micro average.<br>
     *
     * @return classification recall.
     */
    public double classificationRecall() {
        if (averageType == AverageType.MACRO) {
            double average = 0;
            int averageCount = 0;
            for (Integer feature : getFeatures()) {
                double TP = getTP(feature);
                double FN = getFN(feature);
                if (TP + FN > 0) {
                    average += TP / (TP + FN);
                    averageCount++;
                }
            }
            return averageCount == 0 ? 0 : average / (double)averageCount;
        }
        else {
            return (double)(getTPTotal()) / (double)(getTPTotal() + getFNTotal());
        }
    }

    /**
     * Returns classification specificity (selectivity, true negative rate).<br>
     * Specificity is calculated as TN / (TN + FP).<br>
     * Measures share of correctly classified negative samples out of all samples actually negative.<br>
     * Takes into consideration if statistics is defined as macro or micro average.<br>
     *
     * @return classification specificity.
     */
    public double classificationSpecificity() {
        if (averageType == AverageType.MACRO) {
            double average = 0;
            int averageCount = 0;
            for (Integer feature : getFeatures()) {
                double FP = getFP(feature);
                double TN = getTN(feature);
                if (TN + FP > 0) {
                    average += TN / (TN + FP);
                    averageCount++;
                }
            }
            return averageCount == 0 ? 0 : average / (double)averageCount;
        }
        else {
            return (double)(getTNTotal()) / (double)(getTNTotal() + getFPTotal());
        }
    }

    /**
     * Returns classification F1 score.<br>
     * F1 is calculated as 2 * precision * recall / (precision + recall).<br>
     * Takes into consideration if statistics is defined as macro or micro average.<br>
     *
     * @return classification F1 score.
     */
    public double classificationF1Score() {
        if (averageType == AverageType.MACRO) {
            double average = 0;
            int averageCount = 0;
            for (Integer feature : getFeatures()) {
                double TP = getTP(feature);
                double FP = getFP(feature);
                double FN = getFN(feature);
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
            double TP = getTPTotal();
            double FP = getFPTotal();
            double FN = getFNTotal();
            double precision = TP / (TP + FP);
            double recall =  TP / (TP + FN);
            return precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
        }
    }

    /**
     * Prints classification report.
     *
     */
    public void printReport() {
        double classificationAccuracy = classificationAccuracy();
        double classificationErrorRate = classificationErrorRate();
        double classificationPrecision = classificationPrecision();
        double classificationRecall = classificationRecall();
        double classificationSpecificity = classificationSpecificity();
        double classificationF1Score = classificationF1Score();
        System.out.println("Classification report:");
        System.out.println("  Accuracy: " + classificationAccuracy);
        System.out.println("  Error rate: " + classificationErrorRate);
        System.out.println("  Precision: " + classificationPrecision);
        System.out.println("  Recall: " + classificationRecall);
        System.out.println("  Specificity: " + classificationSpecificity);
        System.out.println("  F1 Score: " + classificationF1Score);
        if (printConfusionMatrix) printConfusionMatrix();
        if (showConfusionMatrix) {
            if (confusionMatrixChart == null) confusionMatrixChart = new ConfusionMatrixChart();
            confusionMatrixChart.updateConfusion(getFeatures(), getConfusionMatrix(), classificationAccuracy, classificationErrorRate, classificationPrecision, classificationRecall, classificationSpecificity, classificationF1Score);
        }
    }

    /**
     * Prints confusion matrix.
     *
     */
    public void printConfusionMatrix() {
        System.out.println("Confusion matrix (actual value as rows, predicted value as columns):");
        for (Integer predictedRow : getFeatures()) {
            System.out.print("[");
            int index = 0;
            for (Integer actualRow : getFeatures()) {
                System.out.print(getConfusionValue(predictedRow, actualRow));
                if (index++ < getFeatures().size() - 1) System.out.print(" ");
            }
            System.out.println("]");
        }
    }

}
