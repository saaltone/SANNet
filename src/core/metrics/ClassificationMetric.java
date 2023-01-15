/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.metrics;

import core.network.NeuralNetworkException;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.sampling.Sequence;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

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
    private AverageType averageType = AverageType.MACRO;

    /**
     * If true assumes multi label classification otherwise assumes single label classification.<br>
     * Single label assumes that only one label is true (1) and others false (0). Assumes that max output value takes true value.<br>
     * Multi label assumes that any value above threshold is true (1) otherwise false (0).<br>
     *
     */
    private boolean multiLabel = false;

    /**
     * Defines threshold value for multi label classification. If value of label is below threshold it is classified as negative (0) otherwise classified as positive (1).
     *
     */
    private double multiLabelThreshold = 0.5;

    /**
     * Features classified.
     *
     */
    private final HashSet<Integer> features = new HashSet<>();

    /**
     * True positive counts for each feature.
     *
     */
    private final HashMap<Integer, Integer> TP = new HashMap<>();

    /**
     * False positive counts for each feature.
     *
     */
    private final HashMap<Integer, Integer> FP = new HashMap<>();

    /**
     * True negative counts for each feature.
     *
     */
    private final HashMap<Integer, Integer> TN = new HashMap<>();

    /**
     * False negative counts for each feature.
     *
     */
    private final HashMap<Integer, Integer> FN = new HashMap<>();

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
    private final HashMap<Integer, HashMap<Integer, Integer>> confusion = new HashMap<>();

    /**
     * If true print confusion matrix along other classification metrics.
     *
     */
    private boolean printConfusionMatrix = true;

    /**
     * Default constructor for classification class.
     *
     */
    public ClassificationMetric() {}

    /**
     * Constructor for classification.
     *
     * @param averageType average type.
     */
    public ClassificationMetric(AverageType averageType) {
        this.averageType = averageType;
    }

    /**
     * Constructor for classification metric.
     *
     * @param averageType average type.
     * @param multiLabel if true assumes multi label classification otherwise assumes single label.
     */
    public ClassificationMetric(AverageType averageType, boolean multiLabel) {
        this(averageType);
        this.multiLabel = multiLabel;
    }

    /**
     * Constructor for classification metric.
     *
     * @param averageType average type.
     * @param multiLabel if true assumes multi label classification otherwise assumes single label.
     * @param multiLabelThreshold if class probability is below threshold is it classified as negative (0) otherwise as positive (1).
     */
    public ClassificationMetric(AverageType averageType, boolean multiLabel, double multiLabelThreshold) {
        this(averageType, multiLabel);
        this.multiLabelThreshold = multiLabelThreshold;
    }

    /**
     * Constructor for classification metric.
     *
     * @param averageType average type.
     * @param multiLabel if true assumes multi label classification otherwise assumes single label.
     * @param multiLabelThreshold if class probability is below threshold is it classified as negative (0) otherwise as positive (1).
     * @param printConfusionMatrix if true verbosing prints confusion matrix otherwise not.
     */
    public ClassificationMetric(AverageType averageType, boolean multiLabel, double multiLabelThreshold, boolean printConfusionMatrix) {
        this(averageType, multiLabel, multiLabelThreshold);
        this.printConfusionMatrix = printConfusionMatrix;
    }

    /**
     * Constructor for classification metric.
     *
     * @param multiLabel if true assumes multi label classification otherwise assumes single label.
     */
    public ClassificationMetric(boolean multiLabel) {
        this.multiLabel = multiLabel;
    }

    /**
     * Constructor for classification metric.
     *
     * @param multiLabel if true assumes multi label classification otherwise assumes single label.
     * @param multiLabelThreshold if class probability is below threshold is it classified as negative (0) otherwise as positive (1).
     */
    public ClassificationMetric(boolean multiLabel, double multiLabelThreshold) {
        this(multiLabel);
        this.multiLabelThreshold = multiLabelThreshold;
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
     * Returns reference metric.
     *
     * @return reference metric.
     */
    public Metric reference() {
        return new ClassificationMetric(averageType, multiLabel, multiLabelThreshold, getPrintConfusionMatrix());
    }

    /**
     * Reports errors and handles them as either regression or classification errors depending on metrics initialization.
     *
     * @param predicted predicted errors.
     * @param actual actual (true) error.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if reporting of errors fails.
     */
    public void report(Sequence predicted, Sequence actual) throws MatrixException, NeuralNetworkException {
        updateConfusion(predicted, actual);
    }

    /**
     * Updates classification statistics and confusion matrix for a predicted / actual (true) sample pair.
     *
     * @param predicted predicted sample.
     * @param actual actual (true) sample.
     */
    public void update(Matrix predicted, Matrix actual) {
        int actualRows = actual.getRows();
        int predictedRows = predicted.getRows();
        for (int predictedRow = 0; predictedRow < predictedRows; predictedRow++) {
            features.add(predictedRow);
            double actualValue = actual.getValue(predictedRow, 0);
            for (int actualRow = 0; actualRow < actualRows; actualRow++) {
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
        HashMap<Integer, Integer> actuals = confusion.computeIfAbsent(predictedRow, k -> new HashMap<>());
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
        confusion.clear();
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
    private Sequence getClassification(Sequence predicted) throws MatrixException {
        Sequence classified = new Sequence();
        for (Map.Entry<Integer, MMatrix> entry : predicted.entrySet()) {
            int sampleIndex = entry.getKey();
            MMatrix predictedSample = entry.getValue();
            int depth = predictedSample.getDepth();
            MMatrix classifiedSample = new MMatrix(depth);
            classified.put(sampleIndex, classifiedSample);
            for (int depthIndex = 0; depthIndex < depth; depthIndex++) {
                classifiedSample.put(depthIndex, getClassification(predictedSample.get(depthIndex)));
            }
        }
        return classified;
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
        update(predicted, actual);
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
     * Returns confusion matrix.
     *
     * @return confusion matrix.
     */
    public HashMap<Integer, HashMap<Integer, Integer>> confusionMatrix() {
        return getConfusion();
    }

    /**
     * Prints classification report.
     *
     */
    public void printReport() {
        System.out.println("Classification report:");
        System.out.println("  Accuracy: " + classificationAccuracy());
        System.out.println("  Precision: " + classificationPrecision());
        System.out.println("  Recall: " + classificationRecall());
        System.out.println("  Specificity: " + classificationSpecificity());
        System.out.println("  F1 Score: " + classificationF1Score());
        if (printConfusionMatrix) printConfusionMatrix();
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
