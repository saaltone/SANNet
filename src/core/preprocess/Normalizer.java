/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.preprocess;

import core.network.NeuralNetworkException;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.sampling.Sequence;

import java.util.HashMap;
import java.util.HashSet;

/**
 * Implements functionality for sample data normalization.<br>
 * Normalized either by using min max normalization or z- score normalization.<br>
 *
 */
public class Normalizer {

    /**
     * Flag to indicate if normalizer is already adjusted or not.
     *
     */
    private boolean adjustedMinMax = false;

    /**
     * Rows normalized with Min Max.
     *
     */
    private final HashSet<Integer> minMaxRows = new HashSet<>();

    /**
     * Min values for min max normalizer.
     *
     */
    private final HashMap<Integer, Double> minimumValues = new HashMap<>();

    /**
     * Max values for min max normalizer.
     *
     */
    private final HashMap<Integer, Double> maximumValues = new HashMap<>();

    /**
     * Rows normalized with Z-Score.
     *
     */
    private final HashSet<Integer> zScoreRows = new HashSet<>();

    /**
     * Flag to indicate if normalizer is already adjusted or not.
     *
     */
    private boolean adjustedZScore = false;

    /**
     * Mean values for z- score normalizer.
     *
     */
    private final HashMap<Integer, Double> means = new HashMap<>();

    /**
     * Standard deviation values for z- score normalizer.
     *
     */
    private final HashMap<Integer, Double> standardDeviations = new HashMap<>();

    /**
     * Default constructor for normalizer.
     *
     */
    public Normalizer() {
    }

    /**
     * Executes min max normalization.<br>
     *
     * @param data data to be normalized.
     * @throws NeuralNetworkException throws exception if normalizer is not yet adjusted.
     */
    public void minMaxSample(Matrix data) throws NeuralNetworkException {
        minMaxSample (new MMatrix(data));
    }

    /**
     * Executes min max normalization.<br>
     *
     * @param data data to be normalized.
     * @throws NeuralNetworkException throws exception if normalizer is not yet adjusted.
     */
    public void minMaxSample(MMatrix data) throws NeuralNetworkException {
        HashMap<Integer, MMatrix> inputData = new HashMap<>();
        inputData.put(0, data);
        minMaxSamples (inputData, 0, 1, false, null);
    }

    /**
     * Executes min max normalization.<br>
     *
     * @param data data to be normalized.
     * @throws NeuralNetworkException throws exception if normalizer is not yet adjusted.
     */
    public void minMaxSamples(HashMap<Integer, MMatrix> data) throws NeuralNetworkException {
        minMaxSamples (data, 0, 1, false, null);
    }

    /**
     * Executes min max normalization.<br>
     * Can either adjust normalizer or use earlier adjustment. This is useful if training data and test data must be adjusted by same normalization setting.<br>
     *
     * @param data data to be normalized.
     * @param adjust true if normalizer is adjusted with current data otherwise earlier normalization setting is applied.
     * @throws NeuralNetworkException throws exception if normalizer is not yet adjusted.
     */
    public void minMaxSamples(HashMap<Integer, MMatrix> data, boolean adjust) throws NeuralNetworkException {
        minMaxSamples (data, 0, 1, adjust, null);
    }

    /**
     * Executes min max normalization.<br>
     * Can either adjust normalizer or use earlier adjustment. This is useful if training data and test data must be adjusted by same normalization setting.<br>
     *
     * @param data data to be normalized.
     * @param newMinimum manually define minimum value for mapping.
     * @param newMaximum manually define maximum value for mapping.
     * @param adjust true if normalizer is adjusted with current data otherwise earlier normalization setting is applied.
     * @param normalizableRows rows to be normalized.
     * @throws NeuralNetworkException throws exception if normalizer is not yet adjusted.
     */
    public void minMaxSamples(HashMap<Integer, MMatrix> data, double newMinimum, double newMaximum, boolean adjust, HashSet<Integer> normalizableRows) throws NeuralNetworkException {
        if (data.size() == 0) return;
        if (!adjustedMinMax && !adjust) throw new NeuralNetworkException("Normalizer is not adjusted");
        if (adjust) {
            int dataRows = data.values().toArray(new MMatrix[0])[0].get(0).getRows();
            minMaxRows.clear();
            minimumValues.clear();
            maximumValues.clear();
            if (normalizableRows == null || normalizableRows.isEmpty()) for (int row = 0; row < dataRows; row++) minMaxRows.add(row);
            else for (Integer row : normalizableRows) if (row >= 0 && row < dataRows) minMaxRows.add(row);
        }
        for (Integer row : minMaxRows) {
            if (adjust) {
                minimumValues.put(row, Double.POSITIVE_INFINITY);
                maximumValues.put(row, Double.NEGATIVE_INFINITY);
                for (MMatrix mMatrix : data.values()) {
                    for (Matrix matrix : mMatrix.values()) {
                        minimumValues.put(row, Math.min(minimumValues.get(row), matrix.getValue(row, 0)));
                        maximumValues.put(row, Math.max(maximumValues.get(row), matrix.getValue(row, 0)));
                    }
                }
                adjustedMinMax = true;
            }
            double delta = maximumValues.get(row) - minimumValues.get(row) != 0 ? maximumValues.get(row) - minimumValues.get(row) : 1;
            for (MMatrix mMatrix : data.values()) {
                for (Matrix matrix : mMatrix.values()) {
                    double newValue = (matrix.getValue(row, 0) - minimumValues.get(row)) / delta * (newMaximum - newMinimum) + newMinimum;
                    matrix.setValue(row, 0, newValue);
                }
            }
        }
    }

    /**
     * Executes min max normalization.<br>
     *
     * @param data data to be normalized.
     * @throws NeuralNetworkException throws exception if normalizer is not yet adjusted.
     */
    public void minMaxSequence(Sequence data) throws NeuralNetworkException {
        HashMap<Integer, Sequence> inputData = new HashMap<>();
        inputData.put(0, data);
        minMaxSequences (inputData, 0, 1, false, null);
    }

    /**
     * Executes min max normalization.<br>
     *
     * @param data data to be normalized.
     * @throws NeuralNetworkException throws exception if normalizer is not yet adjusted.
     */
    public void minMaxSequences(HashMap<Integer, Sequence> data) throws NeuralNetworkException {
        minMaxSequences (data, 0, 1, false, null);
    }

    /**
     * Executes min max normalization.<br>
     * Can either adjust normalizer or use earlier adjustment. This is useful if training data and test data must be adjusted by same normalization setting.<br>
     *
     * @param data data to be normalized.
     * @param adjust true if normalizer is adjusted with current data otherwise earlier normalization setting is applied.
     * @throws NeuralNetworkException throws exception if normalizer is not yet adjusted.
     */
    public void minMaxSequences(HashMap<Integer, Sequence> data, boolean adjust) throws NeuralNetworkException {
        minMaxSequences (data, 0, 1, adjust, null);
    }

    /**
     * Executes min max normalization.<br>
     * Can either adjust normalizer or use earlier adjustment. This is useful if training data and test data must be adjusted by same normalization setting.<br>
     *
     * @param data data to be normalized.
     * @param newMinimum manually define minimum value for mapping.
     * @param newMaximum manually define maximum value for mapping.
     * @param adjust true if normalizer is adjusted with current data otherwise earlier normalization setting is applied.
     * @param normalizableRows rows to be normalized.
     * @throws NeuralNetworkException throws exception if normalizer is not yet adjusted.
     */
    public void minMaxSequences(HashMap<Integer, Sequence> data, double newMinimum, double newMaximum, boolean adjust, HashSet<Integer> normalizableRows) throws NeuralNetworkException {
        if (data.size() == 0) return;
        if (!adjustedMinMax && !adjust) throw new NeuralNetworkException("Normalizer is not adjusted");
        if (adjust) {
            int dataRows = data.values().toArray(new Sequence[0])[0].get(0).get(0).getRows();
            minMaxRows.clear();
            minimumValues.clear();
            maximumValues.clear();
            if (normalizableRows == null || normalizableRows.isEmpty()) for (int row = 0; row < dataRows; row++) minMaxRows.add(row);
            else for (Integer row : normalizableRows) if (row >= 0 && row < dataRows) minMaxRows.add(row);
        }
        for (Integer row : minMaxRows) {
            if (adjust) {
                minimumValues.put(row, Double.POSITIVE_INFINITY);
                maximumValues.put(row, Double.NEGATIVE_INFINITY);
                for (Sequence sequence : data.values()) {
                    for (MMatrix mMatrix : sequence.values()) {
                        for (Matrix matrix : mMatrix.values()) {
                            minimumValues.put(row, Math.min(minimumValues.get(row), matrix.getValue(row, 0)));
                            maximumValues.put(row, Math.max(maximumValues.get(row), matrix.getValue(row, 0)));
                        }
                    }
                }
                adjustedMinMax = true;
            }
            double delta = maximumValues.get(row) - minimumValues.get(row) != 0 ? maximumValues.get(row) - minimumValues.get(row) : 1;
            for (Sequence sequence : data.values()) {
                for (MMatrix mMatrix : sequence.values()) {
                    for (Matrix matrix : mMatrix.values()) {
                        double newValue = (matrix.getValue(row, 0) - minimumValues.get(row)) / delta * (newMaximum - newMinimum) + newMinimum;
                        matrix.setValue(row, 0, newValue);
                    }
                }
            }
        }
    }

    /**
     * Returns minimum values.
     *
     * @return minimum values.
     */
    public HashMap<Integer, Double> getMinimumValues() {
        return minimumValues;
    }

    /**
     * Returns minimum value of a specific row.
     *
     * @param row row corresponding minimum value.
     * @return minimum value.
     */
    public double getMinValue(int row) {
        return minimumValues.get(row);
    }

    /**
     * Returns maximum values.
     *
     * @return maximum values.
     */
    public HashMap<Integer, Double> getMaximumValues() {
        return maximumValues;
    }

    /**
     * Returns maximum values of a specific row.
     *
     * @param row row corresponding maximum value.
     * @return maximum value.
     */
    public double getMaxValue(int row) {
        return maximumValues.get(row);
    }

    /**
     * Executes z- score normalization.<br>
     *
     * @param data data to be normalized.
     * @throws NeuralNetworkException throws exception if normalizer is not yet adjusted.
     */
    public void zScoreSample(Matrix data) throws NeuralNetworkException {
        zScoreSample(new MMatrix(data));
    }

    /**
     * Executes z- score normalization.<br>
     *
     * @param data data to be normalized.
     * @throws NeuralNetworkException throws exception if normalizer is not yet adjusted.
     */
    public void zScoreSample(MMatrix data) throws NeuralNetworkException {
        HashMap<Integer, MMatrix> inputData = new HashMap<>();
        inputData.put(0, data);
        zScoreSamples(inputData, false, null);
    }

    /**
     * Executes z- score normalization.<br>
     *
     * @param data data to be normalized.
     * @throws NeuralNetworkException throws exception if normalizer is not yet adjusted.
     */
    public void zScoreSamples(HashMap<Integer, MMatrix> data) throws NeuralNetworkException {
        zScoreSamples(data, false, null);
    }

    /**
     * Executes z- score normalization.<br>
     * Can either adjust normalizer or use earlier adjustment. This is useful if training data and test data must be adjusted by same normalization setting.<br>
     *
     * @param data data to be normalized.
     * @param adjust true if normalizer is adjusted with current data otherwise earlier normalization setting is applied.
     * @param normalizableRows rows to be normalized.
     * @throws NeuralNetworkException throws exception if normalizer is not yet adjusted.
     */
    public void zScoreSamples(HashMap<Integer, MMatrix> data, boolean adjust, HashSet<Integer> normalizableRows) throws NeuralNetworkException {
        if (data.size() == 0) return;
        if (!adjustedZScore && !adjust) throw new NeuralNetworkException("Normalizer is not adjusted");
        if (adjust) {
            zScoreRows.clear();
            means.clear();
            standardDeviations.clear();
            int dataRows = data.values().toArray(new MMatrix[0])[0].get(0).getRows();
            if (normalizableRows == null || normalizableRows.isEmpty()) for (int row = 0; row < dataRows; row++) zScoreRows.add(row);
            else for (Integer row : normalizableRows) if (row >= 0 && row < dataRows) zScoreRows.add(row);
        }
        for (Integer row : zScoreRows) {
            if (adjust) {
                means.put(row, (double)0);
                for (MMatrix mMatrix : data.values()) {
                    for (Matrix matrix : mMatrix.values()) {
                        means.put(row, means.get(row) + matrix.getValue(row, 0));
                    }
                }
                means.put(row, means.get(row) / (double)data.size());

                standardDeviations.put(row, (double)0);
                for (MMatrix mMatrix : data.values()) {
                    for (Matrix matrix : mMatrix.values()) {
                        standardDeviations.put(row, standardDeviations.get(row) + Math.pow(matrix.getValue(row, 0) - means.get(row), 2));
                    }
                }
                standardDeviations.put(row, standardDeviations.get(row) > 0 ? Math.sqrt(standardDeviations.get(row) / ((double)data.size() - 1)) : 0);

                adjustedZScore = true;
            }
            if (standardDeviations.get(row) != 0) {
                for (MMatrix mMatrix : data.values()) {
                    for (Matrix matrix : mMatrix.values()) {
                        double newValue = (matrix.getValue(row, 0) - means.get(row)) / standardDeviations.get(row);
                        matrix.setValue(row, 0, newValue);
                    }
                }
            }
        }
    }

    /**
     * Executes z- score normalization.<br>
     *
     * @param data data to be normalized.
     * @throws NeuralNetworkException throws exception if normalizer is not yet adjusted.
     */
    public void zScoreSequences(Sequence data) throws NeuralNetworkException {
        HashMap<Integer, Sequence> inputData = new HashMap<>();
        inputData.put(0, data);
        zScoreSequences(inputData, false, null);
    }

    /**
     * Executes z- score normalization.<br>
     *
     * @param data data to be normalized.
     * @throws NeuralNetworkException throws exception if normalizer is not yet adjusted.
     */
    public void zScoreSequences(HashMap<Integer, Sequence> data) throws NeuralNetworkException {
        zScoreSequences(data, false, null);
    }

    /**
     * Executes z- score normalization.<br>
     * Can either adjust normalizer or use earlier adjustment. This is useful if training data and test data must be adjusted by same normalization setting.<br>
     *
     * @param data data to be normalized.
     * @param adjust true if normalizer is adjusted with current data otherwise earlier normalization setting is applied.
     * @param normalizableRows rows to be normalized.
     * @throws NeuralNetworkException throws exception if normalizer is not yet adjusted.
     */
    public void zScoreSequences(HashMap<Integer, Sequence> data, boolean adjust, HashSet<Integer> normalizableRows) throws NeuralNetworkException {
        if (data.size() == 0) return;
        if (!adjustedZScore && !adjust) throw new NeuralNetworkException("Normalizer is not adjusted");
        if (adjust) {
            zScoreRows.clear();
            means.clear();
            standardDeviations.clear();
            int dataRows = data.values().toArray(new Sequence[0])[0].get(0).get(0).getRows();
            if (normalizableRows == null || normalizableRows.isEmpty()) for (int row = 0; row < dataRows; row++) zScoreRows.add(row);
            else for (Integer row : normalizableRows) if (row >= 0 && row < dataRows) zScoreRows.add(row);
        }
        for (Integer row : zScoreRows) {
            if (adjust) {
                means.put(row, (double)0);
                for (Sequence sequence : data.values()) {
                    for (MMatrix mMatrix : sequence.values()) {
                        for (Matrix matrix : mMatrix.values()) {
                            means.put(row, means.get(row) + matrix.getValue(row, 0));
                        }
                    }
                }
                means.put(row, means.get(row) / (double)data.size());

                standardDeviations.put(row, (double)0);
                for (Sequence sequence : data.values()) {
                    for (MMatrix mMatrix : sequence.values()) {
                        for (Matrix matrix : mMatrix.values()) {
                            standardDeviations.put(row, standardDeviations.get(row) + Math.pow(matrix.getValue(row, 0) - means.get(row), 2));
                        }
                    }
                }
                standardDeviations.put(row, standardDeviations.get(row) > 0 ? Math.sqrt(standardDeviations.get(row) / ((double)data.size() - 1)) : 0);

                adjustedZScore = true;
            }
            if (standardDeviations.get(row) != 0) {
                for (Sequence sequence : data.values()) {
                    for (MMatrix mMatrix : sequence.values()) {
                        for (Matrix matrix : mMatrix.values()) {
                            double newValue = (matrix.getValue(row, 0) - means.get(row)) / standardDeviations.get(row);
                            matrix.setValue(row, 0, newValue);
                        }
                    }
                }
            }
        }
    }

    /**
     * Returns mean values.
     *
     * @return mean values.
     */
    public HashMap<Integer, Double> getMeans() {
        return means;
    }

    /**
     * Returns mean value of a specific row.
     *
     * @param row row corresponding mean value.
     * @return mean value.
     */
    public double getMean(int row) {
        return means.get(row);
    }

    /**
     * Returns standard deviation values.
     *
     * @return standard deviation values.
     */
    public HashMap<Integer, Double> getStandardDeviations() {
        return standardDeviations;
    }

    /**
     * Returns standard deviation value of a specific row.
     *
     * @param row row corresponding standard deviation value.
     * @return standard deviation value.
     */
    public double getStandardDeviation(int row) {
        return standardDeviations.get(row);
    }

}
