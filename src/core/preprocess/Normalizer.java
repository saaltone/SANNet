/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.preprocess;

import core.NeuralNetworkException;
import utils.Sample;

import java.util.LinkedHashMap;

/**
 * Class for sample data normalization.<br>
 * Normalized either by using min max normalization or z- score normalization.<br>
 *
 */
public class Normalizer {

    /**
     * Min value for min max normalizer.
     *
     */
    private double min;

    /**
     * Max value for min max normalizer.
     *
     */
    private double max;

    /**
     * Mean value for z- score normalizer.
     *
     */
    private double mean;

    /**
     * Standard deviation value for z- score normalizer.
     *
     */
    private double std;

    /**
     * Flag to indicate if normalizer is already adjusted or not.
     *
     */
    private boolean adjusted = false;

    /**
     * Default constructor for normalizer.
     *
     */
    public Normalizer() {
    }

    /**
     * Executes min max normalization.<br>
     * Can either adjust normalizer or use earlier adjustment. This is useful if training data and test data must be adjusted by same normalization setting.<br>
     *
     * @param data data to be normalized.
     * @param adjust true if normalizer is adjusted with current data otherwise earlier normalization setting is applied.
     * @throws NeuralNetworkException throws exception if normalizer is not yet adjusted.
     */
    public void minMax(LinkedHashMap<Integer, Sample> data, boolean adjust) throws NeuralNetworkException {
        minMax (data, 0, 1, adjust);
    }

    /**
     * Executes min max normalization.<br>
     * Can either adjust normalizer or use earlier adjustment. This is useful if training data and test data must be adjusted by same normalization setting.<br>
     *
     * @param data data to be normalized.
     * @param newMin manually define minimum value for mapping.
     * @param newMax manually define maximum value for mapping.
     * @param adjust true if normalizer is adjusted with current data otherwise earlier normalization setting is applied.
     * @throws NeuralNetworkException throws exception if normalizer is not yet adjusted.
     */
    public void minMax(LinkedHashMap<Integer, Sample> data, double newMin, double newMax, boolean adjust) throws NeuralNetworkException {
        if (!adjusted && !adjust) throw new NeuralNetworkException("Normalizer is not adjusted");
        for (int itemRow = 0; itemRow < data.values().toArray(new Sample[0])[0].get(0).getRows(); itemRow++) {
            if (adjust) {
                min = Double.MAX_VALUE;
                max = Double.MIN_VALUE;
                for (Integer entry : data.keySet()) {
                    for (Integer depth : data.get(entry).keySet()) {
                        min = Math.min(min, data.get(entry).get(depth).getValue(itemRow, 0));
                        max = Math.max(max, data.get(entry).get(depth).getValue(itemRow, 0));
                    }
                }
                adjusted = true;
            }
            double delta = max - min != 0 ? max - min : 1;
            for (Integer entry : data.keySet()) {
                for (Integer depth : data.get(entry).keySet()) {
                    double newValue = (data.get(entry).get(depth).getValue(itemRow, 0) - min) / delta * (newMax - newMin) + newMin;
                    data.get(entry).get(depth).setValue(itemRow, 0, newValue);
                }
            }
        }
    }

    /**
     * Executes z- score normalization.<br>
     * Can either adjust normalizer or use earlier adjustment. This is useful if training data and test data must be adjusted by same normalization setting.<br>
     *
     * @param data data to be normalized.
     * @param adjust true if normalizer is adjusted with current data otherwise earlier normalization setting is applied.
     * @throws NeuralNetworkException throws exception if normalizer is not yet adjusted.
     */
    public void zScore(LinkedHashMap<Integer, Sample> data, boolean adjust) throws NeuralNetworkException {
        if (!adjusted && !adjust) throw new NeuralNetworkException("Normalizer is not adjusted");
        for (int itemRow = 0; itemRow < data.get(0).get(0).getRows(); itemRow++) {
            if (adjust) {
                mean = 0;
                for (Integer entry : data.keySet()) {
                    for (Integer depth : data.get(entry).keySet()) {
                        mean += data.get(entry).get(depth).getValue(itemRow, 0);
                    }
                }
                mean = mean / data.size();
                std = 0;
                for (Integer entry : data.keySet()) {
                    for (Integer depth : data.get(entry).keySet()) {
                        std += Math.pow(data.get(entry).get(depth).getValue(itemRow, 0) - mean, 2);
                    }
                }
                std = std > 0 ? Math.sqrt(std / (data.size() - 1)) : 0;
                adjusted = true;
            }
            for (Integer entry : data.keySet()) {
                for (Integer depth : data.get(entry).keySet()) {
                    double newValue = (data.get(entry).get(depth).getValue(itemRow, 0) - mean) / std;
                    data.get(entry).get(depth).setValue(itemRow, 0, newValue);
                }
            }
        }
    }

}
