/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.normalization;

import core.layer.Connector;
import utils.*;

import java.io.Serializable;
import java.util.HashMap;
import java.util.TreeMap;

/**
 * Class that implements Batch Normalization for neural network layer.<br>
 * <br>
 * Reference: http://proceedings.mlr.press/v37/ioffe15.pdf<br>
 *
 */
public class BatchNormalization implements Normalization, Serializable {

    private static final long serialVersionUID = 3466341546851269706L;

    /**
     * Reference to connector between previous and next layer.
     *
     */
    private final Connector connector;

    /**
     * If true neural network is in state otherwise false.
     *
     */
    private transient boolean isTraining;

    /**
     * Stores rolling average mean of samples when neural network is in training mode.<br>
     * Stored rolling average mean is used for normalization when neural network is in inference mode.<br>
     *
     */
    private Matrix avgMean;

    /**
     * Stores rolling average variance of samples when neural network is in training mode.<br>
     * Stored rolling average variance is used for normalization when neural network is in inference mode.<br>
     *
     */
    private Matrix avgVar;

    /**
     * Sample size for a batch. Double is used for calculation purposes.
     *
     */
    private transient double batchSize;

    /**
     * Tree map to store inputs normalized by mean.<br>
     * Used in backward propagation step for gradient calculation.<br>
     *
     */
    private transient TreeMap<Integer, Matrix> unMeanIns;

    /**
     * Matrix to store variance of current batch.<br>
     * Variance is calculated over batch samples per single feature.<br>
     *
     */
    private transient Matrix var;

    /**
     * Matrix to store inverse squared variance (1 / squared variance) of current batch.<br>
     * This intermediate value is used for backpropagation gradient calculation.<br>
     *
     */
    private transient Matrix iSqrVar;

    /**
     * True if batch normalization is used calculation only with mean and variance excluded.
     *
     */
    private boolean meanOnly = false;

    /**
     * Constructor for batch normalization class.
     *
     * @param connector reference to connector between previous and next layer.
     */
    public BatchNormalization(Connector connector) {
        this.connector = connector;
    }

    /**
     * Constructor for batch normalization class.
     *
     * @param connector reference to connector between previous and next layer.
     * @param params parameters for batch normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public BatchNormalization(Connector connector, String params) throws DynamicParamException {
        this(connector);
        this.setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Gets parameters used for batch normalization.
     *
     * @return parameters used for batch normalization.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("meanOnly", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for Batch Normalization.<br>
     * <br>
     * Supported parameters are:<br>
     *     - meanOnly: true if normalization is done only by using mean otherwise false (default value).<br>
     *
     * @param params parameters used for batch normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("meanOnly")) meanOnly = params.getValueAsBoolean("meanOnly");
    }

    /**
     * Resets batch normalizer.
     *
     */
    public void reset() {
        avgMean = null;
        avgVar = null;
        clear();
    }

    /**
     * Clears cached values
     *
     */
    private void clear() {
        unMeanIns = new TreeMap<>();
        var = null;
        iSqrVar = null;
    }

    /**
     * Sets flag for batch normalization if neural network is in training state.
     *
     * @param isTraining if true neural network is in state otherwise false.
     */
    public void setTraining(boolean isTraining) {
        this.isTraining = isTraining;
    }

    /**
     * Executes forward propagation step for batch normalization at step start.<br>
     * Calculates feature wise mean and variance for batch of samples.<br>
     * Stores mean and variance into rolling averages respectively.<br>
     * Removes mean and variance from input samples.<br>
     *
     * @param ins input samples for forward step.
     * @param channels not used by batch normalization. Only relevant for convolutional layer.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forwardPre(TreeMap<Integer, Matrix> ins, int channels) throws MatrixException {
        if (ins.size() < 2) return;
        /**
         * Epsilon term for batch normalization. Default value 10E-8.<br>
         * Term provides mathematical stability for normalizer.<br>
         *
         */
        double epsilon = 10E-8;
        if (isTraining) {
            clear();
            batchSize = ins.size();
            int rows = ins.get(ins.firstKey()).getRows();
            int cols = ins.get(ins.firstKey()).getCols();

            // Calculate mean and cumulate average rolling mean
            Matrix mean = new DMatrix(rows, cols);
            for (Matrix input : ins.values()) mean.add(input, mean);
            mean.divide(batchSize, mean);

            /**
             * Smoothing factor for rolling mean and average calculation.<br>
             * Calculates avgSmoothFactor * current value + (1 - avgSmoothFactor) * new value.<br>
             *
             */
            double avgSmoothFactor = 0.9;
            if (avgMean == null) avgMean = mean;
            else avgMean = mean.multiply(1 - avgSmoothFactor).add(avgMean.multiply(avgSmoothFactor));

            if (!meanOnly) {
                // Calculate variance and cumulate average rolling variance
                var = new DMatrix(rows, cols);
                for (Matrix input : ins.values()) var.add(input.subtract(mean).power(2), var);
                var.divide(batchSize, var);
                //  Add epsilon for numerical stability, then sqrt
                var = var.add(epsilon);
                iSqrVar = var.sqrt().mulinv();

                if (avgVar == null) avgVar = var;
                else avgVar = var.multiply(1 - avgSmoothFactor).add(avgVar.multiply(avgSmoothFactor));
            }

            // Normalize mini batch by (output - mean) / sqrt(variance)
            for (Integer inIndex : ins.keySet()) {
                Matrix unMeanIn = ins.get(inIndex).subtract(mean);
                unMeanIns.put(inIndex, unMeanIn);
                if (!meanOnly) ins.put(inIndex, unMeanIn.multiply(iSqrVar));
                else ins.put(inIndex, unMeanIn);
            }
        }
        else {
//            Matrix iAvgSqrVar = avgVar.multiply(ins.size() / (ins.size() - 1)).add(epsilon).sqrt().mulinv();
            Matrix iAvgSqrVar = avgVar.add(epsilon).sqrt().mulinv();
            for (Integer inIndex : ins.keySet()) {
                Matrix unMeanIn = ins.get(inIndex).subtract(avgMean);
                if (!meanOnly) ins.put(inIndex, unMeanIn.multiply(iAvgSqrVar));
                else ins.put(inIndex, unMeanIn);
            }
        }
    }

    /**
     * Not used.
     *
     * @param outs output samples for forward step.
     */
    public void forwardPost(TreeMap<Integer, Matrix> outs) {}

    /**
     * Executes backward propagation step for batch normalization.<br>
     * Calculates gradients backwards at step end for previous layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backward() throws MatrixException {
        TreeMap<Integer, Matrix> dEos = connector.getNLayer().getdEos();
        if (dEos.size() < 2) return;

        int rows = dEos.get(dEos.firstKey()).getRows();
        int cols = dEos.get(dEos.firstKey()).getCols();
        Matrix dsigma = new DMatrix(rows, cols);
        Matrix dmu = new DMatrix(rows, cols);
        for (Integer index : dEos.keySet()) {
            Matrix dEo = dEos.get(index);
            dsigma.add(dEo.multiply(unMeanIns.get(index)), dsigma);
            dmu.add(dEo, dmu);
        }
        dsigma.multiply(-0.5 / batchSize, dsigma);
        dmu.multiply(-1 / batchSize, dmu);
        if (!meanOnly) {
            dsigma.multiply(var.power(-1.5), dsigma);
            dmu.multiply(iSqrVar, dmu);
        }
        for (Integer index : dEos.keySet()) {
            Matrix dEo = dEos.get(index);
            Matrix unMeanIn = unMeanIns.get(index);
            Matrix dEoP;
            if (!meanOnly) dEoP = dEo.multiply(iSqrVar).add(unMeanIn.multiply(2).multiply(dsigma)).add(dmu);
            else dEoP = dEo.add(unMeanIn.multiply(2).multiply(dsigma)).add(dmu);
            dEos.put(index, dEoP);
        }
    }

}