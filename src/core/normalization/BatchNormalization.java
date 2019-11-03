/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.normalization;

import utils.*;
import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.UnaryFunctionType;
import utils.procedure.Node;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Set;
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
     * Epsilon term for batch normalization. Default value 10E-8.<br>
     * Term provides mathematical stability for normalizer.<br>
     *
     */
    private final double epsilon = 10E-8;

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
    private final HashMap<Node, Matrix> avgMeans = new HashMap<>();

    /**
     * Stores rolling average variance of samples when neural network is in training mode.<br>
     * Stored rolling average variance is used for normalization when neural network is in inference mode.<br>
     *
     */
    private final HashMap<Node, Matrix> avgVars = new HashMap<>();

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
    private final HashMap<Node, TreeMap<Integer, Matrix>> unMeanIns = new HashMap<>();

    /**
     * Matrix to store variance of current batch.<br>
     * Variance is calculated over batch samples per single feature.<br>
     *
     */
    private final HashMap<Node, Matrix> vars = new HashMap<>();

    /**
     * Matrix to store inverse squared variance (1 / squared variance) of current batch.<br>
     * This intermediate value is used for backpropagation gradient calculation.<br>
     *
     */
    private final HashMap<Node, Matrix> iSqrVars = new HashMap<>();

    /**
     * Number of mean and average samples collected.
     *
     */
    private int sampleCount = 0;

    /**
     * True if batch normalization is used calculation only with mean and variance excluded.
     *
     */
    private boolean meanOnly = false;

    /**
     * Constructor for batch normalization class.
     *
     */
    public BatchNormalization() {
    }

    /**
     * Constructor for batch normalization class.
     *
     * @param params parameters for batch normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public BatchNormalization(String params) throws DynamicParamException {
        this.setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for batch normalization.
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
        unMeanIns.clear();
        vars.clear();
        iSqrVars.clear();
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
     * @param node node for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forward(Node node) throws MatrixException {
        if (unMeanIns.containsKey(node)) return;
        batchSize = node.size();
        if (batchSize < 2) return;

        Set<Integer> keySet = node.keySet();

        if (isTraining) {
            sampleCount++;

            int rows = node.getMatrix(node.firstKey()).getRows();
            int cols = node.getMatrix(node.firstKey()).getCols();

            // Calculate mean and cumulate average rolling mean
            Matrix mean = new DMatrix(rows, cols);
            for (Integer sampleIndex : keySet) mean.add(node.getMatrix(sampleIndex), mean);
            mean.divide(batchSize, mean);

            if (!avgMeans.containsKey(node)) avgMeans.put(node, mean);
            else avgMeans.put(node, movingAverage(avgMeans.get(node), mean, sampleCount));

            if (!meanOnly) {
                // Calculate variance and cumulate average rolling variance
                Matrix var = new DMatrix(rows, cols);
                for (Integer sampleIndex : keySet) var.add(node.getMatrix(sampleIndex).subtract(mean).power(2), var);
                var.divide(batchSize, var);
                vars.put(node, var);

                iSqrVars.put(node, var.add(epsilon).apply(UnaryFunctionType.SQRT).apply(UnaryFunctionType.MULINV));

                if (!avgVars.containsKey(node)) avgVars.put(node, var);
                else avgVars.put(node, movingAverage(avgVars.get(node), var, sampleCount * sampleCount));
            }

            // Normalize mini batch by (output - mean) / sqrt(variance)
            TreeMap<Integer, Matrix> unMeanInsNode = new TreeMap<>();
            unMeanIns.put(node, unMeanInsNode);
            for (Integer sampleIndex : keySet) {
                Matrix unMeanIn = node.getMatrix(sampleIndex).subtract(mean);
                unMeanInsNode.put(sampleIndex, unMeanIn);
                if (!meanOnly) node.setMatrix(sampleIndex, unMeanIn.multiply(iSqrVars.get(node)));
                else node.setMatrix(sampleIndex, unMeanIn);
            }
        }
        else {
            if (!meanOnly) {
                Matrix iAvgSqrVar = avgVars.get(node).multiply(batchSize / (batchSize - 1)).add(epsilon).apply(UnaryFunctionType.SQRT).apply(UnaryFunctionType.MULINV);
                Matrix avgMean = avgMeans.get(node);
                for (Integer sampleIndex : keySet) {
                    node.setMatrix(sampleIndex, node.getMatrix(sampleIndex).subtract(avgMean).multiply(iAvgSqrVar));
                }
            }
            else {
                Matrix avgMean = avgMeans.get(node);
                for (Integer sampleIndex : keySet) {
                    node.setMatrix(sampleIndex, node.getMatrix(sampleIndex).subtract(avgMean));
                }
            }
        }
    }

    /**
     * Returns and updates moving average with sample.
     *
     * @param previousMovingAverage previous total moving average.
     * @param sample new sample.
     * @param sampleCount total number of samples.
     * @return updates moving average.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Matrix movingAverage(Matrix previousMovingAverage, Matrix sample, int sampleCount) throws MatrixException {
        return previousMovingAverage.add(sample.subtract(previousMovingAverage).divide(sampleCount));
    }

    /**
     * Executes backward propagation step for batch normalization.<br>
     * Calculates gradients backwards at step end for previous layer.<br>
     *
     * @param node node for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backward(Node node) throws MatrixException {
        if (!unMeanIns.containsKey(node)) return;

        int rows = node.getGradient(node.firstKey()).getRows();
        int cols = node.getGradient(node.firstKey()).getCols();

        Set<Integer> keySet = node.keySet();

        Matrix dsigma = new DMatrix(rows, cols);
        Matrix dmu = new DMatrix(rows, cols);
        TreeMap<Integer, Matrix> unMeanInsNode = unMeanIns.get(node);
        for (Integer sampleIndex : keySet) {
            dsigma.add(node.getGradient(sampleIndex).multiply(unMeanInsNode.get(sampleIndex)), dsigma);
            dmu.add(node.getGradient(sampleIndex), dmu);
        }
        dsigma.multiply(-0.5 / batchSize, dsigma);
        dmu.multiply(-1 / batchSize, dmu);
        if (!meanOnly) {
            dsigma.multiply(vars.get(node).add(epsilon).power(-1.5), dsigma);
            dmu.multiply(iSqrVars.get(node), dmu);
        }
        for (Integer sampleIndex : keySet) {
            Matrix dEoP;
            if (!meanOnly) dEoP = node.getGradient(sampleIndex).multiply(iSqrVars.get(node)).add(unMeanInsNode.get(sampleIndex).multiply(2).multiply(dsigma)).add(dmu);
            else dEoP = node.getGradient(sampleIndex).add(unMeanInsNode.get(sampleIndex).multiply(2).multiply(dsigma)).add(dmu);
            node.setGradient(sampleIndex, dEoP);
        }
    }

    /**
     * Not used.
     *
     * @param node node for normalization.
     * @param inputIndex input index for normalization.
     */
    public void forward(Node node, int inputIndex) {}

    /**
     * Not used.
     *
     * @param node node for normalization.
     * @param outputIndex input index for normalization.
     */
    public void backward(Node node, int outputIndex) {}

    /**
     * Not used.
     *
     * @param W weight for normalization.
     * @return normalized weight.
     */
    public Matrix forward(Matrix W) { return W; }

    /**
     * Not used.
     *
     * @param W weight for backward's normalization.
     * @param dW gradient of weight for backward normalization.
     * @return input weight gradients for backward normalization.
     */
    public Matrix backward(Matrix W, Matrix dW) { return dW; }

}