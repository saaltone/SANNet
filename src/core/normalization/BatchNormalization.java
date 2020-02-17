/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.normalization;

import core.optimization.Optimizer;
import utils.*;
import utils.matrix.*;
import utils.procedure.Node;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Random;
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
     * Epsilon term for Batch normalization. Default value 10E-8.<br>
     * Term provides mathematical stability for normalizer.<br>
     *
     */
    private final double epsilon = 10E-8;

    /**
     * Degree of weighting decrease for exponential moving average. Default value 0.9.
     *
     */
    private double eavgWeighting = 0.9;

    /**
     * Learnable parameter gammas of Batch normalization layer.<br>
     *
     */
    private final HashMap<Node, Matrix> gammas = new HashMap<>();

    /**
     * Learnable parameter betas of Batch normalization layer.<br>
     *
     */
    private final HashMap<Node, Matrix> betas = new HashMap<>();

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
     * Tree map to store normalized outputs.<br>
     * Used in backward propagation step for gradient calculation.<br>
     *
     */
    private final HashMap<Node, TreeMap<Integer, Matrix>> normOuts = new HashMap<>();

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
     * True if Batch normalization is used calculation only with mean and variance excluded.
     *
     */
    private boolean meanOnly = false;

    /**
     * Optimizer for Batch normalization;
     *
     */
    private Optimizer optimizer;

    /**
     * Default constructor for Batch normalization class.
     *
     */
    public BatchNormalization() {
    }

    /**
     * Constructor for Batch normalization class.
     *
     * @param params parameters for Batch normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public BatchNormalization(String params) throws DynamicParamException {
        this.setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for Batch normalization.
     *
     * @return parameters used for Batch normalization.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("meanOnly", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("beta", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for Batch Normalization.<br>
     * <br>
     * Supported parameters are:<br>
     *     - meanOnly: true if normalization is done only by using mean otherwise false (default value).<br>
     *     - beta: degree of weighting decrease for exponential moving average. Default value 0.9.<br>
     *
     * @param params parameters used for Batch normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("meanOnly")) meanOnly = params.getValueAsBoolean("meanOnly");
        if (params.hasParam("beta")) eavgWeighting = params.getValueAsDouble("beta");
    }

    /**
     * Resets Batch normalizer.
     *
     */
    public void reset() {
        unMeanIns.clear();
        normOuts.clear();
        vars.clear();
        iSqrVars.clear();
    }

    /**
     * Sets flag for Batch normalization if neural network is in training state.
     *
     * @param isTraining if true neural network is in state otherwise false.
     */
    public void setTraining(boolean isTraining) {
        this.isTraining = isTraining;
    }

    /**
     * Sets optimizer for normalizer.
     *
     * @param optimizer optimizer
     */
    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
    }

    /**
     * Executes forward propagation step for Batch normalization at step start.<br>
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

        int rows = node.getMatrix(node.firstKey()).getRows();
        int cols = node.getMatrix(node.firstKey()).getCols();

        Matrix gamma;
        if (gammas.containsKey(node)) gamma = gammas.get(node);
        else {
            gammas.put(node, gamma = new DMatrix(rows, cols));
            gamma.initialize((row, col) -> new Random().nextGaussian() * 0.1);
        }

        Matrix beta;
        if (betas.containsKey(node)) beta = betas.get(node);
        else betas.put(node, beta = new DMatrix(rows, cols));

        if (isTraining) {
            // Calculate mean and cumulate average rolling mean
            Matrix mean = new DMatrix(rows, cols);
            for (Integer sampleIndex : keySet) mean.add(node.getMatrix(sampleIndex), mean);
            mean.divide(batchSize, mean);
            avgMeans.put(node, Matrix.exponentialMovingAverage(avgMeans.get(node), mean, eavgWeighting));

            if (!meanOnly) {
                // Calculate variance and cumulate average rolling variance
                Matrix var = new DMatrix(rows, cols);
                for (Integer sampleIndex : keySet) var.add((node.getMatrix(sampleIndex).subtract(mean)).power(2), var);
                var.divide(batchSize, var);
                vars.put(node, var);
                iSqrVars.put(node, (var.add(epsilon).apply(UnaryFunctionType.SQRT)).apply(UnaryFunctionType.MULINV));
                avgVars.put(node, Matrix.exponentialMovingAverage(avgVars.get(node), var, eavgWeighting));
            }

            // Normalize mini batch by (output - mean) / sqrt(variance)
            TreeMap<Integer, Matrix> unMeanInsNode = new TreeMap<>();
            unMeanIns.put(node, unMeanInsNode);
            TreeMap<Integer, Matrix> normOutsNode = new TreeMap<>();
            normOuts.put(node, normOutsNode);
            for (Integer sampleIndex : keySet) {
                Matrix unMeanIn = node.getMatrix(sampleIndex).subtract(mean);
                unMeanInsNode.put(sampleIndex, unMeanIn);
                Matrix normOut = !meanOnly ? unMeanIn.multiply(iSqrVars.get(node)) : unMeanIn;
                normOutsNode.put(sampleIndex, normOut);
                node.setMatrix(sampleIndex, normOut.multiply(gamma).add(beta));
            }
        }
        else {
            Matrix avgMean = avgMeans.get(node);
            if (!meanOnly) {
                Matrix iAvgSqrVar = avgVars.get(node).multiply(batchSize / (batchSize - 1)).add(epsilon).apply(UnaryFunctionType.SQRT).apply(UnaryFunctionType.MULINV);
                for (Integer sampleIndex : keySet) {
                    node.setMatrix(sampleIndex, node.getMatrix(sampleIndex).subtract(avgMean).multiply(iAvgSqrVar).multiply(gamma).add(beta));
                }
            }
            else {
                for (Integer sampleIndex : keySet) {
                    node.setMatrix(sampleIndex, node.getMatrix(sampleIndex).subtract(avgMean).multiply(gamma).add(beta));
                }
            }
        }
    }

    /**
     * Executes backward propagation step for Batch normalization.<br>
     * Calculates gradients backwards at step end for previous layer.<br>
     *
     * @param node node for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backward(Node node) throws MatrixException {
        if (!unMeanIns.containsKey(node)) return;

        int rows = node.getGradient(node.firstKey()).getRows();
        int cols = node.getGradient(node.firstKey()).getCols();

        Matrix gamma = gammas.get(node);
        Matrix dgamma = new DMatrix(gamma.getRows(), gamma.getCols());
        Matrix beta = betas.get(node);
        Matrix dbeta = new DMatrix(beta.getRows(), beta.getCols());

        Set<Integer> keySet = node.keySet();

        TreeMap<Integer, Matrix> normOutsNode = normOuts.get(node);
        TreeMap<Integer, Matrix> inGrads = new TreeMap<>();
        for (Integer sampleIndex : keySet) {
            Matrix gradient = node.getGradient(sampleIndex);
            inGrads.put(sampleIndex, gradient.multiply(gamma));
            dgamma.add(gradient.multiply(normOutsNode.get(sampleIndex)), dgamma);
            dbeta.add(gradient, dbeta);
        }

        Matrix dsigma = new DMatrix(rows, cols);
        Matrix dmu = new DMatrix(rows, cols);
        TreeMap<Integer, Matrix> unMeanInsNode = unMeanIns.get(node);
        for (Integer sampleIndex : keySet) {
            dsigma.add(inGrads.get(sampleIndex).multiply(unMeanInsNode.get(sampleIndex)), dsigma);
            dmu.add(inGrads.get(sampleIndex), dmu);
        }
        dsigma.multiply(-1 / batchSize, dsigma);
        dmu.multiply(-1 / batchSize, dmu);
        if (!meanOnly) {
            dsigma.multiply(vars.get(node).add(epsilon).power(-1.5), dsigma);
            dmu.multiply(iSqrVars.get(node), dmu);
        }
        for (Integer sampleIndex : keySet) {
            Matrix dEoP;
            if (!meanOnly) dEoP = inGrads.get(sampleIndex).multiply(iSqrVars.get(node)).add(unMeanInsNode.get(sampleIndex).multiply(dsigma)).add(dmu);
            else dEoP = inGrads.get(sampleIndex).add(unMeanInsNode.get(sampleIndex).multiply(2).multiply(dsigma)).add(dmu);
            node.setGradient(sampleIndex, dEoP);
        }

        optimizer.optimize(gamma, dgamma);
        optimizer.optimize(beta, dbeta);
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