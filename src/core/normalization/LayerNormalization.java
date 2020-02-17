/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.normalization;

import core.optimization.Optimizer;
import utils.*;
import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.procedure.Node;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Random;
import java.util.TreeMap;

/**
 * Class that implements Layer Normalization for neural network layer.<br>
 * Layer Normalization is particularly well suited for recurrent neural networks as it normalizes each sample independently.<br>
 * <br>
 * Reference: https://www.cs.toronto.edu/~hinton/absps/LayerNormalization.pdf<br>
 *
 */
public class LayerNormalization implements Normalization, Serializable {

    private static final long serialVersionUID = 3466341546851269706L;

    /**
     * Epsilon term for Layer normalization. Default value 10E-8.<br>
     * Term provides mathematical stability for normalizer.<br>
     *
     */
    private final double epsilon = 10E-8;

    /**
     * Learnable parameter gammas of Layer normalization layer.<br>
     *
     */
    private final HashMap<Node, Matrix> gammas = new HashMap<>();

    /**
     * Learnable parameter betas of Layer normalization layer.<br>
     *
     */
    private final HashMap<Node, Matrix> betas = new HashMap<>();

    /**
     * If true neural network is in state otherwise false.
     *
     */
    private transient boolean isTraining;

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
    private final HashMap<Node, TreeMap<Integer, Double>> vars = new HashMap<>();

    /**
     * Matrix to store inverse squared variance (1 / squared variance) of current batch.<br>
     * This intermediate value is used for backpropagation gradient calculation.<br>
     *
     */
    private final HashMap<Node, TreeMap<Integer, Double>> iSqrVars = new HashMap<>();

    /**
     * True if layer normalization is used calculation only with mean and variance excluded.
     *
     */
    private boolean meanOnly = false;

    /**
     * Optimizer for Layer normalization;
     *
     */
    private Optimizer optimizer;

    /**
     * Constructor for Layer normalization class.
     *
     */
    public LayerNormalization() {
    }

    /**
     * Constructor for Layer normalization class.
     *
     * @param params parameters for Layer normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public LayerNormalization(String params) throws DynamicParamException {
        this.setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for Layer normalization.
     *
     * @return parameters used for Layer normalization.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("meanOnly", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for Layer normalization.<br>
     * <br>
     * Supported parameters are:<br>
     *     - meanOnly: true if normalization is done only by using mean otherwise false (default value).<br>
     *
     * @param params parameters used for Layer normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("meanOnly")) meanOnly = params.getValueAsBoolean("meanOnly");
    }

    /**
     * Resets Layer normalizer.
     *
     */
    public void reset() {
        unMeanIns.clear();
        vars.clear();
        iSqrVars.clear();
    }

    /**
     * Sets flag for Layer normalization if neural network is in training state.
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
     * Executes forward propagation step for Layer normalization at step start.<br>
     * Calculates feature wise mean and variance for each sample independently.<br>
     * Removes mean and variance from input samples.<br>
     *
     * @param node node for normalization.
     * @param inputIndex input index for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forward(Node node, int inputIndex) throws MatrixException {
        if (node.getMatrix(inputIndex).getSize() < 2) return;

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

        Matrix inputMatrix = node.getMatrix(inputIndex);
        if (isTraining) {
            TreeMap<Integer, Matrix> unMeanInEntry;
            TreeMap<Integer, Double> varEntry;
            TreeMap<Integer, Double> iSqrVarEntry;
            TreeMap<Integer, Matrix> normOutsEntry;
            if (unMeanIns.containsKey(node)) {
                unMeanInEntry = unMeanIns.get(node);
                varEntry = vars.get(node);
                iSqrVarEntry = iSqrVars.get(node);
                normOutsEntry = normOuts.get(node);
            }
            else {
                unMeanIns.put(node, unMeanInEntry = new TreeMap<>());
                vars.put(node, varEntry = new TreeMap<>());
                iSqrVars.put(node, iSqrVarEntry = new TreeMap<>());
                normOuts.put(node, normOutsEntry = new TreeMap<>());
            }

            // Calculate mean
            double mean = inputMatrix.mean();
            Matrix unMeanIn = inputMatrix.subtract(mean);
            unMeanInEntry.put(inputIndex, unMeanIn);

            if (!meanOnly) {
                // Calculate variance
                double var = inputMatrix.var();
                varEntry.put(inputIndex, var);
                double iSqrVar = 1 / Math.sqrt(var + epsilon);
                iSqrVarEntry.put(inputIndex, iSqrVar);
                Matrix normOut = unMeanIn.multiply(iSqrVar);
                normOutsEntry.put(inputIndex, normOut);
                node.setMatrix(inputIndex, normOut.multiply(gamma).add(beta));
            }
            else {
                normOutsEntry.put(inputIndex, unMeanIn);
                node.setMatrix(inputIndex, unMeanIn.multiply(gamma).add(beta));
            }
        }
        else {
            Matrix unMeanIn = inputMatrix.subtract(inputMatrix.mean());
            if (!meanOnly) node.setMatrix(inputIndex, unMeanIn.multiply(1 / Math.sqrt(inputMatrix.var() + epsilon)).multiply(gamma).add(beta));
            else node.setMatrix(inputIndex, unMeanIn.multiply(gamma).add(beta));
        }
    }

    /**
     * Executes backward propagation step for Layer normalization.<br>
     * Calculates gradients backwards at step end for previous layer.<br>
     *
     * @param node node for normalization.
     * @param outputIndex input index for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backward(Node node, int outputIndex) throws MatrixException {
        if (!unMeanIns.containsKey(node)) return;

        Matrix gamma = gammas.get(node);
        Matrix beta = betas.get(node);

        Matrix gradient = node.getGradient(outputIndex);
        Matrix inGrad =  gradient.multiply(gamma);
        Matrix dgamma = gradient.multiply(normOuts.get(node).get(outputIndex));
        Matrix dbeta = gradient;

        int size = inGrad.getSize();
        Matrix unMeanIn = unMeanIns.get(node).get(outputIndex);
        if (!meanOnly) {
            double dsigma = inGrad.multiply(unMeanIn).multiply(-1 / (double)size * Math.pow(vars.get(node).get(outputIndex) + epsilon, -1.5)).sum();
            double dmu = inGrad.multiply(-1 / (double)size * iSqrVars.get(node).get(outputIndex)).sum();
            node.setGradient(outputIndex, inGrad.multiply(iSqrVars.get(node).get(outputIndex)).add(unMeanIn.multiply(dsigma)).add(dmu));
        }
        else {
            double dsigma = inGrad.multiply(unMeanIn).multiply(-1 / (double)size).sum();
            double dmu = inGrad.multiply(-1 / (double)size).sum();
            node.setGradient(outputIndex, inGrad.add(unMeanIn.multiply(dsigma)).add(dmu));
        }

        optimizer.optimize(gamma, dgamma);
        optimizer.optimize(beta, dbeta);
    }

    /**
     * Not used.
     *
     * @param node node for normalization.
     */
    public void forward(Node node) {}

    /**
     * Not used.
     *
     * @param node node for normalization.
     */
    public void backward(Node node) {}

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