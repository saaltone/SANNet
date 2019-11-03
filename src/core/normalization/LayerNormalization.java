/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.normalization;

import utils.*;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.procedure.Node;

import java.io.Serializable;
import java.util.HashMap;
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
     * Epsilon term for layer normalization. Default value 10E-8.<br>
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
     * Constructor for layer normalization class.
     *
     */
    public LayerNormalization() {
    }

    /**
     * Constructor for layer normalization class.
     *
     * @param params parameters for layer normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public LayerNormalization(String params) throws DynamicParamException {
        this.setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for layer normalization.
     *
     * @return parameters used for layer normalization.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("meanOnly", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for Layer Normalization.<br>
     * <br>
     * Supported parameters are:<br>
     *     - meanOnly: true if normalization is done only by using mean otherwise false (default value).<br>
     *
     * @param params parameters used for layer normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("meanOnly")) meanOnly = params.getValueAsBoolean("meanOnly");
    }

    /**
     * Resets layer normalizer.
     *
     */
    public void reset() {
        unMeanIns.clear();
        vars.clear();
        iSqrVars.clear();
    }

    /**
     * Sets flag for layer normalization if neural network is in training state.
     *
     * @param isTraining if true neural network is in state otherwise false.
     */
    public void setTraining(boolean isTraining) {
        this.isTraining = isTraining;
    }

    /**
     * Executes forward propagation step for layer normalization at step start.<br>
     * Calculates feature wise mean and variance for each sample independently.<br>
     * Removes mean and variance from input samples.<br>
     *
     * @param node node for normalization.
     * @param inputIndex input index for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forward(Node node, int inputIndex) throws MatrixException {
        if (node.getMatrix(inputIndex).getSize() < 2) return;

        Matrix inputMatrix = node.getMatrix(inputIndex);
        if (isTraining) {
            TreeMap<Integer, Matrix> unMeanInEntry;
            TreeMap<Integer, Double> varEntry;
            TreeMap<Integer, Double> iSqrVarEntry;
            if (unMeanIns.containsKey(node)) {
                unMeanInEntry = unMeanIns.get(node);
                varEntry = vars.get(node);
                iSqrVarEntry = iSqrVars.get(node);
            }
            else {
                unMeanIns.put(node, unMeanInEntry = new TreeMap<>());
                vars.put(node, varEntry = new TreeMap<>());
                iSqrVars.put(node, iSqrVarEntry = new TreeMap<>());
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
                node.setMatrix(inputIndex, unMeanIn.multiply(iSqrVar));
            }
            else node.setMatrix(inputIndex, unMeanIn);
        }
        else {
            Matrix unMeanIn = inputMatrix.subtract(inputMatrix.mean());
            if (!meanOnly) node.setMatrix(inputIndex, unMeanIn.multiply(1 / Math.sqrt(inputMatrix.var() + epsilon)));
            else node.setMatrix(inputIndex, unMeanIn);
        }
    }

    /**
     * Executes backward propagation step for layer normalization.<br>
     * Calculates gradients backwards at step end for previous layer.<br>
     *
     * @param node node for normalization.
     * @param outputIndex input index for normalization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backward(Node node, int outputIndex) throws MatrixException {
        if (!unMeanIns.containsKey(node)) return;

        int size = node.getGradient(outputIndex).getSize();
        Matrix outputGradMatrix = node.getGradient(outputIndex);
        Matrix unMeanIn = unMeanIns.get(node).get(outputIndex);
        double dsigma = -0.5 * outputGradMatrix.multiply(unMeanIn).sum() / size;
        double dmu = -1 * outputGradMatrix.sum() / size;
        if (!meanOnly) {
            dsigma *= Math.pow(vars.get(node).get(outputIndex) + epsilon, -1.5);
            dmu *= iSqrVars.get(node).get(outputIndex);
            node.setGradient(outputIndex, outputGradMatrix.multiply(iSqrVars.get(node).get(outputIndex)).add(unMeanIn.multiply(2 * dsigma)).add(dmu));
        }
        else {
            node.setGradient(outputIndex, outputGradMatrix.add(unMeanIn.multiply(2 * dsigma)).add(dmu));
        }
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