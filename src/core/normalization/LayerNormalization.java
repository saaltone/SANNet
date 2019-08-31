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
 * Class that implements Layer Normalization for neural network layer.<br>
 * Layer Normalization is particularly well suited for recurrent neural networks as it normalizes each sample independently.<br>
 * <br>
 * Reference: https://www.cs.toronto.edu/~hinton/absps/LayerNormalization.pdf<br>
 *
 */
public class LayerNormalization implements Normalization, Serializable {

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
     * Tree map to store inputs normalized by mean.<br>
     * Used in backward propagation step for gradient calculation.<br>
     *
     */
    private transient TreeMap<Integer, Matrix> unMeanIns;

    /**
     * Tree to store variances of each sample.<br>
     * Variance is calculated over single sample over all features.<br>
     *
     */
    private transient TreeMap<Integer, Double> vars;

    /**
     * Tree to store inverse squared variances (1 / squared variance) of each sample.<br>
     * This intermediate value is used for backpropagation gradient calculation.<br>
     *
     */
    private transient TreeMap<Integer, Double> iSqrVars;

    /**
     * True if layer normalization is used calculation only with mean and variance excluded.
     *
     */
    private boolean meanOnly = false;

    /**
     * Constructor for layer normalization class.
     *
     * @param connector reference to connector between previous and next layer.
     */
    public LayerNormalization(Connector connector) {
        this.connector = connector;
    }

    /**
     * Constructor for layer normalization class.
     *
     * @param connector reference to connector between previous and next layer.
     * @param params parameters for layer normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public LayerNormalization(Connector connector, String params) throws DynamicParamException {
        this(connector);
        this.setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Gets parameters used for layer normalization.
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
        unMeanIns = new TreeMap<>();
        vars = new TreeMap<>();
        iSqrVars = new TreeMap<>();
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
     * @param ins input samples for forward step.
     * @param channels not used. Only relevant for convolutional layer.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forwardPre(TreeMap<Integer, Matrix> ins, int channels) throws MatrixException {
        if (ins.get(ins.firstKey()).getSize() < 2) return;
        /**
         * Epsilon term for layer normalization. Default value 10E-8.<br>
         * Term provides mathematical stability for normalizer.<br>
         *
         */
        double epsilon = 10E-8;
        if (isTraining) {
            reset();
            for (Integer index : ins.keySet()) {
                Matrix input = ins.get(index);

                // Calculate mean
                double mean = input.mean();

                Matrix unMeanIn = input.subtract(mean);
                unMeanIns.put(index, unMeanIn);

                double iSqrVar;
                if (!meanOnly) {
                    // Calculate variance
                    double var = input.var() + epsilon;
                    vars.put(index, var);
                    iSqrVar = 1 / Math.sqrt(var);
                    iSqrVars.put(index, iSqrVar);

                    ins.put(index, unMeanIn.multiply(iSqrVar));
                }
                else ins.put(index, unMeanIn);
            }
        }
        else {
            for (Integer index : ins.keySet()) {
                Matrix input = ins.get(index);
                Matrix unMeanIn = input.subtract(input.mean());
                if (!meanOnly) ins.put(index, unMeanIn.multiply(1 / Math.sqrt(input.var() + epsilon)));
                else ins.put(index, unMeanIn);
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
     * Executes backward propagation step for layer normalization.<br>
     * Calculates gradients backwards at step end for previous layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backward() throws MatrixException {
        TreeMap<Integer, Matrix> dEos = connector.getNLayer().getdEos();
        double size = dEos.get(dEos.firstKey()).getSize();
        if (size < 2) return;

        for (Integer index : dEos.keySet()) {
            Matrix dEo = dEos.get(index);
            double var = vars.get(index);
            double iSqrVar = iSqrVars.get(index);
            Matrix unMeanIn = unMeanIns.get(index);
            Matrix dEoP;
            double dsigma = -0.5 * dEo.multiply(unMeanIn).sum() / size;
            double dmu = -1 * dEo.sum() / size;
            if (!meanOnly) {
                dsigma *= Math.pow(var, -1.5);
                dmu *= iSqrVar;
                dEoP = dEo.multiply(iSqrVar).add(unMeanIn.multiply(2 * dsigma)).add(dmu);
            }
            else {
                dEoP = dEo.add(unMeanIn.multiply(2 * dsigma)).add(dmu);
            }
            dEos.put(index, dEoP);
        }
    }

}