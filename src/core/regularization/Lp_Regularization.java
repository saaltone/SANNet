/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.regularization;

import core.layer.Connector;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.Matrix;
import utils.MatrixException;

import java.io.Serializable;
import java.util.HashMap;
import java.util.TreeMap;

/**
 * Implements Lp regularization (experimental). P here is any norm higher or equal to 1.<br>
 * Setting p = 1 this becomes L1 regularization and setting p = 2 this becomes L2 regularization.<br>
 * <br>
 * This is experimental regularization method.<br>
 *
 */
public class Lp_Regularization implements Regularization, Serializable {

    private static final long serialVersionUID = -7833984930510523396L;

    /**
     * Reference to connector between previous and next layer.
     *
     */
    private Connector connector;

    /**
     * Regularization rate.
     *
     */
    private double lambda = 0.01;

    /**
     * Order of norm.
     *
     */
    private int p = 3;

    /**
     * Constructor for Lp regularization class.
     *
     * @param connector reference to connector between previous and next layer.
     * @param toHiddenLayer true if next layer if hidden layer otherwise false.
     */
    public Lp_Regularization(Connector connector, boolean toHiddenLayer) {
        this.connector = connector;
    }

    /**
     * Constructor for Lp regularization class.
     *
     * @param connector reference to connector between previous and next layer.
     * @param toHiddenLayer true if next layer if hidden layer otherwise false.
     * @param params parameters for Lp regularization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Lp_Regularization(Connector connector, boolean toHiddenLayer, String params) throws DynamicParamException {
        this(connector, toHiddenLayer);
        this.setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Gets parameters used for Lp regularization.
     *
     * @return parameters used for Lp regularization.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("lambda", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("p", DynamicParam.ParamType.INT);
        return paramDefs;
    }

    /**
     * Sets parameters used for Lp regularization.<br>
     * <br>
     * Supported parameters are:<br>
     *     - lambda: lambda value for regularization. Default value: 0.01.<br>
     *     - p: p norm of normalizer. Default 3.<br>
     *
     * @param params parameters used for Lp regularization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("lambda")) lambda = params.getValueAsDouble("lambda");
        if (params.hasParam("p")) p = params.getValueAsInteger("p");
    }

    /**
     * Not used.
     *
     */
    public void reset() {}

    /**
     * Not used.
     *
     */
    public void setTraining(boolean isTraining) {
    }

    /**
     * Not used.
     *
     * @param ins input samples for forward step.
     * @param index if index is zero or positive value operation is executed for this sample. if index is -1 operation is executed for all samples.
     */
    public void forwardPre(TreeMap<Integer, Matrix> ins, int index) {}

    /**
     * Not used.
     *
     * @param outs output samples for forward step.
     */
    public void forwardPost(TreeMap<Integer, Matrix> outs) {}

    /**
     * Calculates and returns cumulated error from Lp regularization.<br>
     * This is added to the total output error of neural network.<br>
     *
     * @return cumulated error from Lp regularization.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double error() throws MatrixException {
        double error = 0;
        for (Matrix W : connector.getReg()) {
            error += lambda * W.norm(p);
        }
        return error;
    }

    /**
     * Not used.
     *
     * @param index if index is zero or positive value operation is executed for this sample. if index is -1 operation is executed for all samples.
     */
    public void backward(int index) {
    }

    /**
     * Regulates weights by calculating p- norm of weights and adding it to weight gradient sum.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void update() throws MatrixException {
        Matrix.MatrixUniOperation function = (value) -> value != 0 ? p * lambda * Math.pow(Math.abs(value), p - 1) / value : 0;
        for (Matrix W : connector.getReg()) {
            Matrix dWSum = connector.getdWsSums(W);
            dWSum.add(W.apply(function), dWSum);
        }
    }

}
