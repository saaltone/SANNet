/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.regularization;

import utils.DynamicParam;
import utils.DynamicParamException;
import utils.Sequence;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serializable;
import java.util.HashMap;

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
     */
    public Lp_Regularization() {
    }

    /**
     * Constructor for Lp regularization class.
     *
     * @param params parameters for Lp regularization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Lp_Regularization(String params) throws DynamicParamException {
        this.setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for Lp regularization.
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
    public void setTraining(boolean isTraining) {
    }

    /**
     * Not used.
     *
     */
    public void reset() {}

    /**
     * Not used.
     *
     * @param sequence input sequence.
     */
    public void forward(Sequence sequence) {}

    /**
     * Not used.
     *
     * @param W weight matrix.
     */
    public void forward(Matrix W) {}

    /**
     * Calculates and returns cumulated error from Lp regularization.<br>
     * This is added to the total output error of neural network.<br>
     *
     * @param W weight matrix.
     * @return cumulated error from L2 regularization.
     */
    public double error(Matrix W) {
        return lambda * W.norm(p);
    }

    /**
     * Regulates weights by calculating p- norm of weights and adding it to weight gradient sum.
     *
     * @param W weight matrix.
     * @param dWSum gradient sum of weight.
     */
    public void backward(Matrix W, Matrix dWSum) throws MatrixException {
        Matrix.MatrixUnaryOperation function = (value) -> value != 0 ? p * lambda * Math.pow(Math.abs(value), p - 1) / value : 0;
        dWSum.add(W.apply(function), dWSum);
    }

}
