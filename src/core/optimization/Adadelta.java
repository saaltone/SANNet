/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.optimization;

import utils.*;
import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.UnaryFunctionType;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Class that implements Adadelta optimizer.<br>
 * <br>
 * Reference: http://ruder.io/optimizing-gradient-descent/<br>
 *
 */
public class Adadelta implements Optimizer, Serializable {

    private static final long serialVersionUID = 1620048040058081811L;

    /**
     * Learning rate for Adadelta. Default value 1.
     *
     */
    private double learningRate = 1;

    /**
     * Gamma term for Adadelta. Default value 0.95.
     *
     */
    private double gamma = 0.95;

    /**
     * Hash map to store gradients from previous steps.
     *
     */
    private transient HashMap<Matrix, Matrix> eg2;

    /**
     * Hash map to store gradient deltas from previous steps.
     *
     */
    private transient HashMap<Matrix, Matrix> ed2;

    /**
     * Default constructor for Adadelta.
     *
     */
    public Adadelta() {
    }

    /**
     * Constructor for Adadelta.
     *
     * @param params parameters for Adadelta.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Adadelta(String params) throws DynamicParamException {
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for Adadelta.
     *
     * @return parameters used for Adadelta.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("learningRate", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("gamma", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for Adadelta.<br>
     * <br>
     * Supported parameters are:<br>
     *     - learningRate: learning rate for optimizer. Default value 0.1.<br>
     *     - gamma: gamma value for optimizer. Default value 0.95.<br>
     *
     * @param params parameters used for Adadelta.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("learningRate")) learningRate = params.getValueAsDouble("learningRate");
        if (params.hasParam("gamma")) gamma = params.getValueAsDouble("gamma");
    }

    /**
     * Resets optimizer state.
     *
     */
    public void reset() {
        eg2 = new HashMap<>();
        ed2 = new HashMap<>();
    }

    /**
     * Optimizes given weight (W) and bias (B) pair with given gradients respectively.
     *
     * @param W weight matrix to be optimized.
     * @param dW weight gradients for optimization step.
     * @param B bias matrix to be optimized.
     * @param dB bias gradients for optimization step.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void optimize(Matrix W, Matrix dW, Matrix B, Matrix dB) throws MatrixException {
        optimize(W, dW);
        optimize(B, dB);
    }

    /**
     * Optimizes single matrix (M) using calculated matrix gradient (dM).<br>
     * Matrix can be for example weight or bias matrix with gradient.<br>
     *
     * @param M matrix to be optimized.
     * @param dM matrix gradients for optimization step.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void optimize(Matrix M, Matrix dM) throws MatrixException {
        if (eg2 == null) eg2 = new HashMap<>();
        Matrix mEg2;
        if (eg2.containsKey(M)) mEg2 = eg2.get(M);
        else eg2.put(M, mEg2 = new DMatrix(M.getRows(), M.getCols()));

        if (ed2 == null) ed2 = new HashMap<>();
        Matrix mEd2;
        if (ed2.containsKey(M)) mEd2 = ed2.get(M);
        else ed2.put(M, mEd2 = new DMatrix(M.getRows(), M.getCols()));

        mEg2 = mEg2.multiply(gamma).add(dM.power(2).multiply(1 - gamma));

        double epsilon = 10E-8;
        Matrix Ed = mEd2.add(epsilon).apply(UnaryFunctionType.SQRT).multiply(mEg2.add(epsilon).apply(UnaryFunctionType.SQRT).apply(UnaryFunctionType.MULINV)).multiply(dM);
        M.subtract(Ed.multiply(learningRate), M);
        mEd2 = mEd2.multiply(gamma).add(Ed.power(2).multiply(1 - gamma));

        eg2.put(M, mEg2);
        ed2.put(M, mEd2);
    }

}
