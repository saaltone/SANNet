/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.optimization;

import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.UnaryFunctionType;

import java.util.HashMap;

/**
 * Implements Rectified Adam optimizer.<br>
 * <br>
 * Reference: <a href="https://arxiv.org/abs/1908.03265">...</a> <br>
 *
 */
public class RAdam extends AbstractOptimizer {

    /**
     * Parameter name types for RAdam.
     *     - learningRate: learning rate for optimizer. Default value 0.001.<br>
     *     - beta1: beta1 value for optimizer. Default value 0.9.<br>
     *     - beta2: beta2 value for optimizer. Default value 0.999.<br>
     *
     */
    private final static String paramNameTypes = "(learningRate:DOUBLE), " +
            "(beta1:DOUBLE), " +
            "(beta2:DOUBLE)";

    /**
     * Learning rate for RAdam. Default value 0.010.
     *
     */
    private double learningRate;

    /**
     * Beta1 term for RAdam. Default value 0.9.
     *
     */
    private double beta1;

    /**
     * Beta2 term for RAdam. Default value 0.999.
     *
     */
    private double beta2;

    /**
     * Hash map to store iteration counts.
     *
     */
    private final HashMap<Matrix, Integer> iterations = new HashMap<>();

    /**
     * Maximum length of approximated SMA.
     *
     */
    private double pinf;

    /**
     * Hash map to store first moments (means).
     *
     */
    private final HashMap<Matrix, Matrix> m = new HashMap<>();

    /**
     * Hash map to store second moments (uncentered variances).
     *
     */
    private final HashMap<Matrix, Matrix> v = new HashMap<>();

    /**
     * Default constructor for RAdam.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public RAdam() throws DynamicParamException {
        super(OptimizationType.RADAM, RAdam.paramNameTypes);
    }

    /**
     * Constructor for RAdam.
     *
     * @param params parameters for RAdam.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public RAdam(String params) throws DynamicParamException {
        super(OptimizationType.RADAM, RAdam.paramNameTypes, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        learningRate = 0.001;
        beta1 = 0.9;
        beta2 = 0.999;
        pinf = 2 / (1 - beta2) - 1;
    }

    /**
     * Sets parameters used for RAdam.<br>
     * <br>
     * Supported parameters are:<br>
     *     - learningRate: learning rate for optimizer. Default value 0.001.<br>
     *     - beta1: beta1 value for optimizer. Default value 0.9.<br>
     *     - beta2: beta2 value for optimizer. Default value 0.999.<br>
     *
     * @param params parameters used for RAdam.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("learningRate")) learningRate = params.getValueAsDouble("learningRate");
        if (params.hasParam("beta1")) beta1 = params.getValueAsDouble("beta1");
        if (params.hasParam("beta2")) beta2 = params.getValueAsDouble("beta2");
        pinf = 2 / (1 - beta2) - 1;
    }

    /**
     * Resets optimizer state.
     *
     */
    public void reset() {
        iterations.clear();
        m.clear();
        v.clear();
    }

    /**
     * Optimizes single matrix (M) using calculated matrix gradient (dM).<br>
     * Matrix can be for example weight or bias matrix with gradient.<br>
     *
     * @param matrix matrix to be optimized.
     * @param matrixGradient matrix gradients for optimization step.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void optimize(Matrix matrix, Matrix matrixGradient) throws MatrixException, DynamicParamException {
        int iteration;
        iterations.put(matrix, iteration = iterations.getOrDefault(matrix, 0) + 1);

        Matrix mM = m.get(matrix);
        mM = mM == null ? matrixGradient.multiply(1 - beta1) : mM.multiply(beta1).add(matrixGradient.multiply(1 - beta1));
        setParameterMatrix(m, matrix, mM);

        Matrix vM = v.get(matrix);
        vM = vM == null ? matrixGradient.power(2).multiply(1 - beta2) : vM.multiply(beta2).add(matrixGradient.power(2).multiply(1 - beta2));
        setParameterMatrix(v, matrix, vM);

        double beta1Iteration = Math.pow(beta1, iteration);
        double beta2Iteration = Math.pow(beta2, iteration);

        Matrix mMhat = mM.divide(1 - beta1Iteration);

        double stepSize = learningRate;
        double pt = pinf - 2 * iteration * beta2Iteration / (1 - beta2Iteration);
        if (pt > 4) {
            stepSize *=  Math.sqrt((1 - beta2Iteration) * ((pt - 4) * (pt - 2) * pinf) / ((pinf - 4) * (pinf - 2) * pt));
            double epsilon = 10E-8;
            matrix.subtractBy(mMhat.divide(vM.apply(UnaryFunctionType.SQRT).add(epsilon)).multiply(stepSize));
        }
        else {
            matrix.subtractBy(mMhat.multiply(stepSize));
        }
    }

}

