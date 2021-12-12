/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.optimization;

import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.UnaryFunctionType;

import java.util.HashMap;

/**
 * Class that implements AMSGrad optimizer.<br>
 * <br>
 * Reference: https://towardsdatascience.com/10-gradient-descent-optimisation-algorithms-86989510b5e9 <br>
 *
 */
public class AMSGrad extends AbstractOptimizer {

    /**
     * Parameter name types for AMSGrad.
     *     - learningRate: learning rate for optimizer. Default value 0.001.<br>
     *     - beta1: beta1 value for optimizer. Default value 0.9.<br>
     *     - beta2: beta2 value for optimizer. Default value 0.999.<br>
     *
     */
    private final static String paramNameTypes = "(learningRate:DOUBLE), " +
            "(beta1:DOUBLE), " +
            "(beta2:DOUBLE)";

    /**
     * Learning rate for AMSGrad. Default value 0.001.
     *
     */
    private double learningRate;

    /**
     * Beta1 term for AMSGrad. Default value 0.9.
     *
     */
    private double beta1;

    /**
     * Beta2 term for AMSGrad. Default value 0.999.
     *
     */
    private double beta2;

    /**
     * Hash map to store first moments (means).
     *
     */
    private transient HashMap<Matrix, Matrix> m;

    /**
     * Hash map to store second moments (uncentered variances).
     *
     */
    private transient HashMap<Matrix, Matrix> v;

    /**
     * Default constructor for AMSGrad.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AMSGrad() throws DynamicParamException {
        super(OptimizationType.AMSGRAD, AMSGrad.paramNameTypes);
    }

    /**
     * Constructor for AMSGrad.
     *
     * @param params parameters for AMSGrad.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AMSGrad(String params) throws DynamicParamException {
        super(OptimizationType.AMSGRAD, AMSGrad.paramNameTypes, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        learningRate = 0.001;
        beta1 = 0.9;
        beta2 = 0.999;
    }

    /**
     * Sets parameters used for AMSGrad.<br>
     * <br>
     * Supported parameters are:<br>
     *     - learningRate: learning rate for optimizer. Default value 0.001.<br>
     *     - beta1: beta1 value for optimizer. Default value 0.9.<br>
     *     - beta2: beta2 value for optimizer. Default value 0.999.<br>
     *
     * @param params parameters used for AMSGrad.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("learningRate")) learningRate = params.getValueAsDouble("learningRate");
        if (params.hasParam("beta1")) beta1 = params.getValueAsDouble("beta1");
        if (params.hasParam("beta2")) beta2 = params.getValueAsDouble("beta2");
    }

    /**
     * Resets optimizer state.
     *
     */
    public void reset() {
        m = new HashMap<>();
        v = new HashMap<>();
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
        if (m == null) m = new HashMap<>();
        if (v == null) v = new HashMap<>();

        Matrix mM;
        if (m.containsKey(matrix)) mM = m.get(matrix);
        else m.put(matrix, mM = new DMatrix(matrix.getRows(), matrix.getColumns()));

        Matrix vM;
        if (v.containsKey(matrix)) vM = v.get(matrix);
        else v.put(matrix, vM = new DMatrix(matrix.getRows(), matrix.getColumns()));

        // mt = β1*mt − 1 + (1 − β1)*gt
        mM.multiply(beta1).add(matrixGradient.multiply(1 - beta1), mM);

        // vt = β2*vt − 1 + (1 − β2)*g2t
        Matrix vM_temp = vM.multiply(beta2).add(matrixGradient.power(2).multiply(1 - beta2));

        // vt = max(vt, vt-1)
        vM_temp.max(v.get(matrix), vM);

        // θt+1 = θt − η / (√^vt + ϵ) * mt
        double epsilon = 10E-8;
        matrix.subtract(mM.divide(vM.add(epsilon).apply(UnaryFunctionType.SQRT)).multiply(learningRate), matrix);
    }

}

