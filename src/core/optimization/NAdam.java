/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.optimization;

import utils.*;
import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.UnaryFunctionType;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Class that implements Nadam optimizer.<br>
 * <br>
 * Reference: http://ruder.io/optimizing-gradient-descent/<br>
 *
 */
public class NAdam implements Optimizer, Serializable {

    private static final long serialVersionUID = 6575858816658305979L;

    /**
     * Optimization type.
     *
     */
    private final OptimizationType optimizationType;

    /**
     * Learning rate for Nadam. Default value 0.001.
     *
     */
    private double learningRate = 0.001;

    /**
     * Beta1 term for Nadam. Default value 0.9.
     *
     */
    private double beta1 = 0.9;

    /**
     * Beta2 term for Nadam. Default value 0.999.
     *
     */
    private double beta2 = 0.999;

    /**
     * Hash map to store iteration counts.
     *
     */
    private transient HashMap<Matrix, Integer> iterations;

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
     * Default constructor for Nadam.
     *
     * @param optimizationType optimizationType.
     */
    public NAdam(OptimizationType optimizationType) {
        this.optimizationType = optimizationType;
    }

    /**
     * Constructor for Nadam.
     *
     * @param optimizationType optimizationType.
     * @param params parameters for Nadam.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public NAdam(OptimizationType optimizationType, String params) throws DynamicParamException {
        this(optimizationType);
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for Nadam.
     *
     * @return parameters used for Nadam.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("learningRate", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("beta1", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("beta2", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for Nadam.<br>
     * <br>
     * Supported parameters are:<br>
     *     - learningRate: learning rate for optimizer. Default value 0.001.<br>
     *     - beta1: beta1 value for optimizer. Default value 0.9.<br>
     *     - beta2: beta2 value for optimizer. Default value 0.999.<br>
     *
     * @param params parameters used for Nadam.
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
        iterations = new HashMap<>();
        m = new HashMap<>();
        v = new HashMap<>();
    }

    /**
     * Optimizes given weight (W) and bias (B) pair with given gradients respectively.
     *
     * @param weight weight matrix to be optimized.
     * @param weightGradient weight gradients for optimization step.
     * @param bias bias matrix to be optimized.
     * @param biasGradient bias gradients for optimization step.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void optimize(Matrix weight, Matrix weightGradient, Matrix bias, Matrix biasGradient) throws MatrixException {
        optimize(weight, weightGradient);
        optimize(bias, biasGradient);
    }

    /**
     * Optimizes single matrix (M) using calculated matrix gradient (dM).<br>
     * Matrix can be for example weight or bias matrix with gradient.<br>
     *
     * @param matrix matrix to be optimized.
     * @param matrixGradient matrix gradients for optimization step.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void optimize(Matrix matrix, Matrix matrixGradient) throws MatrixException {
        if (iterations == null) iterations = new HashMap<>();
        if (m == null) m = new HashMap<>();
        if (v == null) v = new HashMap<>();

        int iteration;
        if (iterations.containsKey(matrix)) iterations.put(matrix, iteration = iterations.get(matrix) + 1);
        else iterations.put(matrix, iteration = 1);

        Matrix mM;
        if (m.containsKey(matrix)) mM = m.get(matrix);
        else m.put(matrix, mM = new DMatrix(matrix.getRows(), matrix.getColumns()));

        Matrix vM;
        if (v.containsKey(matrix)) vM = v.get(matrix);
        else v.put(matrix, vM = new DMatrix(matrix.getRows(), matrix.getColumns()));

        // mt = β1*mt − 1 + (1 − β1)*gt
        mM.multiply(beta1).add(matrixGradient.multiply(1 - beta1), mM);

        // vt = β2*vt − 1 + (1 − β2)*g2t
        vM.multiply(beta2).add(matrixGradient.power(2).multiply(1 - beta2), vM);

        // mt = mt / (1 − βt1)
        Matrix mM_hat = mM.divide(1 - Math.pow(beta1, iteration));

        // vt = vt / (1 − βt2)
        Matrix vM_hat = vM.divide(1 - Math.pow(beta2, iteration));

        // θt+1 = θt − η / (√^vt+ϵ) * (β1 * mt + (1 − β1) * gt / (1 − βt1))
        double epsilon = 10E-8;
        matrix.subtract(mM_hat.multiply(beta1).add(matrixGradient.multiply((1 - beta1) / (1 - Math.pow(beta1, iteration)))).divide(vM_hat.add(epsilon).apply(UnaryFunctionType.SQRT)).multiply(learningRate), matrix);
    }

    /**
     * Returns name of optimizer.
     *
     * @return name of optimizer.
     */
    public String getName() {
        return optimizationType.toString();
    }

}
