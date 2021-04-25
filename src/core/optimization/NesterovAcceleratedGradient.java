/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.optimization;

import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashMap;

/**
 * Class that implements Nesterov's Accelerated Gradient Descent optimizer.<br>
 * <br>
 * Reference: http://ruder.io/optimizing-gradient-descent/ <br>
 *
 */
public class NesterovAcceleratedGradient implements Optimizer, Serializable {

    @Serial
    private static final long serialVersionUID = -783588127072068825L;

    /**
     * Optimization type.
     *
     */
    private final OptimizationType optimizationType = OptimizationType.NESTEROV_ACCELERATED_GRADIENT;

    /**
     * Learning rate for Nesterov Accelerated Gradient. Default value 0.001.
     *
     */
    private double learningRate = 0.001;

    /**
     * Momentum term for Nesterov Accelerated Gradient. Default value 0.0001.
     *
     */
    private double mu = 0.0001;

    /**
     * Hash map to store previous gradients.
     *
     */
    private HashMap<Matrix, Matrix> dPrev;

    /**
     * Hash map to store previous velocities.
     *
     */
    private HashMap<Matrix, Matrix> vPrev;

    /**
     * Default constructor for Nesterov Accelerated Gradient.
     *
     */
    public NesterovAcceleratedGradient() {
    }

    /**
     * Constructor for Nesterov Accelerated Gradient.
     *
     * @param params parameters for Nesterov Accelerated Gradient.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public NesterovAcceleratedGradient(String params) throws DynamicParamException {
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for Nesterov Accelerated Descent.
     *
     * @return parameters used for Nesterov Accelerated Descent.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("learningRate", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("mu", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for Nesterov Accelerated Descent.<br>
     * <br>
     * Supported parameters are:<br>
     *     - learningRate: learning rate for optimizer. Default value 0.001.<br>
     *     - mu: mu (momentum) value for optimizer. Default value 0.0001.<br>
     *
     * @param params parameters used for Nesterov Accelerated Descent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("learningRate")) learningRate = params.getValueAsDouble("learningRate");
        if (params.hasParam("mu")) mu = params.getValueAsDouble("mu");
    }

    /**
     * Resets optimizer state.
     *
     */
    public void reset() {
        dPrev = new HashMap<>();
        vPrev = new HashMap<>();
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
        if (dPrev == null) dPrev = new HashMap<>();
        if (vPrev == null) vPrev = new HashMap<>();
        Matrix dMPrev;
        if (dPrev.containsKey(matrix)) dMPrev = dPrev.get(matrix);
        else dPrev.put(matrix, dMPrev = new DMatrix(matrix.getRows(), matrix.getColumns()));

        Matrix vMPrev;
        if (vPrev.containsKey(matrix)) vMPrev = dPrev.get(matrix);
        else vPrev.put(matrix, vMPrev = new DMatrix(matrix.getRows(), matrix.getColumns()));

        // vt=μvt−1−ϵ∇f(θt−1+μvt−1)
        Matrix vM = vMPrev.multiply(mu).subtract(dMPrev.add(vMPrev.multiply(mu)).multiply(learningRate));

        matrix.add(vM, matrix);

        vPrev.put(matrix, vM);
        dPrev.put(matrix, matrixGradient);

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

