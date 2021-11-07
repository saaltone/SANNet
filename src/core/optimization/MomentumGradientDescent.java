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

import java.util.HashMap;

/**
 * Class that implements Gradient Descent with Momentum optimizer.<br>
 * <br>
 * Reference: http://ruder.io/optimizing-gradient-descent/ <br>
 *
 */
public class MomentumGradientDescent extends AbstractOptimizer {

    /**
     * Parameter name types for MomentumGradientDescent.
     *     - learningRate: learning rate for optimizer. Default value 0.001.<br>
     *     - mu: mu (momentum) value for optimizer. Default value 0.0001.<br>
     *
     */
    private final static String paramNameTypes = "(learningRate:DOUBLE), " +
            "(mu:DOUBLE)";

    /**
     * Learning rate for Momentum Gradient Descent. Default value 0.001.
     *
     */
    private double learningRate;

    /**
     * Momentum term for Momentum Gradient Descent. Default value 0.0001.
     *
     */
    private double mu;

    /**
     * Hash map to store previous gradients.
     *
     */
    private HashMap<Matrix, Matrix> dPrev;

    /**
     * Default constructor for Momentum Gradient Descent.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public MomentumGradientDescent() throws DynamicParamException {
        super(OptimizationType.MOMENTUM_GRADIENT_DESCENT, MomentumGradientDescent.paramNameTypes);
    }

    /**
     * Constructor for Momentum Gradient Descent.
     *
     * @param params parameters for Momentum Gradient Descent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public MomentumGradientDescent(String params) throws DynamicParamException {
        super(OptimizationType.MOMENTUM_GRADIENT_DESCENT, MomentumGradientDescent.paramNameTypes, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        learningRate = 0.001;
        mu = 0.0001;
    }

    /**
     * Sets parameters used for Momentum Gradient Descent.<br>
     * <br>
     * Supported parameters are:<br>
     *     - learningRate: learning rate for optimizer. Default value 0.001.<br>
     *     - mu: mu (momentum) value for optimizer. Default value 0.0001.<br>
     *
     * @param params parameters used for Momentum Gradient Descent.
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
        Matrix dMPrev;
        if (dPrev.containsKey(matrix)) dMPrev = dPrev.get(matrix);
        else dPrev.put(matrix, dMPrev = new DMatrix(matrix.getRows(), matrix.getColumns()));

        // θt+1=θt+μtvt−εt∇f(θt)
        Matrix dMDelta = dMPrev.multiply(mu).subtract(matrixGradient.multiply(learningRate));

        matrix.add(dMDelta, matrix);

        // vt+1=μtvt−εt∇f(θt)
        dPrev.put(matrix, dMDelta);

    }

}

