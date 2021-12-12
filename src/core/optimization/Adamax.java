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
 * Class that implements Adamax optimizer.<br>
 * <br>
 * Reference: http://ruder.io/optimizing-gradient-descent/ <br>
 *
 */
public class Adamax extends AbstractOptimizer {

    /**
     * Parameter name types for Adamax.
     *     - learningRate: learning rate for optimizer. Default value 0.001.<br>
     *     - beta1: beta1 value for optimizer. Default value 0.9.<br>
     *     - beta2: beta2 value for optimizer. Default value 0.999.<br>
     *
     */
    private final static String paramNameTypes = "(learningRate:DOUBLE), " +
            "(beta1:DOUBLE), " +
            "(beta2:DOUBLE)";

    /**
     * Learning rate for Adamax. Default value 0.001.
     *
     */
    private double learningRate;

    /**
     * Beta1 term for Adamax. Default value 0.9.
     *
     */
    private double beta1;

    /**
     * Beta2 term for Adamax. Default value 0.999.
     *
     */
    private double beta2;

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
     * Default constructor for Adamax.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Adamax() throws DynamicParamException {
        super(OptimizationType.ADAMAX, Adamax.paramNameTypes);
    }

    /**
     * Constructor for Adamax.
     *
     * @param params parameters for Adamax.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Adamax(String params) throws DynamicParamException {
        super(OptimizationType.ADAMAX, Adamax.paramNameTypes, params);
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
     * Sets parameters used for Adamax.<br>
     * <br>
     * Supported parameters are:<br>
     *     - learningRate: learning rate for optimizer. Default value 0.001.<br>
     *     - beta1: beta1 value for optimizer. Default value 0.9.<br>
     *     - beta2: beta2 value for optimizer. Default value 0.999.<br>
     *
     * @param params parameters used for Adamax.
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
     * Optimizes single matrix (M) using calculated matrix gradient (dM).<br>
     * Matrix can be for example weight or bias matrix with gradient.<br>
     *
     * @param matrix matrix to be optimized.
     * @param matrixGradient matrix gradients for optimization step.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void optimize(Matrix matrix, Matrix matrixGradient) throws MatrixException, DynamicParamException {
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

        Matrix matrixGradientAbs = matrixGradient.apply(UnaryFunctionType.ABS);

        // mt = β1*mt−1 + (1 − β1)*gt
        mM.multiply(beta1).add(matrixGradient.multiply(1 - beta1), mM);

        // vt = β2*vt−1 + (1 − β2)*|gt|
        vM.multiply(beta2).add(matrixGradientAbs.multiply(1 - beta2), vM);

        // mt = mt / (1 − βt1)
        Matrix mM_hat = mM.divide(1 - Math.pow(beta1, iteration));

        // ut= max(β2⋅vt−1,|gt|)
        Matrix uM = (vM.multiply(beta2)).max(matrixGradientAbs);

        // θt+1 = θt − η / ut * mt
        matrix.subtract(mM_hat.divide(uM).multiply(learningRate), matrix);
    }

}

