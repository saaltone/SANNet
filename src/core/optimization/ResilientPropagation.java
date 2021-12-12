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
 * Class that implements Resilient Propagation optimizer.<br>
 * Resilient Propagation is full batch algorithm.<br>
 * <br>
 * Reference: http://130.243.105.49/~lilien/ml/seminars/2007_03_12c-Markus_Ingvarsson-RPROP.pdf <br>
 *
 */
public class ResilientPropagation extends AbstractOptimizer {

    /**
     * Hash map to store previous gradients.
     *
     */
    private transient HashMap<Matrix, Matrix> dPrev;

    /**
     * Hash map to store previous directions.
     *
     */
    private transient HashMap<Matrix, Matrix> wPrev;

    /**
     * Default constructor for Resilient Propagation.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public ResilientPropagation() throws DynamicParamException {
        super(OptimizationType.RESILIENT_PROPAGATION, null);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
    }

    /**
     * Sets parameters used for ResilientPropagation.<br>
     *
     * @param params parameters used for ResilientPropagation.
     */
    public void setParams(DynamicParam params) {
    }

    /**
     * Resets optimizer state.
     *
     */
    public void reset() {
        dPrev = new HashMap<>();
        wPrev = new HashMap<>();
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
        if (dPrev == null) dPrev = new HashMap<>();
        if (wPrev == null) wPrev = new HashMap<>();

        Matrix dMPrev;
        if (dPrev.containsKey(matrix)) dMPrev = dPrev.get(matrix);
        else dPrev.put(matrix, dMPrev = matrixGradient);

        Matrix WPrev;
        if (wPrev.containsKey(matrix)) WPrev = wPrev.get(matrix);
        else wPrev.put(matrix, WPrev = new DMatrix(matrix.getRows(), matrix.getColumns()));

        Matrix dWDir = dMPrev.sgnmul(matrixGradient);

        Matrix.MatrixBinaryOperation rpropRule = (value1, value2) -> value1 == -1 ? Math.max(0.5 * value2, 10E-6) : value1 == 1 ? Math.min(1.2 * value2, 50) : value2;
        wPrev.put(matrix, WPrev = dWDir.applyBi(WPrev, rpropRule));

        matrix.subtract(matrixGradient.apply(UnaryFunctionType.SGN).multiply(WPrev), matrix);

        dPrev.put(matrix, dWDir.applyBi(matrixGradient, (value1, value2) -> value1 == -1 ? 0 : value2));
    }

}
