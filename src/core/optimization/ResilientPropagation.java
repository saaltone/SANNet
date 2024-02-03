/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.optimization;

import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.util.HashMap;

/**
 * Implements Resilient Propagation optimizer.<br>
 * Resilient Propagation is full batch algorithm.<br>
 * <br>
 * Reference: <a href="http://130.243.105.49/~lilien/ml/seminars/2007_03_12c-Markus_Ingvarsson-RPROP.pdf">...</a> <br>
 *
 */
public class ResilientPropagation extends AbstractOptimizer {

    /**
     * Hash map to store previous gradients.
     *
     */
    private final HashMap<Matrix, Matrix> dPrev = new HashMap<>();

    /**
     * Hash map to store previous directions.
     *
     */
    private final HashMap<Matrix, Matrix> wPrev = new HashMap<>();

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
        dPrev.clear();
        wPrev.clear();
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
        Matrix WPrev = wPrev.get(matrix);
        if (WPrev == null) wPrev.put(matrix, WPrev = new DMatrix(matrix.getRows(), matrix.getColumns(), matrix.getDepth()));

        Matrix dMPrev = dPrev.computeIfAbsent(matrix, k -> matrixGradient);
        Matrix dWDir = dMPrev.sgnmul(matrixGradient);

        WPrev = dWDir.applyBi(WPrev, new BinaryFunction((value1, value2) -> value1 == -1 ? Math.max(0.5 * value2, 10E-6) : value1 == 1 ? Math.min(1.2 * value2, 50) : value2));
        setParameterMatrix(wPrev, matrix, WPrev);

        matrix.subtractBy(matrixGradient.apply(UnaryFunctionType.SGN).multiply(WPrev));

        dWDir = dWDir.applyBi(matrixGradient, new BinaryFunction((value1, value2) -> value1 == -1 ? 0 : value2));
        setParameterMatrix(dPrev, matrix, dWDir);
    }

}
