/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements gradient clipping matrix operation.
 *
 */
public class GradientClippingMatrixOperation extends AbstractMatrixOperation {

    /**
     * Threshold.
     *
     */
    private final double threshold;

    /**
     * Constructor for gradient clipping matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     * @param threshold threshold.
     */
    public GradientClippingMatrixOperation(int rows, int columns, int depth, double threshold) {
        super(rows, columns, depth, true);
        this.threshold = threshold;
    }

    /**
     * Applies matrix operation.
     *
     * @param first first matrix.
     * @param inplace if true operation is applied in place otherwise result is returned as new matrix.
     * @return result of operation.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix first, boolean inplace) throws MatrixException {
        Matrix result = first;
        double l2Norm = Math.sqrt(first.norm(2));
        if (l2Norm > threshold) {
            if (inplace) first.multiplyBy(threshold / l2Norm);
            else result = first.multiply(threshold / l2Norm);
        }
        return result;
    }

    /**
     * Applies operation.<br>
     * Ignores masking of other matrix.<br>
     *
     * @param row    current row.
     * @param column current column.
     * @param depth  current depth.
     * @param value  current value.
     * @param result result matrix.
     */
    public void apply(int row, int column, int depth, double value, Matrix result) {
    }

}
