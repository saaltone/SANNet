/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements matrix multi-label classification operation.
 *
 */
public class ClassifyMatrixOperation extends AbstractMatrixOperation {

    /**
     * First matrix.
     *
     */
    private Matrix first;

    /**
     * Implements threshold value for multi label classification. If value of label is below threshold it is classified as negative (0) otherwise classified as positive (1).
     *
     */
    private final double multiLabelThreshold;

    /**
     * Constructor for classify matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     */
    public ClassifyMatrixOperation(int rows, int columns, int depth) {
        super(rows, columns, depth, true);
        this.multiLabelThreshold = 0.5;
    }

    /**
     * Constructor for classify matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     * @param multiLabelThreshold if class probability is below threshold is it classified as negative (0) otherwise as positive (1).
     */
    public ClassifyMatrixOperation(int rows, int columns, int depth, double multiLabelThreshold) {
        super(rows, columns, depth, true);
        this.multiLabelThreshold = multiLabelThreshold;
    }

    /**
     * Applies operation.
     *
     * @param first first matrix.
     * @param result result matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix first, Matrix result) throws MatrixException {
        this.first = first;
        return applyMatrixOperation(first, null, result);
    }

    /**
     * Applies operation.
     *
     * @param row    current row.
     * @param column current column.
     * @param depth  current depth.
     * @param value  current value.
     * @param result result matrix.
     */
    public void apply(int row, int column, int depth, double value, Matrix result) {
        result.setValue(row, column, depth, first.getValue(row, column, depth) < multiLabelThreshold ? 0 : 1);
    }

}
