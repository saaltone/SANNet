/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Defines matrix multi-label classification operation.
 *
 */
public class ClassifyMatrixOperation extends AbstractMatrixOperation {

    /**
     * First matrix.
     *
     */
    private Matrix first;

    /**
     * Result matrix.
     *
     */
    private Matrix result;

    /**
     * Defines threshold value for multi label classification. If value of label is below threshold it is classified as negative (0) otherwise classified as positive (1).
     *
     */
    private double multiLabelThreshold = 0.5;

    /**
     * Constructor for matrix unary operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     */
    public ClassifyMatrixOperation(int rows, int columns) {
        super(rows, columns, true);
    }

    /**
     * Constructor for matrix unary operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param multiLabelThreshold if class probability is below threshold is it classified as negative (0) otherwise as positive (1).
     */
    public ClassifyMatrixOperation(int rows, int columns, double multiLabelThreshold) {
        this(rows, columns);
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
        this.result = result;
        applyMatrixOperation();
        return result;
    }

    /**
     * Returns target matrix.
     *
     * @return target matrix.
     */
    protected Matrix getTargetMatrix() {
        return first;
    }

    /**
     * Returns another matrix used in operation.
     *
     * @return another matrix used in operation.
     */
    public Matrix getAnother() {
        return null;
    }

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param value current value.
     */
    public void apply(int row, int column, double value) {
        result.setValue(row, column, first.getValue(row, column) < multiLabelThreshold ? 0 : 1);
    }

}