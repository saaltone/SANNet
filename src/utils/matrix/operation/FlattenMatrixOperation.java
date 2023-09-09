/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements flatten matrix operation.
 *
 */
public class FlattenMatrixOperation extends AbstractMatrixOperation {

    /**
     * Input matrix.
     *
     */
    protected transient Matrix input;

    /**
     * Result.
     *
     */
    protected transient Matrix result;


    /**
     * Constructor for flatten matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     */
    public FlattenMatrixOperation(int rows, int columns, int depth) {
        super(rows, columns, depth, true);
    }


    /**
     * Applies matrix operation.
     *
     * @param input input matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix input) throws MatrixException {
        this.input = input;
        this.result = input.redimension(getRows() * getColumns() * getDepth(), 1, 1, false);
        return result;
    }

    /**
     * Returns target matrix.
     *
     * @return target matrix.
     */
    protected Matrix getTargetMatrix() {
        return input;
    }

    /**
     * Returns another matrix used in operation.
     *
     * @return another matrix used in operation.
     */
    public Matrix getOther() {
        return null;
    }

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     * @param value current value.
     */
    public void apply(int row, int column, int depth, double value) {
    }

}
