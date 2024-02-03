/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.*;

import java.util.TreeMap;

/**
 * Implements matrix join operation.
 *
 */
public class JoinMatrixOperation extends AbstractMatrixOperation {

    /**
     * If true joins matrices vertically otherwise horizontally.
     *
     */
    private final boolean joinedVertically;

    /**
     * Constructor for join binary operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     * @param joinedVertically if true joined vertically otherwise horizontally
     */
    public JoinMatrixOperation(int rows, int columns, int depth, boolean joinedVertically) {
        super(rows, columns, depth, true);
        this.joinedVertically = joinedVertically;
    }

    /**
     * Applies matrix operation.
     *
     * @param first first matrix.
     * @param second second matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix first, Matrix second) throws MatrixException {
        return new JMatrix(new TreeMap<>() {{ put(0, first); put(1, second); }}, joinedVertically);
    }

    /**
     * Calculates gradient.
     *
     * @param outputGradient output gradient.
     * @param firstMatrix if true gradient is returned for first matrix.
     * @return input gradient
     */
    public Matrix applyGradient(Matrix outputGradient, boolean firstMatrix) {
        return firstMatrix ? outputGradient.getSubMatrices().get(0) : outputGradient.getSubMatrices().get(1);
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
    }

}
