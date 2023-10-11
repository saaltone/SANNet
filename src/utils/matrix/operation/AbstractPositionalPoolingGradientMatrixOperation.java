/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashMap;

/**
 * Implements abstract positional pooling gradient matrix operation.
 *
 */
public abstract class AbstractPositionalPoolingGradientMatrixOperation extends AbstractMatrixOperation {

    /**
     * Number of input gradient rows.
     *
     */
    private final int inputGradientRowSize;

    /**
     * Number of input gradient columns.
     *
     */
    private final int inputGradientColumnSize;

    /**
     * Input position for each resulting row and column.
     *
     */
    private transient HashMap<Integer, Integer> inputPos;

    /**
     * Constructor for abstract positional pooling gradient matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     * @param inputRowSize number of input rows.
     * @param inputColumnSize number of input columns.
     * @param stride stride step
     */
    public AbstractPositionalPoolingGradientMatrixOperation(int rows, int columns, int depth, int inputRowSize, int inputColumnSize, int stride) {
        super(rows, columns, depth, true, stride);
        this.inputGradientRowSize = inputRowSize;
        this.inputGradientColumnSize = inputColumnSize;
    }

    /**
     * Applies matrix operation.
     *
     * @param outputGradient output gradient.
     * @param inputPos input positions.
     * @return input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */

    public Matrix apply(Matrix outputGradient, HashMap<Integer, Integer> inputPos) throws MatrixException {
        this.inputPos = inputPos;
        return applyMatrixOperation(outputGradient, null, outputGradient.getNewMatrix(inputGradientRowSize, inputGradientColumnSize, getDepth()));
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
        final int position = 2 * (depth * getRows() * getColumns() + column * getRows() + row);
        result.addByValue(inputPos.get(position), inputPos.get(position + 1), depth, value);
    }

}
