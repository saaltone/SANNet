/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package utils.matrix.operation;

/**
 * Implements cyclic pooling gradient matrix operation.<br>
 * Traverses cyclically each filter row and column through step by step and propagates selected row and column.<br>
 *
 */
public class CyclicPoolGradientMatrixOperation extends AbstractPositionalPoolingGradientMatrixOperation {

    /**
     * Constructor for cyclic pooling gradient matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     * @param inputRowSize number of input rows.
     * @param inputColumnSize number of input columns.
     * @param stride stride step
     */
    public CyclicPoolGradientMatrixOperation(int rows, int columns, int depth, int inputRowSize, int inputColumnSize, int stride) {
        super(rows, columns, depth, inputRowSize, inputColumnSize, stride);
    }

}
