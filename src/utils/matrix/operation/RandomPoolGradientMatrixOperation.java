/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

/**
 * Implements random pooling gradient matrix operation.<br>
 * Selects each input of pool for propagation randomly with uniform probability.<br>
 *
 */
public class RandomPoolGradientMatrixOperation extends AbstractPositionalPoolingGradientMatrixOperation {

    /**
     * Constructor for random pooling gradient matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param depth depth for operation.
     * @param inputRowSize number of input rows.
     * @param inputColumnSize number of input columns.
     * @param stride stride step
     */
    public RandomPoolGradientMatrixOperation(int rows, int columns, int depth, int inputRowSize, int inputColumnSize, int stride) {
        super(rows, columns, depth, inputRowSize, inputColumnSize, stride);
    }

}
