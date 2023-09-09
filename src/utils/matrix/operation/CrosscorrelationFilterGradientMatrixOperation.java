/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

/**
 * Implements crosscorrelation filter gradient matrix operation.
 *
 */
public class CrosscorrelationFilterGradientMatrixOperation extends AbstractConvolutionFilterGradientMatrixOperation {

    /**
     * Constructor for crosscorrelation filter gradient matrix operation.
     *
     * @param rows             number of rows for operation.
     * @param columns          number of columns for operation.
     * @param depth            depth for operation.
     * @param inputDepth       input depth.
     * @param filterRowSize    filter row size
     * @param filterColumnSize filter column size.
     * @param dilation         dilation step
     * @param stride           stride step
     * @param isDepthSeparable if true convolution is depth separable
     */
    public CrosscorrelationFilterGradientMatrixOperation(int rows, int columns, int depth, int inputDepth, int filterRowSize, int filterColumnSize, int dilation, int stride, boolean isDepthSeparable) {
        super(rows, columns, depth, inputDepth, filterRowSize, filterColumnSize, dilation, stride, isDepthSeparable, false);
    }

}
