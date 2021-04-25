package utils.matrix.operation;

/**
 * Defines convolution input gradient matrix operation.
 *
 */
public class ConvolutionInputGradientMatrixOperation extends AbstractConvolutionInputGradientMatrixOperation {

    /**
     * Constructor for convolution input gradient matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param filterRowSize filter row size
     * @param filterColumnSize filter column size.
     * @param dilation dilation step
     */
    public ConvolutionInputGradientMatrixOperation(int rows, int columns, int filterRowSize, int filterColumnSize, int dilation) {
        super(rows, columns, filterRowSize, filterColumnSize, dilation);
    }

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param value current value.
     */
    public void apply(int row, int column, double value) {
        for (int filterRow = 0; filterRow < filterRowSize; filterRow += dilation) {
            for (int filterColumn = 0; filterColumn < filterColumnSize; filterColumn += dilation) {
                inputGradient.incrementByValue(row + filterRow, column + filterColumn, filter.getValue(filterRowSize - 1 - filterRow, filterColumnSize - 1 - filterColumn) * value);
            }
        }
    }

}
