package utils.matrix.operation;

/**
 * Defines crosscorrelation matrix operation.
 *
 */
public class CrosscorrelationMatrixOperation extends AbstractConvolutionMatrixOperation {

    /**
     * Constructor for crosscorrelation matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param filterRowSize filter row size
     * @param filterColumnSize filter column size.
     * @param dilation dilation step
     */
    public CrosscorrelationMatrixOperation(int rows, int columns, int filterRowSize, int filterColumnSize, int dilation) {
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
        double resultValue = 0;
        for (int filterRow = 0; filterRow < filterRowSize; filterRow += dilation) {
            for (int filterColumn = 0; filterColumn < filterColumnSize; filterColumn += dilation) {
                resultValue += input.getValue(row + filterRow, column + filterColumn) * filter.getValue(filterRow, filterColumn);
            }
        }
        result.setValue(row, column, resultValue);
    }

}
