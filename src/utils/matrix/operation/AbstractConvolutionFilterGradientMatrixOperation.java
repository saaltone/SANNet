package utils.matrix.operation;

import utils.matrix.Matrix;

/**
 * Defines abstract convolution filter gradient matrix operation.
 *
 */
public abstract class AbstractConvolutionFilterGradientMatrixOperation extends AbstractMatrixOperation {

    /**
     * Input matrix.
     *
     */
    protected Matrix input;

    /**
     * Number of rows in filter.
     *
     */
    protected final int filterRowSize;

    /**
     * Number of columns in filter.
     *
     */
    protected final int filterColumnSize;

    /**
     * Resulting filter gradient.
     *
     */
    protected Matrix filterGradient;

    /**
     * Dilation.
     *
     */
    protected final int dilation;

    /**
     * Constructor for abstract convolution filter gradient matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param filterRowSize filter row size
     * @param filterColumnSize filter column size.
     * @param dilation dilation step
     */
    public AbstractConvolutionFilterGradientMatrixOperation(int rows, int columns, int filterRowSize, int filterColumnSize, int dilation) {
        super(rows, columns, true);
        this.filterRowSize = filterRowSize;
        this.filterColumnSize = filterColumnSize;
        this.dilation = dilation;
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
     * Sets input matrix.
     *
     * @param input input matrix.
     */
    public void setInput(Matrix input) {
        this.input = input;
    }

    /**
     * Returns input matrix.
     *
     * @return input matrix.
     */
    public Matrix getInput() {
        return input;
    }

    /**
     * Sets filter gradient matrix.
     *
     * @param filterGradient filter gradient matrix.
     */
    public void setFilterGradient(Matrix filterGradient) {
        this.filterGradient = filterGradient;
    }

    /**
     * Returns filter gradient matrix.
     *
     * @return filter gradient matrix.
     */
    public Matrix getFilterGradient() {
        return filterGradient;
    }

}
