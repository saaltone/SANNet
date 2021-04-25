package utils.matrix.operation;

import utils.matrix.Matrix;

/**
 * Defines abstract convolution matrix operation.
 *
 */
public abstract class AbstractConvolutionMatrixOperation extends AbstractMatrixOperation {

    /**
     * Input matrix.
     *
     */
    protected Matrix input;

    /**
     * Filter matrix.
     *
     */
    protected Matrix filter;

    /**
     * Result matrix.
     *
     */
    protected Matrix result;

    /**
     * Matrix dilation value.
     *
     */
    protected final int dilation;

    /**
     * Filter row size.
     *
     */
    protected final int filterRowSize;

    /**
     * Filter column size.
     *
     */
    protected final int filterColumnSize;

    /**
     * Constructor for abstract convolution matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param filterRowSize filter row size
     * @param filterColumnSize filter column size.
     * @param dilation dilation step
     */
    public AbstractConvolutionMatrixOperation(int rows, int columns, int filterRowSize, int filterColumnSize, int dilation) {
        super(rows, columns, false);
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
     * Sets filter matrix.
     *
     * @param filter filter matrix.
     */
    public void setFilter(Matrix filter) {
        this.filter = filter;
    }

    /**
     * Returns filter matrix.
     *
     * @return filter matrix.
     */
    public Matrix getFilter() {
        return filter;
    }

    /**
     * Sets result matrix.
     *
     * @param result result matrix.
     */
    public void setResult(Matrix result) {
        this.result = result;
    }

    /**
     * Returns result matrix.
     *
     * @return result matrix.
     */
    public Matrix getResult() {
        return result;
    }

}
