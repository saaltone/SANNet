package utils.matrix.operation;

import utils.matrix.Matrix;

/**
 * Defines abstract convolution input gradient matrix operation.
 *
 */
public abstract class AbstractConvolutionInputGradientMatrixOperation extends AbstractMatrixOperation {

    /**
     * Filter matrix.
     *
     */
    protected Matrix filter;

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
     * Resulting input gradient.
     *
     */
    protected Matrix inputGradient;

    /**
     * Dilation.
     *
     */
    protected final int dilation;

    /**
     * Constructor for abstract convolution input gradient matrix operation.
     *
     * @param rows number of rows for operation.
     * @param columns number of columns for operation.
     * @param filterRowSize filter row size
     * @param filterColumnSize filter column size.
     * @param dilation dilation step
     */
    public AbstractConvolutionInputGradientMatrixOperation(int rows, int columns, int filterRowSize, int filterColumnSize, int dilation) {
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
     * Sets input gradient matrix.
     *
     * @param inputGradient input gradient matrix.
     */
    public void setInputGradient(Matrix inputGradient) {
        this.inputGradient = inputGradient;
    }

    /**
     * Returns input gradient matrix.
     *
     * @return input gradient matrix.
     */
    public Matrix getInputGradient() {
        return inputGradient;
    }

}
