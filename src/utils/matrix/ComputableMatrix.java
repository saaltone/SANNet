package utils.matrix;

import java.io.Serializable;
import java.util.Random;

/**
 * Class that defines computable operations for matrices.
 *
 */
public abstract class ComputableMatrix extends AbstractMatrix {

    /**
     * Defines interface for matrix operation.
     *
     */
    private interface MatrixOperation {

        /**
         * Method to apply matrix operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        void apply(int row, int column, double value);
    }

    /**
     * Defines equal matrix operation.
     *
     */
    private static class EqualMatrixOperation implements MatrixOperation {

        /**
         * Other matrix.
         *
         */
        final Matrix other;

        /**
         * Constructor for equal operation.
         *
         * @param other other matrix.
         */
        EqualMatrixOperation(Matrix other) {
            this.other = other;
        }

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            other.setValue(row, column, value);
        }
    }

    /**
     * Defines matrix unary operation.
     *
     */
    private static class UnaryMatrixOperation implements MatrixOperation {

        /**
         * Result matrix.
         *
         */
        final Matrix result;

        /**
         * Matrix unary operation.
         *
         */
        final Matrix.MatrixUnaryOperation matrixUnaryOperation;

        /**
         * Constructor for matrix unary operation.
         *
         * @param result result matrix.
         * @param matrixUnaryOperation matrix unary operation.
         */
        UnaryMatrixOperation(Matrix result, Matrix.MatrixUnaryOperation matrixUnaryOperation) {
            this.result = result;
            this.matrixUnaryOperation = matrixUnaryOperation;
        }

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            result.setValue(row, column, matrixUnaryOperation.execute(value));
        }

    }

    /**
     * Defines matrix binary operation.
     *
     */
    private static class BinaryMatrixOperation implements MatrixOperation {

        /**
         * Other matrix.
         *
         */
        final Matrix other;

        /**
         * Result matrix.
         *
         */
        final Matrix result;

        /**
         * Matrix binary operation.
         *
         */
        final Matrix.MatrixBinaryOperation matrixBinaryOperation;

        /**
         * Constructor for matrix binary operation.
         *
         * @param other other matrix.
         * @param result result matrix.
         * @param matrixBinaryOperation matrix binary operation.
         */
        BinaryMatrixOperation(Matrix other, Matrix result, Matrix.MatrixBinaryOperation matrixBinaryOperation) {
            this.other = other;
            this.result = result;
            this.matrixBinaryOperation = matrixBinaryOperation;
        }

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            result.setValue(row, column, matrixBinaryOperation.execute(value, other.getValue(row, column)));
        }

    }

    private static class DotMatrixOperation implements MatrixOperation {

        /**
         * First matrix.
         *
         */
        final Matrix first;

        /**
         * Second matrix.
         *
         */
        final Matrix second;

        /**
         * Result matrix.
         *
         */
        final Matrix result;

        /**
         * Constructor for dot matrix operation.
         *
         * @param first first (this) matrix.
         * @param second second (other) matrix.
         * @param result result matrix.
         */
        public DotMatrixOperation(Matrix first, Matrix second, Matrix result) {
            this.first = first;
            this.second = second;
            this.result = result;
        }

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            for (int x = 0; x < first.getColumns(); x++) {
                result.setValue(row, column, result.getValue(row, column) + first.getValue(row, x) * second.getValue(x, column));
            }
        }

    }

    /**
     * Defines sum matrix operation.
     *
     */
    private static class SumMatrixOperation implements MatrixOperation {

        /**
         * Cumulated value.
         *
         */
        double value;

        /**
         * Number of counted entries.
         *
         */
        int count;

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            this.value += value;
            count++;
        }
    }

    /**
     * Defines variance matrix operation.
     *
     */
    private static class VarianceMatrixOperation implements MatrixOperation {

        /**
         * Mean value for variance operation.
         *
         */
        final double mean;

        /**
         * Cumulated variance value.
         *
         */
        double value;

        /**
         * Number of counted entries.
         *
         */
        int count;

        /**
         * Constructor for variance operation.
         *
         * @param mean mean value for variance operation.
         */
        VarianceMatrixOperation(double mean) {
            this.mean = mean;
        }

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            this.value += Math.pow(value - mean, 2);
            count++;
        }

    }

    /**
     * Defines norm matrix operation.
     *
     */
    private static class NormMatrixOperation implements MatrixOperation {

        /**
         * Power for norm operation.
         *
         */
        final int p;

        /**
         * Cumulated norm value.
         *
         */
        double value;

        /**
         * Constructor for norm matrix operation.
         *
         * @param p power for norm operation.
         */
        NormMatrixOperation(int p) {
            this.p = p;
        }

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            this.value += Math.pow(Math.abs(value), p);
        }
    }

    /**
     * Defines normalize matrix operation.
     *
     */
    private static class NormalizeMatrixOperation implements MatrixOperation {

        /**
         * Mean for normalize operation.
         *
         */
        final double mean;

        /**
         * Variance for normalize operation.
         *
         */
        final double variance;

        /**
         * Cumulated norm value.
         *
         */
        final Matrix result;

        /**
         * Constructor for normalize matrix operation.
         *
         * @param mean mean for normalize operation.
         * @param variance variance for normalize operation.
         */
        NormalizeMatrixOperation(double mean, double variance, Matrix result) {
            this.mean = mean;
            this.variance = variance;
            this.result = result;
        }

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            result.setValue(row, column, (value - mean) / variance);
        }
    }

    /**
     * Defines minimum matrix operation.
     *
     */
    private static class MinMatrixOperation implements MatrixOperation {

        /**
         * Minimum value.
         *
         */
        double value = Double.POSITIVE_INFINITY;

        /**
         * Minimum row.
         *
         */
        int row = -1;

        /**
         * Minimum column.
         *
         */
        int column = -1;

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            if (value < this.value) {
                this.value = value;
                this.row = row;
                this.column = column;
            }
        }
    }

    /**
     * Defines maximum matrix operation.
     *
     */
    private static class MaxMatrixOperation implements MatrixOperation {

        /**
         * Maximum value.
         *
         */
        double value = Double.NEGATIVE_INFINITY;

        /**
         * Maximum row.
         *
         */
        int row = -1;

        /**
         * Maximum column.
         *
         */
        int column = -1;

        /**
         * Applies operation.
         *
         * @param row current row.
         * @param column current column.
         * @param value current value.
         */
        public void apply(int row, int column, double value) {
            if (value > this.value) {
                this.value = value;
                this.row = row;
                this.column = column;
            }
        }
    }

    /**
     * Stride size for convolutional and pooling operations.
     *
     */
    private int stride;

    /**
     * Dilation step size for convolutional operations.
     *
     */
    private int dilation;

    /**
     * Filter size for convolutional operations.
     *
     */
    private int filterSize;

    /**
     * Pool size for pooling operations.
     *
     */
    private int poolSize;

    /**
     * Random function for matrix class.
     *
     */
    private final Random random = new Random();

    /**
     * Constructor for matrix.
     *
     * @param isScalar true if matrix is scalar (size 1x1).
     */
    protected ComputableMatrix(boolean isScalar) {
        super(isScalar);
    }

    /**
     * Constructor for matrix.
     *
     * @param isScalar true if matrix is scalar (size 1x1).
     * @param name name if matrix.
     */
    protected ComputableMatrix(boolean isScalar, String name) {
        super(isScalar, name);
    }

    /**
     * Checks if data of other matrix is equal to data of this matrix
     *
     * @param other matrix to be compared.
     * @return true is data of this and other matrix are equal otherwise false.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public boolean equals(Matrix other) throws MatrixException {
        if (other.getRows() != getRows() || other.getColumns() != getColumns()) {
            throw new MatrixException("Incompatible target matrix size: " + other.getRows() + "x" + other.getColumns());
        }

        for (int row = 0; row < other.getRows(); row++) {
            for (int column = 0; column < other.getColumns(); column++) {
                if (getValue(row, column) != other.getValue(row, column)) return false;
            }
        }
        return true;
    }

    /**
     * Makes current matrix data equal to other matrix data.
     *
     * @param other other matrix to be copied as data of this matrix.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void setEqualTo(Matrix other) throws MatrixException {
        if (other.getRows() != getRows() || other.getColumns() != getColumns()) {
            throw new MatrixException("Incompatible target matrix size: " + other.getRows() + "x" + other.getColumns());
        }

        EqualMatrixOperation equalMatrixOperation = new EqualMatrixOperation(other);
        // Ignores masking of other matrix.
        applyMatrixOperation(equalMatrixOperation, null, getRows(), getColumns(), true);

    }

    /**
     * Applies single variable operation to this matrix and stores operation result into result matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param result matrix which stores operation result.
     * @param matrixUnaryOperation single variable operation defined as lambda operator.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and result matrix are not of equal dimensions.
     */
    public Matrix apply(Matrix result, Matrix.MatrixUnaryOperation matrixUnaryOperation) throws MatrixException {
        if (result.getRows() != getRows() || result.getColumns() != getColumns()) {
            throw new MatrixException("Incompatible result matrix sizes: " + result.getRows() + "x" + result.getColumns());
        }

        UnaryMatrixOperation unaryMatrixOperation = new UnaryMatrixOperation(result, matrixUnaryOperation);
        applyMatrixOperation(unaryMatrixOperation, null, getRows(), getColumns(), true);

        return result;
    }

    /**
     * Applies two variable operation to this matrix and other matrix and stores operation result into result matrix.<br>
     * Example of operation can be subtraction of other matrix from this matrix or
     * multiplying current matrix with other matrix.<br>
     * Applies masking element wise if either matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @param matrixBinaryOperation two variable operation defined as lambda operator.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this, other and result matrix are not of equal dimensions.
     */
    public Matrix applyBi(Matrix other, Matrix result, Matrix.MatrixBinaryOperation matrixBinaryOperation) throws MatrixException {
        if (!isScalar() && !other.isScalar() && (getRows() != other.getRows() || getColumns() != other.getColumns())) {
            throw new MatrixException("Incompatible matrix sizes: " + getRows() + "x" + getColumns() + " by " + other.getRows() + "x" + other.getColumns());
        }
        if (!isScalar() && !result.isScalar() && (getRows() != result.getRows() || getColumns() != result.getColumns())) {
            throw new MatrixException("Incompatible result matrix sizes: " + result.getRows() + "x" + result.getColumns());
        }

        // Checks if there is need to broadcast or un-broadcast due to scalar matrix.
        int rows = !isScalar() ? getRows() : other.getRows();
        int columns = !isScalar() ? getColumns() : other.getColumns();

        BinaryMatrixOperation binaryMatrixOperation = new BinaryMatrixOperation(other, result, matrixBinaryOperation);
        applyMatrixOperation(binaryMatrixOperation, other, rows, columns, true);

        return result;
    }

    /**
     * Takes matrix dot product of this and other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if columns of this matrix and rows of other matrix are not matching or rows of this and result matrix or columns of result and other matrix are not matching.
     */
    protected void applyDot(Matrix other, Matrix result) throws MatrixException {
        if (getColumns() != other.getRows()) {
            throw new MatrixException("Incompatible matrix sizes: " + getRows() + "x" + getColumns() + " by " + other.getRows() + "x" + other.getColumns());
        }
        if (getRows() != result.getRows() || other.getColumns() != result.getColumns()) {
            throw new MatrixException("Incompatible result matrix size: " + result.getRows() + "x" + result.getColumns());
        }

        DotMatrixOperation dotMatrixOperation = new DotMatrixOperation(this, other, result);
        applyMatrixOperation(dotMatrixOperation, other, getRows(), other.getColumns(), false);

    }

    /**
     * Applies matrix operation.
     *
     * @param matrixOperation matrix operation.
     */
    private void applyMatrixOperation(MatrixOperation matrixOperation) {
        applyMatrixOperation(matrixOperation, null, getRows(), getColumns(), true);
    }

    /**
     * Applies matrix operation.
     *  @param matrixOperation matrix operation.
     * @param rows number of matrix rows.
     * @param columns number of matrix columns.
     * @param provideValue if true value will be provided for matrix operation otherwise zero (no value) if provided.
     */
    private void applyMatrixOperation(MatrixOperation matrixOperation, Matrix other, int rows, int columns, final boolean provideValue) {
        if (!hasMask(other)) {
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    matrixOperation.apply(row, column, provideValue ? getValue(row, column) : 0);
                }
            }
        }
        else {
            for (int row = 0; row < rows; row++) {
                if (!hasRowMaskAt(row, other)) {
                    for (int column = 0; column < columns; column++) {
                        if (!hasMaskAt(row, column, other) && !hasColumnMaskAt(column, other)) {
                            matrixOperation.apply(row, column, provideValue ? getValue(row, column) : 0);
                        }
                    }
                }
            }
        }
    }

    /**
     * Check if matrix and optionally other matrix has mask.
     *
     * @param other other matrix.
     * @return returns true if this or other matrix has mask.
     */
    private boolean hasMask(Matrix other) {
        return getMask() != null || (other != null && other.getMask() != null);
    }

    /**
     * Check if matrix and optionally other matrix has mask at specific row and column.
     *
     * @param row row.
     * @param column column.
     * @param other other matrix.
     * @return returns true if this or other matrix has mask at specific row and column.
     */
    private boolean hasMaskAt(int row, int column, Matrix other) {
        return hasMaskAt(this, row, column) || (other != null && hasMaskAt(other, row, column));
    }

    /**
     * Check if matrix and optionally other matrix has mask at specific row.
     *
     * @param row row.
     * @param other other matrix.
     * @return returns true if this or other matrix has mask at specific row.
     */
    private boolean hasRowMaskAt(int row, Matrix other) {
        return hasRowMaskAt(this, row) || (other != null && hasRowMaskAt(other, row));
    }

    /**
     * Check if matrix and optionally other matrix has mask at specific column.
     *
     * @param column column.
     * @param other other matrix.
     * @return returns true if this or other matrix has mask at specific column.
     */
    private boolean hasColumnMaskAt(int column, Matrix other) {
        return hasColumnMaskAt(this, column) || (other != null && hasColumnMaskAt(other, column));
    }

    /**
     * Takes element wise cumulative sum of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return cumulative sum of this matrix.
     */
    public double sum() {
        SumMatrixOperation sumMatrixOperation = new SumMatrixOperation();
        applyMatrixOperation(sumMatrixOperation);
        return sumMatrixOperation.value;
    }

    /**
     * Takes mean of elements of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return mean of elements of this matrix.
     */
    public double mean() {
        SumMatrixOperation sumMatrixOperation = new SumMatrixOperation();
        applyMatrixOperation(sumMatrixOperation);
        return sumMatrixOperation.count > 0 ? sumMatrixOperation.value / (double) sumMatrixOperation.count : 0;
    }

    /**
     * Takes variance of elements of this matrix with mean value given as input parameter.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @return variance of elements of this matrix.
     */
    public double variance(double mean) {
        VarianceMatrixOperation varianceMatrixOperation = new VarianceMatrixOperation(mean);
        applyMatrixOperation(varianceMatrixOperation);
        return varianceMatrixOperation.count > 0 ? varianceMatrixOperation.value / (double) varianceMatrixOperation.count : 0;
    }

    /**
     * Takes standard deviation of elements of this matrix with mean value given as input parameter.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @return standard deviation of elements of this matrix.
     */
    public double standardDeviation(double mean) {
        VarianceMatrixOperation varianceMatrixOperation = new VarianceMatrixOperation(mean);
        applyMatrixOperation(varianceMatrixOperation);
        return varianceMatrixOperation.count > 1 ? Math.sqrt(varianceMatrixOperation.value / ((double) varianceMatrixOperation.count - 1)) : 0;
    }

    /**
     * Takes cumulative p- norm (p is number equal or bigger than 1) of this matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param p p value for norm.
     * @return cumulative norm value of matrix.
     */
    public double norm(int p) {
        NormMatrixOperation normMatrixOperation = new NormMatrixOperation(p);
        applyMatrixOperation(normMatrixOperation);
        return Math.pow(normMatrixOperation.value, 1 / (double)p);
    }

    /**
     * Normalizes matrix by removing mean and variance.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @param inplace if true matrix is normalized in place otherwise copy of normalized matrix is returned.
     * @return normalized matrix.
     */
    public Matrix normalize(boolean inplace) {
        Matrix result = inplace ? this : getNewMatrix();
        NormalizeMatrixOperation normalizeMatrixOperation = new NormalizeMatrixOperation(mean(), variance(), result);
        applyMatrixOperation(normalizeMatrixOperation);
        return result;
    }

    /**
     * Returns minimum value of matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return minimum value of matrix.
     */
    public double min() {
        MinMatrixOperation minMatrixOperation = new MinMatrixOperation();
        applyMatrixOperation(minMatrixOperation);
        return minMatrixOperation.value;
    }

    /**
     * Returns argmin meaning row and column of matrix containing minimum value.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return array containing row and column in this order that points to minimum value of matrix.
     */
    public int[] argmin() {
        MinMatrixOperation minMatrixOperation = new MinMatrixOperation();
        applyMatrixOperation(minMatrixOperation);
        int[] result = new int[2];
        result[0] = minMatrixOperation.row;
        result[1] = minMatrixOperation.column;
        return result;
    }

    /**
     * Returns maximum value of matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return maximum value of matrix.
     */
    public double max() {
        MaxMatrixOperation maxMatrixOperation = new MaxMatrixOperation();
        applyMatrixOperation(maxMatrixOperation);
        return maxMatrixOperation.value;
    }

    /**
     * Returns argmax meaning row and column of matrix containing maximum value.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @return array containing row and column in this order that points to maximum value of matrix.
     */
    public int[] argmax() {
        MaxMatrixOperation maxMatrixOperation = new MaxMatrixOperation();
        applyMatrixOperation(maxMatrixOperation);
        int[] result = new int[2];
        result[0] = maxMatrixOperation.row;
        result[1] = maxMatrixOperation.column;
        return result;
    }

    /**
     * Returns softmax of this matrix.
     *
     * @param result result matrix.
     * @return softmax of this matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix softmax(Matrix result) throws MatrixException {
        if (getColumns() != 1) {
            throw new MatrixException("Matrix must be a column vector.");
        }
        if (getRows() != result.getRows() || getColumns() != result.getColumns()) {
            throw new MatrixException("Incompatible result matrix size: " + result.getRows() + "x" + result.getColumns());
        }

        final double maxValue = max();
        apply(result, (Matrix.MatrixUnaryOperation & Serializable) (value) -> Math.exp(value - maxValue));
        result.divide(result.sum(), result);

        return result;
    }

    /**
     * Returns Gumbel softmax of this matrix.<br>
     * Applies sigmoid prior log function plus adds Gumbel noise.<br>
     *
     * @param result result matrix.
     * @param gumbelSoftmaxTau tau value for Gumbel Softmax.
     * @return softmax of this matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix gumbelSoftmax(Matrix result, double gumbelSoftmaxTau) throws MatrixException {
        if (getColumns() != 1) {
            throw new MatrixException("Matrix must be a column vector.");
        }
        if (getRows() != result.getRows() || getColumns() != result.getColumns()) {
            throw new MatrixException("Incompatible result matrix size: " + result.getRows() + "x" + result.getColumns());
        }

        apply(result, (Matrix.MatrixUnaryOperation & Serializable) (value) -> Math.exp((Math.log(Math.exp(value) / (1 + Math.exp(value))) + getGumbelNoise()) / gumbelSoftmaxTau));
        result.divide(result.sum(), result);

        return result;
    }

    /**
     * Returns Gumbel noise.<br>
     *
     * @return Gumbel noise.
     */
    private double getGumbelNoise() {
        double epsilon = 10E-8;
        return -Math.log(-Math.log(random.nextDouble() + epsilon) + epsilon);
    }

    /**
     * Returns softmax gradient of this matrix.<br>
     * Assumes that input matrix is softmax result.<br>
     *
     * @param result result matrix.
     * @return softmax gradient of this matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix softmaxGrad(Matrix result) throws MatrixException {
        if (getColumns() != 1) {
            throw new MatrixException("Matrix must be a column vector.");
        }
        if (getRows() != result.getRows() || getRows() != result.getColumns()) {
            throw new MatrixException("Incompatible result matrix size: " + result.getRows() + "x" + result.getColumns());
        }
        if (getMask() == null) {
            for (int row = 0; row < getRows(); row++) {
                for (int row1 = 0; row1 < getRows(); row1++) {
                    result.setValue(row1, row, (row == row1 ? 1 : 0) - getValue(row1, 0));
                }
            }
        }
        else {
            for (int row = 0; row < getRows(); row++) {
                if (!hasRowMaskAt(this, row) && !hasMaskAt(this, row, 0) && !hasColumnMaskAt(this, 0)) {
                    for (int row1 = 0; row1 < getRows(); row1++) {
                        result.setValue(row1, row, (row == row1 ? 1 : 0) - getValue(row1, 0));
                    }
                }
            }
        }
        return result;
    }

    /**
     * Sets stride size for convolution and pooling operations.
     *
     * @param stride stride size.
     */
    public void setStride(int stride) {
        this.stride = stride;
    }

    /**
     * Returns stride size for convolution and pooling operations.
     *
     * @return stride size.
     */
    public int getStride() {
        return stride;
    }

    /**
     * Sets dilation step size for convolution operations.
     *
     * @param dilation dilation step size.
     */
    public void setDilation(int dilation) {
        this.dilation = dilation;
    }

    /**
     * Returns dilation step size for convolution operations.
     *
     * @return dilation step size.
     */
    public int getDilation() {
        return dilation;
    }

    /**
     * Sets filter size for convolution operations.
     *
     * @param filterSize filter size.
     */
    public void setFilterSize(int filterSize) {
        this.filterSize = filterSize;
    }

    /**
     * Returns filter size.
     *
     * @return filter size
     */
    public int getFilterSize() {
        return filterSize;
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @param result calculated value of convolution.
     * @param asConvolution if true taken operation as convolution otherwise as crosscorrelation.
     */
    protected void applyConvolve(Matrix filter, Matrix result, boolean asConvolution) {
        for (int resultRow = 0; resultRow < result.getRows(); resultRow += stride) {
            for (int resultColumn = 0; resultColumn < result.getColumns(); resultColumn += stride) {
                convolve(filter, result, resultRow, resultColumn, asConvolution);
            }
        }
    }

    /**
     * Calculates convolution between this matrix and filter matrix for a specific slice row and column.
     *
     * @param filter filter
     * @param result result of operation.
     * @param inputRow row of input.
     * @param inputColumn column of input.
     * @param asConvolution if true taken operation as convolution otherwise as crosscorrelation.
     */
    private void convolve(Matrix filter, Matrix result, int inputRow, int inputColumn, boolean asConvolution) {
        double resultValue = 0;
        if (getMask() == null && filter.getMask() == null) {
            for (int filterRow = 0; filterRow < filterSize; filterRow++) {
                for (int filterColumn = 0; filterColumn < filterSize; filterColumn++) {
                    resultValue += getValue(inputRow + filterRow, inputColumn + filterColumn) * getSliceValue(filter, 0, 0, filterRow, filterColumn, filterSize, filterSize, asConvolution);
                }
            }
        }
        else {
            for (int filterRow = 0; filterRow < filterSize; filterRow++) {
                if (!hasRowMaskAt(this, filterRow) && !hasRowMaskAt(filter, filterRow)) {
                    for (int filterColumn = 0; filterColumn < filterSize; filterColumn++) {
                        if (!hasMaskAt(this, filterRow, filterColumn) && !hasColumnMaskAt(this, filterColumn) && !hasMaskAt(filter, filterRow, filterColumn) && !hasColumnMaskAt(filter, filterColumn)) {
                            resultValue += getValue(inputRow + filterRow, inputColumn + filterColumn) * getSliceValue(filter, 0, 0, filterRow, filterColumn, filterSize, filterSize, asConvolution);
                        }
                    }
                }
            }
        }
        result.setValue(inputRow, inputColumn, resultValue);
    }

    /**
     * Return value at specific slice position.
     *
     * @param slice reference to slice.
     * @param sliceStartRow start row at slice.
     * @param sliceStartColumn start column at slice.
     * @param sliceRow row at slice.
     * @param sliceColumn column at slice.
     * @param sliceRows number of slice rows.
     * @param sliceColumns number of slice columns.
     * @param asConvolution if true taken operation as convolution otherwise as crosscorrelation.
     * @return value of slice at specific position.
     */
    private double getSliceValue(Matrix slice, int sliceStartRow, int sliceStartColumn, int sliceRow, int sliceColumn, int sliceRows, int sliceColumns, boolean asConvolution) {
        int trueSliceRow = getSlicePosition(sliceRow, sliceRows, asConvolution);
        int trueSliceColumn = getSlicePosition(sliceColumn, sliceColumns, asConvolution);
        return (trueSliceRow % dilation == 0 && trueSliceColumn % dilation == 0) ? slice.getValue(sliceStartRow + trueSliceRow, sliceStartColumn + trueSliceColumn) : 0;
    }

    /**
     * Result position at slice.
     *
     * @param slicePosition position at slice
     * @param sliceSize size of slice.
     * @param asConvolution if true taken operation as convolution otherwise as crosscorrelation.
     * @return position.
     */
    private int getSlicePosition(int slicePosition, int sliceSize, boolean asConvolution) {
        return asConvolution ? sliceSize - 1 - slicePosition : slicePosition;
    }

    /**
     * Calculates gradient of convolution for output.
     *
     * @param filter filter for convolutional operator.
     * @param resultGradient result gradient.
     * @param asConvolution if true taken operation as convolution otherwise as crosscorrelation.
     */
    public void convolveOutputGradient(Matrix filter, Matrix resultGradient, boolean asConvolution) {
        for (int gradientRow = 0; gradientRow < getRows(); gradientRow += stride) {
            for (int gradientColumn = 0; gradientColumn < getColumns(); gradientColumn += stride) {
                convolveGradient(filter, 0, 0, resultGradient, gradientRow, gradientColumn, getValue(gradientRow, gradientColumn), filterSize, filterSize, asConvolution);
            }
        }
    }

    /**
     * Calculates gradient of convolution for filter.
     *
     * @param input input for convolutional operator.
     * @param resultGradient result gradient.
     * @param asConvolution if true taken operation as convolution otherwise as crosscorrelation.
     */
    public void convolveFilterGradient(Matrix input, Matrix resultGradient, boolean asConvolution) {
        for (int gradientRow = 0; gradientRow < getRows(); gradientRow += stride) {
            for (int gradientColumn = 0; gradientColumn < getColumns(); gradientColumn += stride) {
                convolveGradient(input, gradientRow, gradientColumn, resultGradient, 0, 0, getValue(gradientRow, gradientColumn), resultGradient.getRows(), resultGradient.getColumns(), asConvolution);
            }
        }
    }

    /**
     * Calculates gradient slice for convolution operation. This matrix is output gradient for calculation.
     *
     * @param input input.
     * @param inputRow input row.
     * @param inputColumn input column.
     * @param resultGradient resulting gradient matrix.
     * @param gradientRow gradient row.
     * @param gradientColumn gradient column.
     * @param gradientValue gradient value.
     * @param sliceRows size of slice (filter) in rows.
     * @param sliceColumns size of slice (filter) in columns.
     * @param asConvolution if true taken operation as convolution otherwise as crosscorrelation.
     */
    private void convolveGradient(Matrix input, int inputRow, int inputColumn, Matrix resultGradient, int gradientRow, int gradientColumn, double gradientValue, int sliceRows, int sliceColumns, boolean asConvolution) {
        if (getMask() == null && input.getMask() == null) {
            for (int sliceRow = 0; sliceRow < sliceRows; sliceRow++) {
                for (int sliceColumn = 0; sliceColumn < sliceColumns; sliceColumn++) {
                    resultGradient.incrementByValue(gradientRow + sliceRow, gradientColumn + sliceColumn, getSliceValue(input, inputRow, inputColumn, sliceRow, sliceColumn, sliceRows, sliceColumns, asConvolution) * gradientValue);
                }
            }
        }
        else {
            for (int sliceRow = 0; sliceRow < sliceRows; sliceRow++) {
                if (!hasRowMaskAt(this, sliceRow) && !hasRowMaskAt(input, sliceRows - 1 - sliceRow)) {
                    for (int sliceColumn = 0; sliceColumn < sliceColumns; sliceColumn++) {
                        if (!hasMaskAt(this, sliceRow, sliceColumn) && !hasColumnMaskAt(this, sliceColumn) && !hasMaskAt(input, sliceRows - 1 - sliceRow, sliceColumn) && !hasColumnMaskAt(input, sliceColumns - 1 - sliceColumn)) {
                            resultGradient.incrementByValue(gradientRow + sliceRow, gradientColumn + sliceColumn, getSliceValue(input, inputRow, inputColumn, sliceRow, sliceColumn, sliceRows, sliceColumns, asConvolution) * gradientValue);
                        }
                    }
                }
            }
        }
    }

    /**
     * Sets size of pool for pooling operation.
     *
     * @param poolSize pool size.
     */
    public void setPoolSize(int poolSize) {
        this.poolSize = poolSize;
    }

    /**
     * Returns pool size.
     *
     * @return pool size.
     */
    public int getPoolSize() {
        return poolSize;
    }

    /**
     * Calculates max pooling operation for this matrix and returns max arguments.
     *
     * @param result result matrix.
     * @param maxArgumentsAt arguments on maximum row and col value.
     */
    protected void applyMaxPool(Matrix result, int [][][] maxArgumentsAt) {
        for (int resultRow = 0; resultRow < result.getRows(); resultRow += stride) {
            for (int resultColumn = 0; resultColumn < result.getColumns(); resultColumn += stride) {
                maxPool(result, resultRow, resultColumn, maxArgumentsAt);
            }
        }
    }

    /**
     *
     * @param result result matrix.
     * @param resultRow result row.
     * @param resultColumn result column.
     * @param maxArgs arguments on maximum row and col value.
     */
    private void maxPool(Matrix result, int resultRow, int resultColumn, int [][][] maxArgs) {
        double maxValue = Double.NEGATIVE_INFINITY;
        if (getMask() == null) {
            for (int poolRow = 0; poolRow < poolSize; poolRow++) {
                for (int poolColumn = 0; poolColumn < poolSize; poolColumn++) {
                    int inputRow = resultRow + poolRow;
                    int inputColumn = resultColumn + poolColumn;
                    double inputValue = getValue(inputRow, inputColumn);
                    if (maxValue < inputValue) {
                        maxValue = inputValue;
                        maxArgs[resultRow][resultColumn][0] = inputRow;
                        maxArgs[resultRow][resultColumn][1] = inputColumn;
                    }
                }
            }
        }
        else {
            for (int poolRow = 0; poolRow < poolSize; poolRow++) {
                if (!hasRowMaskAt(this, poolRow)) {
                    for (int poolColumn = 0; poolColumn < poolSize; poolColumn++) {
                        if (!hasMaskAt(this, poolRow, poolColumn) && !hasColumnMaskAt(this, poolColumn)) {
                            int inputRow = resultRow + poolRow;
                            int inputColumn = resultColumn + poolColumn;
                            double inputValue = getValue(inputRow, inputColumn);
                            if (maxValue < inputValue) {
                                maxValue = inputValue;
                                maxArgs[resultRow][resultColumn][0] = inputRow;
                                maxArgs[resultRow][resultColumn][1] = inputColumn;
                            }
                        }
                    }
                }
            }
        }
        result.setValue(resultRow, resultColumn, maxValue);
    }

    /**
     * Calculates gradient for max pool operation.
     *
     * @param result result matrix.
     * @param maxArgs arguments on maximum row and col value.
     */
    public void maxPoolGradient(Matrix result, int[][][] maxArgs) {
        for (int gradientRow = 0; gradientRow < getRows(); gradientRow += stride) {
            for (int gradientColumn = 0; gradientColumn < getColumns(); gradientColumn += stride) {
                maxPoolGradient(result, gradientRow, gradientColumn, maxArgs[gradientRow][gradientColumn]);
            }
        }
    }

    /**
     * Calculates gradient for max pool operation at certain row and column position.
     *
     * @param result result matrix.
     * @param gradientRow gradient row position.
     * @param gradientColumn gradient column position.
     * @param maxArgs arguments on maximum row and col value.
     */
    private void maxPoolGradient(Matrix result, int gradientRow, int gradientColumn, int[] maxArgs) {
        if (getMask() == null) result.setValue(maxArgs[0], maxArgs[1], getValue(gradientRow, gradientColumn));
        else {
            if (!hasRowMaskAt(this, gradientRow) && !hasMaskAt(this, gradientRow, gradientColumn) && !hasColumnMaskAt(this, gradientColumn)) {
                result.setValue(maxArgs[0], maxArgs[1],  getValue(gradientRow, gradientColumn));
            }
        }
    }

    /**
     * Calculates average pooling operation for this matrix.
     *
     * @param result result matrix.
     */
    protected void applyAveragePool(Matrix result) {
        for (int resultRow = 0; resultRow < result.getRows(); resultRow += stride) {
            for (int resultColumn = 0; resultColumn < result.getColumns(); resultColumn += stride) {
                averagePool(result, resultRow, resultColumn);
            }
        }
    }

    /**
     * Calculates average pooling operation for this matrix.
     *
     * @param result result matrix.
     * @param resultRow result row
     * @param resultColumn result column
     */
    private void averagePool(Matrix result, int resultRow, int resultColumn) {
        double sumValue = 0;
        if (getMask() == null) {
            for (int row = 0; row < poolSize; row++) {
                for (int column = 0; column < poolSize; column++) {
                    sumValue += getValue(resultRow + row, resultColumn + column);
                }
            }
        }
        else {
            for (int row = 0; row < poolSize; row++) {
                if (!hasRowMaskAt(this, row)) {
                    for (int column = 0; column < poolSize; column++) {
                        if (!hasMaskAt(this, row, column) && !hasColumnMaskAt(this, column)) {
                            sumValue += getValue(resultRow + row, resultColumn + column);
                        }
                    }
                }
            }
        }
        result.setValue(resultRow, resultColumn, sumValue / (double)(poolSize * poolSize));
    }

    /**
     * Calculates gradient of average pooling operation for this matrix.
     *
     * @param result result matrix.
     */
    public void averagePoolGradient(Matrix result) {
        for (int resultRow = 0; resultRow < result.getRows(); resultRow += stride) {
            for (int resultColumn = 0; resultColumn < result.getColumns(); resultColumn += stride) {
                averagePoolGradient(result, resultRow, resultColumn);
            }
        }
    }

    /**
     * Calculates gradient of average pooling operation for this matrix.
     *
     * @param result result matrix.
     * @param resultRow result row
     * @param resultColumn result column
     */
    private void averagePoolGradient(Matrix result, int resultRow, int resultColumn) {
        if (getMask() == null) result.setValue(resultRow, resultColumn, 1 / (double)(poolSize * poolSize));
        else {
            if (!hasRowMaskAt(this, resultRow) && !hasMaskAt(this, resultRow, resultColumn) && !hasColumnMaskAt(this, resultColumn)) {
                result.setValue(resultRow, resultColumn, 1 / (double)(poolSize * poolSize));
            }
        }
    }

}
