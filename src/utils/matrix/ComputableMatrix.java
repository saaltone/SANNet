package utils.matrix;

import utils.matrix.operation.*;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Random;

/**
 * Class that defines computable operations for matrices.<br>
 *
 */
public abstract class ComputableMatrix extends AbstractMatrix {

    /**
     * Initializer variable.
     *
     */
    private Matrix.Initializer initializer;

    /**
     * If true matrix is treated as scalar (1x1) matrix otherwise as normal matrix.
     *
     */
    private final boolean isScalar;

    /**
     * Stride size for convolutional and pooling operations.
     *
     */
    private int stride = 1;

    /**
     * Dilation step size for convolutional operations.
     *
     */
    private int dilation = 1;

    /**
     * Filter row size for convolutional and pooling operations.
     *
     */
    private int filterRowSize;

    /**
     * Filter column size for convolutional and pooling operations.
     *
     */
    private int filterColumnSize;

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
        this.isScalar = isScalar;
    }

    /**
     * Constructor for matrix.
     *
     * @param isScalar true if matrix is scalar (size 1x1).
     * @param name name if matrix.
     */
    protected ComputableMatrix(boolean isScalar, String name) {
        super(name);
        this.isScalar = isScalar;
    }

    /**
     * Returns value from uniform distribution within -range to +range.
     *
     * @param range range of the distribution.
     * @return random value drawn from the distribution.
     */
    private double uniform(double range) {
        return (2 * random.nextDouble()- 1)  * range;
    }

    /**
     * Returns value from normal distribution defined by standard deviation.
     *
     * @param standardDeviation standard deviation of normal distribution.
     * @return random value drawn from the distribution.
     */
    private double normal(double standardDeviation) {
        return random.nextGaussian() * standardDeviation;
    }

    /**
     * Sets initializer of matrix.
     *
     * @param initializer initializer.
     */
    public void setInitializer(Matrix.Initializer initializer) {
        this.initializer = initializer;
    }

    /**
     * Returns initializer of matrix.
     *
     * @return initializer.
     */
    public Matrix.Initializer getInitializer() {
        return initializer;
    }

    /**
     * Initializes matrix.
     *
     * @param initialization type of initialization defined in class Init.
     */
    public void initialize(Initialization initialization) {
        initialize(initialization, 0, 0);
    }

    /**
     * Initializes matrix.
     *
     * @param initialization type of initialization defined in class Init.
     * @param inputs applied in convolutional initialization defined as channels * filter size * filter size.
     * @param outputs applied in convolutional initialization defined as filters * filter size * filter size.
     */
    public void initialize(Initialization initialization, int inputs, int outputs) {
        switch (initialization) {
            case ZERO:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> 0;
                break;
            case ONE:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> 1;
                initialize(initializer);
                break;
            case RANDOM:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> random.nextDouble();
                initialize(initializer);
                break;
            case IDENTITY:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> (row == col) ? 1 : 0;
                initialize(initializer);
                break;
            case NORMAL_XAVIER:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> normal(Math.sqrt(2 / (double)(getRows() + getColumns())));
                initialize(initializer);
                break;
            case UNIFORM_XAVIER:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> uniform(Math.sqrt(6 / (double)(getRows() + getColumns())));
                initialize(initializer);
                break;
            case NORMAL_HE:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> normal(Math.sqrt(2 / ((double)getRows())));
                initialize(initializer);
                break;
            case UNIFORM_HE:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> uniform(Math.sqrt(6 / (double)(getRows())));
                initialize(initializer);
                break;
            case NORMAL_LECUN:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> normal(Math.sqrt(1 / (double)(getRows())));
                initialize(initializer);
                break;
            case UNIFORM_LECUN:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> uniform(Math.sqrt(3 / (double)(getRows())));
                initialize(initializer);
                break;
            case NORMAL_XAVIER_CONV:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> normal(Math.sqrt(2 / (double)(outputs + inputs)));
                initialize(initializer);
                break;
            case UNIFORM_XAVIER_CONV:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> uniform(Math.sqrt(6 / (double)(outputs + inputs)));
                initialize(initializer);
                break;
            case NORMAL_HE_CONV:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> normal(Math.sqrt(2 / (double)(outputs)));
                initialize(initializer);
                break;
            case UNIFORM_HE_CONV:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> uniform(Math.sqrt(6 / (double)(outputs)));
                initialize(initializer);
                break;
            case NORMAL_LECUN_CONV:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> normal(Math.sqrt(1 / (double)(outputs)));
                initialize(initializer);
                break;
            case UNIFORM_LECUN_CONV:
                initializer = (Matrix.Initializer & Serializable) (row, col) -> uniform(Math.sqrt(3 / (double)(outputs)));
                initialize(initializer);
                break;
            default:
                break;
        }
    }

    /**
     * Returns true if matrix is scalar otherwise false.
     *
     * @return true if matrix is scalar otherwise false.
     */
    public boolean isScalar() {
        return isScalar;
    }

    /**
     * Initializes matrix with given initializer operation.
     *
     * @param initializer initializer operation.
     */
    public void initialize(Matrix.Initializer initializer) {
        for (int row = 0; row < getRows(); row++) {
            for (int col = 0; col < getColumns(); col++) {
                setValue(row, col, initializer.value(row, col));
            }
        }
    }

    /**
     * Initializes matrix with given value.
     *
     * @param value initialization value.
     */
    public void initializeToValue(double value) {
        for (int row = 0; row < getRows(); row++) {
            for (int col = 0; col < getColumns(); col++) {
                setValue(row, col, value);
            }
        }
    }

    /**
     * Increment value of specific row and column.
     *
     * @param row row of value to be added.
     * @param column column of value to be added.
     * @param value to be added.
     */
    public void incrementByValue(int row, int column, double value) {
        setValue(row, column, getValue(row, column) + value);
    }

    /**
     * Decrease value of specific row and column.
     *
     * @param row row of value to be decreased.
     * @param column column of value to be decreased.
     * @param value to be decreased.
     */
    public void decrementByValue(int row, int column, double value) {
        setValue(row, column, getValue(row, column) - value);
    }

    /**
     * Multiply value of specific row and column.
     *
     * @param row row of value to be multiplied.
     * @param column column of value to be multiplied.
     * @param value to be multiplied.
     */
    public void multiplyByValue(int row, int column, double value) {
        setValue(row, column, getValue(row, column) * value);
    }

    /**
     * Divide value of specific row and column.
     *
     * @param row row of value to be divided.
     * @param column column of value to be divided.
     * @param value to be divided.
     */
    public void divideByValue(int row, int column, double value) {
        setValue(row, column, getValue(row, column) / value);
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

        EqualMatrixOperation equalMatrixOperation = new EqualMatrixOperation(getRows(), getColumns());
        equalMatrixOperation.setOther(other);
        // Ignores masking of other matrix.
        applyMatrixOperation(equalMatrixOperation);

    }

    /**
     * Applies single variable operation to this matrix and stores operation result into result matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param result matrix which stores operation result.
     * @param matrixUnaryOperation single variable operation defined as matrix unary operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and result matrix are not of equal dimensions.
     */
    public Matrix apply(Matrix result, Matrix.MatrixUnaryOperation matrixUnaryOperation) throws MatrixException {
        if (result.getRows() != getRows() || result.getColumns() != getColumns()) {
            throw new MatrixException("Incompatible result matrix sizes: " + result.getRows() + "x" + result.getColumns());
        }

        UnaryMatrixOperation unaryMatrixOperation = new UnaryMatrixOperation(getRows(), getColumns(), matrixUnaryOperation);
        unaryMatrixOperation.setResult(result);
        applyMatrixOperation(unaryMatrixOperation);

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
     * @param matrixBinaryOperation two variable operation defined as matrix binary operation.
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

        BinaryMatrixOperation binaryMatrixOperation = new BinaryMatrixOperation(rows, columns, matrixBinaryOperation);
        binaryMatrixOperation.setOther(other);
        binaryMatrixOperation.setResult(result);
        applyMatrixOperation(binaryMatrixOperation);

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

        DotMatrixOperation dotMatrixOperation = new DotMatrixOperation(getRows(), other.getColumns());
        dotMatrixOperation.setFirst(this);
        dotMatrixOperation.setSecond(other);
        dotMatrixOperation.setResult(result);
        applyMatrixOperation(dotMatrixOperation);

    }

    /**
     * Applies matrix operation.
     *
     * @param matrixOperation matrix operation.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void applyMatrixOperation(MatrixOperation matrixOperation) throws MatrixException {
        final int rows = matrixOperation.getRows();
        final int columns = matrixOperation.getColumns();
        final Matrix other = matrixOperation.getAnother();
        final boolean provideValue = matrixOperation.getProvideValue();
        final int rowStride = stride;
        final int columnStride = stride;
        if (!matrixOperation.hasMask(this, other)) {
            for (int row = 0; row < rows; row += rowStride) {
                for (int column = 0; column < columns; column += columnStride) {
                    matrixOperation.apply(row, column, provideValue ? getValue(row, column) : 0);
                }
            }
        }
        else {
            for (int row = 0; row < rows; row += rowStride) {
                for (int column = 0; column < columns; column += columnStride) {
                    if (!matrixOperation.hasMaskAt(row, column, this, other)) {
                        matrixOperation.apply(row, column, provideValue ? getValue(row, column) : 0);
                    }
                }
            }
        }
    }

    /**
     * Calculates sum of matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return sum of matrix.
     */
    public double sum() throws MatrixException {
        SumMatrixOperation sumMatrixOperation = new SumMatrixOperation(getRows(), getColumns());
        applyMatrixOperation(sumMatrixOperation);
        return sumMatrixOperation.getSum();
    }

    /**
     * Calculates mean of matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return mean of matrix.
     */
    public double mean() throws MatrixException {
        SumMatrixOperation sumMatrixOperation = new SumMatrixOperation(getRows(), getColumns());
        applyMatrixOperation(sumMatrixOperation);
        return sumMatrixOperation.getMean();
    }

    /**
     * Calculates variance of matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return variance of matrix.
     */
    public double variance(double mean) throws MatrixException {
        VarianceMatrixOperation varianceMatrixOperation = new VarianceMatrixOperation(getRows(), getColumns(), mean);
        applyMatrixOperation(varianceMatrixOperation);
        return varianceMatrixOperation.getVariance();
    }

    /**
     * Calculates standard deviation of matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @param mean mean value given as input.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return standard deviation of matrix.
     */
    public double standardDeviation(double mean) throws MatrixException {
        VarianceMatrixOperation varianceMatrixOperation = new VarianceMatrixOperation(getRows(), getColumns(), mean);
        applyMatrixOperation(varianceMatrixOperation);
        return varianceMatrixOperation.getStandardDeviation();
    }

    /**
     * Calculates cumulative p- norm (p is number equal or bigger than 1) of matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @param p p value for norm.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return norm of matrix.
     */
    public double norm(int p) throws MatrixException {
        NormMatrixOperation normMatrixOperation = new NormMatrixOperation(getRows(), getColumns(), p);
        applyMatrixOperation(normMatrixOperation);
        return normMatrixOperation.getNorm();
    }

    /**
     * Normalizes matrix by removing mean and variance.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @param inplace if true matrix is normalized in place otherwise copy of normalized matrix is returned.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return normalized matrix.
     */
    public Matrix normalize(boolean inplace) throws MatrixException {
        Matrix result = inplace ? this : getNewMatrix();
        NormalizeMatrixOperation normalizeMatrixOperation = new NormalizeMatrixOperation(getRows(), getColumns(), mean(), variance());
        normalizeMatrixOperation.setResult(result);
        applyMatrixOperation(normalizeMatrixOperation);
        return result;
    }

    /**
     * Returns minimum value of matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return minimum value of matrix.
     */
    public double min() throws MatrixException {
        MinMatrixOperation minMatrixOperation = new MinMatrixOperation(getRows(), getColumns());
        applyMatrixOperation(minMatrixOperation);
        return minMatrixOperation.getValue();
    }

    /**
     * Returns argmin meaning row and column of matrix containing minimum value.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return array containing row and column in this order that points to minimum value of matrix.
     */
    public int[] argmin() throws MatrixException {
        MinMatrixOperation minMatrixOperation = new MinMatrixOperation(getRows(), getColumns());
        applyMatrixOperation(minMatrixOperation);
        int[] result = new int[2];
        result[0] = minMatrixOperation.getRow();
        result[1] = minMatrixOperation.getColumn();
        return result;
    }

    /**
     * Returns maximum value of matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return maximum value of matrix.
     */
    public double max() throws MatrixException {
        MaxMatrixOperation maxMatrixOperation = new MaxMatrixOperation(getRows(), getColumns());
        applyMatrixOperation(maxMatrixOperation);
        return maxMatrixOperation.getValue();
    }

    /**
     * Returns argmax meaning row and column of matrix containing maximum value.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return array containing row and column in this order that points to maximum value of matrix.
     */
    public int[] argmax() throws MatrixException {
        MaxMatrixOperation maxMatrixOperation = new MaxMatrixOperation(getRows(), getColumns());
        applyMatrixOperation(maxMatrixOperation);
        int[] result = new int[2];
        result[0] = maxMatrixOperation.getRow();
        result[1] = maxMatrixOperation.getColumn();
        return result;
    }

    /**
     * Returns softmax of matrix.
     *
     * @param result result matrix.
     * @return softmax of matrix.
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
     * Returns Gumbel softmax of matrix.<br>
     * Applies sigmoid prior log function plus adds Gumbel noise.<br>
     *
     * @param result result matrix.
     * @param gumbelSoftmaxTau tau value for Gumbel Softmax.
     * @return Gumbel softmax of matrix.
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
     * Returns softmax gradient of matrix.<br>
     * Assumes that input matrix is softmax result.<br>
     *
     * @param result result matrix.
     * @return softmax gradient of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix softmaxGrad(Matrix result) throws MatrixException {
        if (getColumns() != 1) {
            throw new MatrixException("Matrix must be a column vector.");
        }
        if (getRows() != result.getRows() || getRows() != result.getColumns()) {
            throw new MatrixException("Incompatible result matrix size: " + result.getRows() + "x" + result.getColumns());
        }

        SoftmaxGradientMatrixOperation softmaxGradientMatrixOperation = new SoftmaxGradientMatrixOperation(getRows(), getRows());
        softmaxGradientMatrixOperation.setFirst(this);
        softmaxGradientMatrixOperation.setResult(result);
        applyMatrixOperation(softmaxGradientMatrixOperation);

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
     * Sets filter row size for convolution and pooling operations.
     *
     * @param filterRowSize filter row size.
     */
    public void setFilterRowSize(int filterRowSize) {
        this.filterRowSize = filterRowSize;
    }

    /**
     * Sets filter column size for convolution and pooling operations.
     *
     * @param filterColumnSize filter column size.
     */
    public void setFilterColumnSize(int filterColumnSize) {
        this.filterColumnSize = filterColumnSize;
    }

    /**
     * Returns filter row size for convolution and pooling operations.
     *
     * @return filter row size for convolution and pooling operations.
     */
    public int getFilterRowSize() {
        return filterRowSize;
    }

    /**
     * Returns filter column size for convolution and pooling operations.
     *
     * @return filter column size for convolution and pooling operations.
     */
    public int getFilterColumnSize() {
        return filterColumnSize;
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @param result calculated result of convolution.
     */
    protected void applyConvolve(Matrix filter, Matrix result) throws MatrixException {
        ConvolutionMatrixOperation convolutionMatrixOperation = new ConvolutionMatrixOperation(result.getRows(), result.getColumns(), filter.getRows(), filter.getColumns(), dilation);
        convolutionMatrixOperation.setInput(this);
        convolutionMatrixOperation.setFilter(filter);
        convolutionMatrixOperation.setResult(result);
        applyMatrixOperation(convolutionMatrixOperation);
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @param result calculated result of convolution.
     */
    protected void applyCrosscorrelate(Matrix filter, Matrix result) throws MatrixException {
        CrosscorrelationMatrixOperation crosscorrelationMatrixOperation = new CrosscorrelationMatrixOperation(result.getRows(), result.getColumns(), filter.getRows(), filter.getColumns(), dilation);
        crosscorrelationMatrixOperation.setInput(this);
        crosscorrelationMatrixOperation.setFilter(filter);
        crosscorrelationMatrixOperation.setResult(result);
        applyMatrixOperation(crosscorrelationMatrixOperation);
    }

    /**
     * Calculates Winograd convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @param result calculated result of convolution.
     */
    protected void applyWinogradConvolve(Matrix filter, Matrix result) throws MatrixException {
        WinogradConvolutionMatrixOperation winogradConvolutionMatrixOperation = new WinogradConvolutionMatrixOperation(result.getRows(), result.getColumns());
        winogradConvolutionMatrixOperation.setInput(this);
        winogradConvolutionMatrixOperation.setFilter(filter);
        winogradConvolutionMatrixOperation.setResult(result);
        applyMatrixOperation(winogradConvolutionMatrixOperation);
    }

    /**
     * Calculates Winograd convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @param result calculated result of convolution.
     * @param A A matrix
     * @param AT A transposed matrix
     * @param C C matrix
     * @param CT C transposed matrix
     * @param G G matrix
     * @param GT G transposed matrix
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void applyWinogradConvolve(Matrix filter, Matrix result, Matrix A, Matrix AT, Matrix C, Matrix CT, Matrix G, Matrix GT) throws MatrixException {
        WinogradConvolutionMatrixOperation winogradConvolutionMatrixOperation = new WinogradConvolutionMatrixOperation(result.getRows(), result.getColumns(), A, AT, C, CT, G, GT);
        winogradConvolutionMatrixOperation.setInput(this);
        winogradConvolutionMatrixOperation.setFilter(filter);
        winogradConvolutionMatrixOperation.setResult(result);
        applyMatrixOperation(winogradConvolutionMatrixOperation);
    }

    /**
     * Calculates Winograd convolution between this matrix and filter matrix.
     *
     * @param preprocessedFilter preprocessed filter matrix.
     * @param result calculated result of convolution.
     * @param A A matrix
     * @param AT A transposed matrix
     * @param C C matrix
     * @param CT C transposed matrix
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void applyWinogradConvolve(Matrix preprocessedFilter, Matrix result, Matrix A, Matrix AT, Matrix C, Matrix CT) throws MatrixException {
        WinogradConvolutionMatrixOperation winogradConvolutionMatrixOperation = new WinogradConvolutionMatrixOperation(result.getRows(), result.getColumns(), A, AT, C, CT);
        winogradConvolutionMatrixOperation.setInput(this);
        winogradConvolutionMatrixOperation.setFilter(preprocessedFilter);
        winogradConvolutionMatrixOperation.setResult(result);
        applyMatrixOperation(winogradConvolutionMatrixOperation);
    }

    /**
     * Calculates gradient of convolution for input.
     *
     * @param filter filter for convolution operator.
     * @throws MatrixException throws exception if matrix operation fails.
     * @param inputGradient input gradient.
     */
    public void convolveInputGradient(Matrix filter, Matrix inputGradient) throws MatrixException {
        ConvolutionInputGradientMatrixOperation convolutionInputGradientMatrixOperation = new ConvolutionInputGradientMatrixOperation(getRows(), getColumns(), filter.getRows(), filter.getColumns(), dilation);
        convolutionInputGradientMatrixOperation.setFilter(filter);
        convolutionInputGradientMatrixOperation.setInputGradient(inputGradient);
        applyMatrixOperation(convolutionInputGradientMatrixOperation);
    }

    /**
     * Calculates gradient of crosscorrelation for input.
     *
     * @param filter filter for crosscorrelation operator.
     * @throws MatrixException throws exception if matrix operation fails.
     * @param inputGradient input gradient.
     */
    public void crosscorrelateInputGradient(Matrix filter, Matrix inputGradient) throws MatrixException {
        CrosscorrelationInputGradientMatrixOperation crosscorrelationInputGradientMatrixOperation = new CrosscorrelationInputGradientMatrixOperation(getRows(), getColumns(), filter.getRows(), filter.getColumns(), dilation);
        crosscorrelationInputGradientMatrixOperation.setFilter(filter);
        crosscorrelationInputGradientMatrixOperation.setInputGradient(inputGradient);
        applyMatrixOperation(crosscorrelationInputGradientMatrixOperation);
    }

    /**
     * Calculates gradient of convolution for filter.
     *
     * @param input input for convolutional operator.
     * @throws MatrixException throws exception if matrix operation fails.
     * @param filterGradient filter gradient.
     */
    public void convolveFilterGradient(Matrix input, Matrix filterGradient) throws MatrixException {
        ConvolutionFilterGradientMatrixOperation convolutionFilterGradientMatrixOperation = new ConvolutionFilterGradientMatrixOperation(getRows(), getColumns(), filterGradient.getRows(), filterGradient.getColumns(), dilation);
        convolutionFilterGradientMatrixOperation.setInput(input);
        convolutionFilterGradientMatrixOperation.setFilterGradient(filterGradient);
        applyMatrixOperation(convolutionFilterGradientMatrixOperation);
    }

    /**
     * Calculates gradient of crosscorrelation for filter.
     *
     * @param input input for crosscorrelation operator.
     * @throws MatrixException throws exception if matrix operation fails.
     * @param filterGradient filter gradient.
     */
    public void crosscorrelateFilterGradient(Matrix input, Matrix filterGradient) throws MatrixException {
        CrosscorrelationFilterGradientMatrixOperation crosscorrelationFilterGradientMatrixOperation = new CrosscorrelationFilterGradientMatrixOperation(getRows(), getColumns(), filterGradient.getRows(), filterGradient.getColumns(), dilation);
        crosscorrelationFilterGradientMatrixOperation.setInput(input);
        crosscorrelationFilterGradientMatrixOperation.setFilterGradient(filterGradient);
        applyMatrixOperation(crosscorrelationFilterGradientMatrixOperation);
    }

    /**
     * Calculates max pooling operation for matrix and returns max arguments.
     *
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     * @param maxPos maximum position for each result row and column value.
     */
    protected void applyMaxPool(Matrix result, HashMap<Integer, Integer> maxPos) throws MatrixException {
        MaxPoolMatrixOperation maxPoolMatrixOperation = new MaxPoolMatrixOperation(result.getRows(), result.getColumns(), getColumns(), getFilterRowSize(), getFilterColumnSize());
        maxPoolMatrixOperation.setInput(this);
        maxPoolMatrixOperation.setResult(result);
        maxPoolMatrixOperation.setMaxPos(maxPos);
        applyMatrixOperation(maxPoolMatrixOperation);
    }

    /**
     * Calculates gradient for max pool operation.
     *
     * @param inputGradient input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     * @param maxPos maximum position for each result row and column value.
     */
    public void maxPoolGradient(Matrix inputGradient, HashMap<Integer, Integer> maxPos) throws MatrixException {
        MaxPoolGradientMatrixOperation maxPoolGradientMatrixOperation = new MaxPoolGradientMatrixOperation(getRows(), getColumns(), inputGradient.getColumns());
        maxPoolGradientMatrixOperation.setInputGradient(inputGradient);
        maxPoolGradientMatrixOperation.setMaxPos(maxPos);
        applyMatrixOperation(maxPoolGradientMatrixOperation);
    }

    /**
     * Calculates average pooling operation for matrix.
     *
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void applyAveragePool(Matrix result) throws MatrixException {
        AveragePoolMatrixOperation averagePoolMatrixOperation = new AveragePoolMatrixOperation(result.getRows(), result.getColumns(), getFilterRowSize(), getFilterColumnSize());
        averagePoolMatrixOperation.setInput(this);
        averagePoolMatrixOperation.setResult(result);
        applyMatrixOperation(averagePoolMatrixOperation);
    }

    /**
     * Calculates gradient of average pooling operation for matrix.
     *
     * @param inputGradient input gradient.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void averagePoolGradient(Matrix inputGradient) throws MatrixException {
        AveragePoolGradientMatrixOperation averagePoolGradientMatrixOperation = new AveragePoolGradientMatrixOperation(getRows(), getColumns(), getFilterRowSize(), getFilterColumnSize());
        averagePoolGradientMatrixOperation.setInputGradient(inputGradient);
        applyMatrixOperation(averagePoolGradientMatrixOperation);
    }

}
