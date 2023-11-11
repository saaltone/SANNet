/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix;

import utils.matrix.operation.*;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Random;

/**
 * Implements computable operations for matrices.<br>
 *
 */
public abstract class ComputableMatrix extends AbstractMatrix {

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
     * Filter depth for convolutional operation.
     *
     */
    private int filterDepth;

    /**
     * If true convolution is depth separable.
     *
     */
    private boolean isDepthSeparable;

    /**
     * Random function for matrix class.
     *
     */
    private final Random random = new Random();

    /**
     * Constructor for computable matrix.
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth defines depth of matrix.
     */
    protected ComputableMatrix(int rows, int columns, int depth) {
        super(rows, columns, depth);
        this.isScalar = (rows == 1 && columns == 1 && depth == 1);
    }

    /**
     * Constructor for computable matrix.<br>
     * If rows and columns do not equal to 1 isScalar parameter is ignored.<br>
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth defines depth of matrix.
     * @param isScalar true if matrix is scalar (size 1x1).
     */
    protected ComputableMatrix(int rows, int columns, int depth, boolean isScalar) {
        super(rows, columns, depth);
        this.isScalar = isScalar && (rows == 1 && columns == 1 && depth == 1);
    }

    /**
     * Constructor for computable matrix.
     * If rows and columns do not equal to 1 isScalar parameter is ignored.<br>
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth defines depth of matrix.
     * @param isScalar true if matrix is scalar (size 1x1).
     * @param isTransposed if true matrix is transposed and if false not transposed.
     */
    protected ComputableMatrix(int rows, int columns, int depth, boolean isScalar, boolean isTransposed) {
        super(rows, columns, depth, isTransposed);
        this.isScalar = isScalar && (rows == 1 && columns == 1 && depth == 1);
    }

    /**
     * Constructor for computable matrix.
     * If rows and columns do not equal to 1 isScalar parameter is ignored.<br>
     *
     * @param rows defines number of rows in matrix.
     * @param columns defines number of columns in matrix.
     * @param depth defines depth of matrix.
     * @param isScalar true if matrix is scalar (size 1x1).
     * @param isTransposed if true matrix is transposed and if false not transposed.
     * @param canBeSliced if true matrix can be slides otherwise cannot be sliced.
     */
    protected ComputableMatrix(int rows, int columns, int depth, boolean isScalar, boolean isTransposed, boolean canBeSliced) {
        super(rows, columns, depth, isTransposed, canBeSliced);
        this.isScalar = isScalar && (rows == 1 && columns == 1 && depth == 1);
    }

    /**
     * Sets parameters for matrix.
     *
     * @param matrix matrix.
     * @throws MatrixException throws exception if cloning of mask fails.
     */
    protected void setParameters(Matrix matrix) throws MatrixException {
        super.setParameters(matrix);
        matrix.setStride(stride);
        matrix.setDilation(dilation);
        matrix.setFilterRowSize(filterRowSize);
        matrix.setFilterColumnSize(filterColumnSize);
        matrix.setFilterDepth(filterDepth);
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
            case ZERO -> initialize((Initializer & Serializable) (row, column) -> 0);
            case ONE -> initialize((Initializer & Serializable) (row, column) -> 1);
            case RANDOM -> initialize((Initializer & Serializable) (row, column) -> random.nextDouble());
            case IDENTITY -> initialize((Initializer & Serializable) (row, column) -> (row == column) ? 1 : 0);
            case NORMAL_XAVIER -> initialize((Initializer & Serializable) (row, column) -> normal(Math.sqrt(2 / (double) (getRows() + getColumns()))));
            case UNIFORM_XAVIER -> initialize((Initializer & Serializable) (row, column) -> uniform(Math.sqrt(6 / (double) (getRows() + getColumns()))));
            case NORMAL_HE -> initialize((Initializer & Serializable) (row, column) -> normal(Math.sqrt(2 / ((double) getRows()))));
            case UNIFORM_HE -> initialize((Initializer & Serializable) (row, column) -> uniform(Math.sqrt(6 / (double) (getRows()))));
            case NORMAL_LECUN -> initialize((Initializer & Serializable) (row, column) -> normal(Math.sqrt(1 / (double) (getRows()))));
            case UNIFORM_LECUN -> initialize((Initializer & Serializable) (row, column) -> uniform(Math.sqrt(3 / (double) (getRows()))));
            case NORMAL_XAVIER_CONV -> initialize((Initializer & Serializable) (row, column) -> normal(Math.sqrt(2 / (double) (outputs + inputs))));
            case UNIFORM_XAVIER_CONV -> initialize((Initializer & Serializable) (row, column) -> uniform(Math.sqrt(6 / (double) (outputs + inputs))));
            case NORMAL_HE_CONV -> initialize((Initializer & Serializable) (row, column) -> normal(Math.sqrt(2 / (double) (outputs))));
            case UNIFORM_HE_CONV -> initialize((Initializer & Serializable) (row, column) -> uniform(Math.sqrt(6 / (double) (outputs))));
            case NORMAL_LECUN_CONV -> initialize((Initializer & Serializable) (row, column) -> normal(Math.sqrt(1 / (double) (outputs))));
            case UNIFORM_LECUN_CONV -> initialize((Initializer & Serializable) (row, column) -> uniform(Math.sqrt(3 / (double) (outputs))));
            default -> {
            }
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
        int rows = getRows();
        int columns = getColumns();
        int totalDepth = getDepth();
        for (int depth = 0; depth < totalDepth; depth++) {
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    setValue(row, column, depth, initializer.value(row, column));
                }
            }
        }
    }

    /**
     * Initializes matrix with given value.
     *
     * @param value initialization value.
     */
    public void initializeToValue(double value) {
        int rows = getRows();
        int columns = getColumns();
        int totalDepth = getDepth();
        for (int depth = 0; depth < totalDepth; depth++) {
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    setValue(row, column, depth, value);
                }
            }
        }
    }

    /**
     * Increment value of specific row, column and depth.
     *
     * @param row row of value to be added.
     * @param column column of value to be added.
     * @param depth depth of value to be added.
     * @param value to be added.
     */
    public void addByValue(int row, int column, int depth, double value) {
        setValue(row, column, depth, getValue(row, column, depth) + value);
    }

    /**
     * Decrease value of specific row, column and depth.
     *
     * @param row row of value to be decreased.
     * @param column column of value to be decreased.
     * @param depth depth of value to be decreaeed.
     * @param value to be decreased.
     */
    public void subtractByValue(int row, int column, int depth, double value) {
        setValue(row, column, depth, getValue(row, column, depth) - value);
    }

    /**
     * Multiply value of specific row, column and depth.
     *
     * @param row row of value to be multiplied.
     * @param column column of value to be multiplied.
     * @param depth depth of value to be multiplied.
     * @param value to be multiplied.
     */
    public void multiplyByValue(int row, int column, int depth, double value) {
        setValue(row, column, depth, getValue(row, column, depth) * value);
    }

    /**
     * Divide value of specific row, column and depth.
     *
     * @param row row of value to be divided.
     * @param column column of value to be divided.
     * @param depth depth of value to be divided.
     * @param value to be divided.
     */
    public void divideByValue(int row, int column, int depth, double value) {
        setValue(row, column, depth, getValue(row, column, depth) / value);
    }

    /**
     * Checks if data of other matrix is equal to data of this matrix
     *
     * @param other matrix to be compared.
     * @return true is data of this and other matrix are equal otherwise false.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public boolean equals(Matrix other) throws MatrixException {
        return new IsEqualMatrixOperation(getRows(), getColumns(), getDepth()).apply(this, other);
    }

    /**
     * Makes current matrix data equal to other matrix data.
     *
     * @param other other matrix to be copied as data of this matrix.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void setEqualTo(Matrix other) throws MatrixException {
        new EqualMatrixOperation(getRows(), getColumns(), getDepth()).apply(other, this);
    }

    /**
     * Applies unaryFunction to this matrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param unaryFunction unary function.
     * @param inplace if true operation is applied in place otherwise result is returned as new matrix.
     * @return result matrix.
     * @throws MatrixException not thrown in any situation.
     */
    protected Matrix applyFunction(UnaryFunction unaryFunction, boolean inplace) throws MatrixException {
        return new UnaryMatrixOperation(getRows(), getColumns(), getDepth(), unaryFunction).applyFunction(this, inplace);
    }

    /**
     * Applies two variable operation to this matrix and other matrix and stores operation result into result matrix.<br>
     * Example of operation can be subtraction of other matrix from this matrix or
     * multiplying current matrix with other matrix.<br>
     * Applies masking element wise if either matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @param binaryFunction binary function.
     * @param inplace if true operation is applied in place otherwise result is returned as new matrix.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this, other and result matrix are not of equal dimensions.
     */
    protected Matrix applyBiFunction(Matrix other, BinaryFunction binaryFunction, boolean inplace) throws MatrixException {
        if (!isScalar() && !other.isScalar() && (getRows() != other.getRows() || getColumns() != other.getColumns() || getDepth() != other.getDepth())) {
            throw new MatrixException("Incompatible matrix sizes: " + getRows() + "x" + getColumns() + "x" + getDepth() + " by " + other.getRows() + "x" + other.getColumns() + "x" + other.getDepth());
        }
        // Checks if there is need to broadcast or un-broadcast due to scalar matrix.
        int rows = !isScalar() ? getRows() : other.getRows();
        int columns = !isScalar() ? getColumns() : other.getColumns();
        return new BinaryMatrixOperation(rows, columns, getDepth(), binaryFunction).applyFunction(this, other, inplace);
    }

    /**
     * Takes matrix dot product of this and other matrix.<br>
     * Applies masking element wise if this or other matrix is masked.<br>
     *
     * @param other matrix which acts as second variable in the operation.
     * @return matrix which stores operation result.
     * @throws MatrixException throws MatrixException if columns of this matrix and rows of other matrix are not matching or rows of this and result matrix or columns of result and other matrix are not matching.
     */
    protected Matrix applyDot(Matrix other) throws MatrixException {
        return new DotMatrixOperation(getRows(), other.getRows(), other.getColumns(), getDepth()).apply(this, other);
    }

    /**
     * Calculates sum of matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return sum of matrix.
     */
    public double sum() throws MatrixException {
        return new SumMatrixOperation(getRows(), getColumns(), getDepth()).applySum(this);
    }

    /**
     * Calculates mean of matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return mean of matrix.
     */
    public double mean() throws MatrixException {
        return new SumMatrixOperation(getRows(), getColumns(), getDepth()).applyMean(this);
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
        return new VarianceMatrixOperation(getRows(), getColumns(), getDepth(), mean).applyVariance(this);
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
        return new VarianceMatrixOperation(getRows(), getColumns(), getDepth(), mean).applyStandardDeviation(this);
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
        return new NormMatrixOperation(getRows(), getColumns(), getDepth(), p).apply(this);
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
        return new NormalizeMatrixOperation(getRows(), getColumns(), getDepth(), mean(), variance()).apply(this, inplace ? this : getNewMatrix());
    }

    /**
     * Returns minimum value of matrix.<br>
     * Applies masking element wise if this matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return minimum value of matrix.
     */
    public double min() throws MatrixException {
        return new MinMatrixOperation(getRows(), getColumns(), getDepth()).applyMin(this);
    }

    /**
     * Returns argmin meaning row and column of matrix containing minimum value.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return array containing row and column in this order that points to minimum value of matrix.
     */
    public int[] argmin() throws MatrixException {
        return new MinMatrixOperation(getRows(), getColumns(), getDepth()).applyArgMin(this);
    }

    /**
     * Returns maximum value of matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return maximum value of matrix.
     */
    public double max() throws MatrixException {
        return new MaxMatrixOperation(getRows(), getColumns(), getDepth()).applyMax(this);
    }

    /**
     * Returns argmax meaning row and column of matrix containing maximum value.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return array containing row and column in this order that points to maximum value of matrix.
     */
    public int[] argmax() throws MatrixException {
        return new MaxMatrixOperation(getRows(), getColumns(), getDepth()).applyArgMax(this);
    }

    /**
     * Calculates entropy of matrix.<br>
     * Applies masking element wise if matrix is masked.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return sum of matrix.
     */
    public double entropy() throws MatrixException {
        return new EntropyMatrixOperation(getRows(), getColumns(), getDepth()).applyEntropy(this);
    }

    /**
     * Returns binomial distribution.
     * Reference: <a href="https://peterchng.com/blog/2020/10/23/building-binomial-and-multinomial-samplers-in-java/">...</a>
     *
     * @param probability probability.
     * @return number of successful trials.
     */
    public int getBinomial(double probability) {
        return getBinomial(1, probability);
    }

    /**
     * Returns binomial distribution.
     * Reference: <a href="https://peterchng.com/blog/2020/10/23/building-binomial-and-multinomial-samplers-in-java/">...</a>
     *
     * @param numberOfTrials number of trials.
     * @param probability probability.
     * @return number of successful trials.
     */
    public int getBinomial(int numberOfTrials, double probability) {
        if (numberOfTrials < 1 || probability < 0) return 0;
        if (probability > 1) return numberOfTrials;

        int numberOfSuccessfulTrials = 0;
        for (int trial = 0; trial < numberOfTrials; trial++) {
            if (random.nextDouble() < probability) numberOfSuccessfulTrials++;
        }
        return numberOfSuccessfulTrials;
    }

    /**
     * Returns multinomial distribution. Assumes single trial.
     *
     * @return multinomial distribution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix getMultinomial() throws MatrixException {
        return getMultinomial(1);
    }

    /**
     * Returns multinomial distribution.
     * Reference: <a href="https://peterchng.com/blog/2020/10/23/building-binomial-and-multinomial-samplers-in-java/">...</a>
     *
     * @param numberOfTrials number of trials.
     * @return multinomial distribution.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix getMultinomial(int numberOfTrials) throws MatrixException {
        if (numberOfTrials < 1) throw new MatrixException("Number of trials cannot be less than 1.");

        Matrix result = getNewMatrix();
        final int rows = getRows();
        final int columns = getColumns();
        final int totalDepth = getDepth();
        double sumLeft = 1;
        int trialsLeft = numberOfTrials;
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                for (int depth = 0; depth < totalDepth; depth++) {
                    double probability = getValue(row, column, depth);
                    double binomial = getBinomial(trialsLeft, probability / sumLeft);
                    result.setValue(row, column, depth, binomial);
                    sumLeft -= probability;
                    trialsLeft -= binomial;
                    if (sumLeft <= 0 || trialsLeft <= 0) break;
                }
            }
        }

        return result;
    }

    /**
     * Implements inverted drop out.<br>
     * Function selectively masks out certain percentage of node governed by parameter probability during training phase.<br>
     * During training phase it also compensates all remaining inputs by dividing by probability.<br>
     *
     * @param probability probability
     * @param inplace if true clipping in done in place otherwise not.
     * @return result of drop out.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix applyDropout(double probability, boolean inplace) throws MatrixException {
        return new DropoutMatrixOperation(getRows(), getColumns(), getDepth(), probability).apply(this, inplace);
    }

    /**
     * Implements matrix noising.
     *
     * @param noise noise
     * @param inplace if true clipping in done in place otherwise not.
     * @return result of drop out.
     */
    public Matrix noise(double noise, boolean inplace) throws MatrixException {
        return apply(new UnaryFunction(value -> value + noise * (1 - 2 * random.nextDouble())), inplace);
    }

    /**
     * Returns softmax of matrix.
     *
     * @param softmaxTau tau value for Softmax.
     * @return softmax of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix softmax(double softmaxTau) throws MatrixException {
        if (getColumns() != 1) throw new MatrixException("Matrix must be a column vector.");

        return finalizeSoftmax(softmaxTau == 1 ? this : divide(softmaxTau));
    }

    /**
     * Returns Gumbel softmax of matrix.<br>
     * Applies sigmoid prior log function plus adds Gumbel noise.<br>
     *
     * @param softmaxTau tau value for Softmax.
     * @return Gumbel softmax of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix gumbelSoftmax(double softmaxTau) throws MatrixException {
        if (getColumns() != 1) throw new MatrixException("Matrix must be a column vector.");

        // https://blog.evjang.com/2016/11/tutorial-categorical-variational.html
        return finalizeSoftmax(applyFunction(new UnaryFunction(value -> softmaxTau == 1 ? (value + getGumbelNoise()) : (value + getGumbelNoise()) / softmaxTau), false));
    }

    /**
     * Finalizes softmax result.<br>
     *
     * @param result result matrix.
     * @return finalized result matrix.
     * @throws MatrixException thrown if matrix operation fails.
     */
    private Matrix finalizeSoftmax(Matrix result) throws MatrixException {
        final double maxValue = result.max();
        result.apply(new UnaryFunction(value -> Math.exp(value - maxValue)), true);
        result.divideBy(result.sum());
        return result;
    }

    /**
     * Returns Gumbel noise.<br>
     *
     * @return Gumbel noise.
     */
    private double getGumbelNoise() {
        double epsilon = 10E-20;
        return -Math.log(-Math.log(random.nextDouble() + epsilon) + epsilon);
    }

    /**
     * Returns softmax gradient of matrix.<br>
     * Assumes that input matrix is softmax result.<br>
     *
     * @return softmax gradient of matrix.
     * @throws MatrixException thrown if index dimensions do not match.
     */
    public Matrix softmaxGrad() throws MatrixException {
        if (getColumns() != 1) throw new MatrixException("Matrix must be a column vector.");
        return new SoftmaxGradientMatrixOperation(getRows(), getRows(), getDepth()).apply(this);
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
     * Sets filter depth.
     *
     * @param filterDepth filter depth.
     */
    public void setFilterDepth(int filterDepth) {
        this.filterDepth = filterDepth;
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
     * Returns filter depth.
     *
     * @return filter depth.
     */
    public int getFilterDepth() {
        return filterDepth;
    }

    /**
     * Sets if convolution is depth separable.
     *
     * @param isDepthSeparable is true convolution is depth separable.
     */
    public void setIsDepthSeparable(boolean isDepthSeparable) {
        this.isDepthSeparable = isDepthSeparable;
    }

    /**
     * Returns if convolution is depth separable.
     *
     * @return if true convolution is depth separable.
     */
    public boolean getIsDepthSeparable() {
        return isDepthSeparable;
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix applyConvolve(Matrix filter) throws MatrixException {
        return new ConvolutionMatrixOperation(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1, getFilterDepth(), getDepth(), filter.getRows(), filter.getColumns(), getDilation(), getStride(), getIsDepthSeparable()).apply(this, filter);
    }

    /**
     * Calculates convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix applyCrosscorrelate(Matrix filter) throws MatrixException {
        return new CrosscorrelationMatrixOperation(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1, getFilterDepth(), getDepth(), filter.getRows(), filter.getColumns(), getDilation(), getStride(), getIsDepthSeparable()).apply(this, filter);
    }

    /**
     * Calculates Winograd convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix applyWinogradConvolve(Matrix filter) throws MatrixException {
        return new WinogradConvolutionMatrixOperation(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1, getFilterDepth()).apply(this, filter);
    }

    /**
     * Calculates Winograd convolution between this matrix and filter matrix.
     *
     * @param filter filter matrix.
     * @param A A matrix
     * @param AT A transposed matrix
     * @param C C matrix
     * @param CT C transposed matrix
     * @param G G matrix
     * @param GT G transposed matrix
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix applyWinogradConvolve(Matrix filter, Matrix A, Matrix AT, Matrix C, Matrix CT, Matrix G, Matrix GT) throws MatrixException {
        return new WinogradConvolutionMatrixOperation(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1, getFilterDepth(), A, AT, C, CT, G, GT).apply(this, filter);
    }

    /**
     * Calculates Winograd convolution between this matrix and filter matrix.
     *
     * @param preprocessedFilter preprocessed filter matrix.
     * @param A A matrix
     * @param AT A transposed matrix
     * @param C C matrix
     * @param CT C transposed matrix
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix applyWinogradConvolve(Matrix preprocessedFilter, Matrix A, Matrix AT, Matrix C, Matrix CT) throws MatrixException {
        return new WinogradConvolutionMatrixOperation(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1, getFilterDepth(), A, AT, C, CT).apply(this, preprocessedFilter);
    }

    /**
     * Calculates max pooling operation for matrix and returns max arguments.
     *
     * @param maxPos maximum position for each result row and column value.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix applyMaxPool(HashMap<Integer, Integer> maxPos) throws MatrixException {
        return new MaxPoolMatrixOperation(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1, getDepth(), getFilterRowSize(), getFilterColumnSize(), getDilation(), getStride()).apply(this, maxPos);
    }

    /**
     * Calculates random pooling operation for matrix and returns input positions.
     *
     * @param inputPos input position for each result row and column value.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix applyRandomPool(HashMap<Integer, Integer> inputPos) throws MatrixException {
        return new RandomPoolMatrixOperation(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1, getDepth(), getFilterRowSize(), getFilterColumnSize(), getDilation(), getStride()).apply(this, inputPos);
    }

    /**
     * Calculates cyclic pooling operation for matrix and returns input positions.
     *
     * @param inputPos input position for each result row and column value.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix applyCyclicPool(HashMap<Integer, Integer> inputPos) throws MatrixException {
        return new CyclicPoolMatrixOperation(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1, getDepth(), getFilterRowSize(), getFilterColumnSize(), getDilation(), getStride()).apply(this, inputPos);
    }

    /**
     * Calculates average pooling operation for matrix.
     *
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix applyAveragePool() throws MatrixException {
        return new AveragePoolMatrixOperation(getRows() - getFilterRowSize() + 1, getColumns() - getFilterColumnSize() + 1, getDepth(), getFilterRowSize(), getFilterColumnSize(), getDilation(), getStride()).apply(this);
    }

    /**
     * Classifies matrix assuming multi-label classification.
     *
     * @return classified matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix classify() throws MatrixException {
        return new ClassifyMatrixOperation(getRows(), getColumns(), getDepth()).apply(this, getNewMatrix());
    }

    /**
     * Classifies matrix assuming multi-label classification.
     *
     * @return classified matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix classify(double multiLabelThreshold) throws MatrixException {
        return new ClassifyMatrixOperation(getRows(), getColumns(), getDepth(), multiLabelThreshold).apply(this, getNewMatrix());
    }

    /**
     * Encodes to value to bit column vector.
     *
     * @param value value
     * @param maxBits max number of bits.
     * @return bit column vector.
     * @throws MatrixException throws exception if binary code size is exceeding number of maximum bits.
     */
    public static Matrix encodeToBitColumnVector(int value, int maxBits) throws MatrixException {
        String binaryCode = String.format("%" + maxBits + "s", Integer.toBinaryString(value)).replaceAll(" ", "0");
        if (binaryCode.length() > maxBits) throw new MatrixException("Binary code: '" + binaryCode + "' has length " + binaryCode.length() + " exceeding number of maximum bits " + maxBits);
        Matrix encodedMatrix = new SMatrix(binaryCode.length(), 1, 1);
        int binaryCodeLength = binaryCode.length();
        for (int charIndex = 0; charIndex < binaryCodeLength; charIndex++) {
            char charAt = binaryCode.charAt(charIndex);
            if (charAt == '1') encodedMatrix.setValue(charIndex, 0, 0, 1);
        }
        return encodedMatrix;
    }

    /**
     * Encodes bit column vector value
     *
     * @return value
     * @throws MatrixException throws exception if matrix is not bit column vector.
     */
    public int encodeToValue() throws MatrixException {
        if (getColumns() != 1) throw new MatrixException("Matrix must be column vector.");
        int rows = getRows();
        int result = 0;
        for (int row = 0; row < rows; row++) {
            double value = getValue(row, 0, 0);
            if (!(value == 0 || value == 1)) throw new MatrixException("Bit column vector must contains values of 0 or 1.");
            result += value * Math.pow(2, (rows - 1) - row);
        }
        return result;
    }

    /**
     * Returns number of bits needed to represent value.
     *
     * @param value value.
     * @return number of bits needed to represent value.
     */
    public static int numberOfBits(int value) {
        return (int)Math.floor((Math.log10(value) / Math.log10(2) + 1));
    }

    /**
     * Samples random variable from gamma distribution.<br>
     * Reference: <a href="https://www.hongliangjie.com/2012/12/19/how-to-generate-gamma-random-variables/">...</a>
     *
     * @param shape shape (alpha) parameter
     * @param scale scale (beta) parameter
     * @param random random function
     * @return random variable from gamma distribution
     */
    public static double sampleGamma(double shape, double scale, Random random) {
        if (shape > 1) {
            double d = shape - 1 / (double)3;
            double c = 1 / Math.sqrt(9 * d);
            while (true) {
                double gaussian = random.nextGaussian();
                if (gaussian > - 1 / c) {
                    double uniform = random.nextDouble();
                    double V = Math.pow(1 + c * gaussian, 3);
                    if (Math.log(uniform) < 0.5 * Math.pow(gaussian, 2) + d - d * V + d * Math.log(V)) return d * V / scale;
                }
            }
        }
        else return sampleGamma(shape + 1, scale, random) * Math.pow(random.nextDouble(), 1 / shape);
    }

    /**
     * Clips value between minimum value and maximum value.
     *
     * @param value value
     * @param minValue minimum value.
     * @param maxValue maximum value.
     * @return clipped value.
     */
    public static double clipValue(double value, double minValue, double maxValue) {
        return (value < minValue) ? minValue : Math.min(value, maxValue);
    }

}
