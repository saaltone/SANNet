/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package utils.matrix.operation;

import utils.configurable.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.Mask;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements F(2x2, 3x3) Winograd convolution.<br>
 *
 * Reference: <a href="http://cs231n.stanford.edu/reports/2016/pdfs/117_Report.pdf">...</a> <br>
 *
 */
public class WinogradConvolutionMatrixOperation extends AbstractMatrixOperation {

    /**
     * Input matrix.
     *
     */
    private transient Matrix input;

    /**
     * Filter matrix.
     *
     */
    private transient Matrix filter;

    /**
     * Result matrix.
     *
     */
    private transient Matrix result;

    /**
     * A matrix for Winograd convolution.
     *
     */
    private final Matrix A;

    /**
     * A transposed matrix for Winograd convolution.
     *
     */
    private final Matrix AT;

    /**
     * C matrix for Winograd convolution.
     *
     */
    private final Matrix C;

    /**
     * C transposed matrix for Winograd convolution.
     *
     */
    private final Matrix CT;

    /**
     * G matrix for Winograd convolution.
     *
     */
    private final Matrix G;

    /**
     * G transposed matrix for Winograd convolution.
     *
     */
    private final Matrix GT;

    /**
     * Constructor for Winograd convolution.
     *
     * @param rows rows
     * @param columns columns
     * @param depth depth.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public WinogradConvolutionMatrixOperation(int rows, int columns, int depth) throws MatrixException, DynamicParamException {
        super(rows, columns, depth, false, 2);
        AT = getATMatrix(getDepth());
        A = AT.transpose().copy(true);

        C = getCMatrix(getDepth());
        CT = C.transpose().copy(true);

        G = getGMatrix(getDepth());
        GT = G.transpose().copy(true);
    }

    /**
     * Return AT matrix.
     *
     * @param depth depth
     * @return AT matrix.
     */
    public static Matrix getATMatrix(int depth) {
        Matrix AT = new DMatrix(2, 4, depth).copy(true);
        for (int currentDepth = 0; currentDepth < depth; currentDepth++) {
            AT.setValue(0, 0, currentDepth, 1);
            AT.setValue(0, 1, currentDepth, 1);
            AT.setValue(0, 2, currentDepth, 1);
            AT.setValue(0, 3, currentDepth, 0);
            AT.setValue(1, 0, currentDepth, 0);
            AT.setValue(1, 1, currentDepth, 1);
            AT.setValue(1, 2, currentDepth, -1);
            AT.setValue(1, 3, currentDepth, -1);
        }
        maskZeros(AT);
        return AT;
    }

    /**
     * Return C matrix.
     *
     * @param depth depth
     * @return C matrix.
     */
    public static Matrix getCMatrix(int depth) {
        Matrix C = new DMatrix(4, 4, depth).copy(true);
        for (int currentDepth = 0; currentDepth < depth; currentDepth++) {
            C.setValue(0, 0, currentDepth, 1);
            C.setValue(0, 1, currentDepth, 0);
            C.setValue(0, 2, currentDepth, -1);
            C.setValue(0, 3, currentDepth, 0);
            C.setValue(1, 0, currentDepth, 0);
            C.setValue(1, 1, currentDepth, 1);
            C.setValue(1, 2, currentDepth, 1);
            C.setValue(1, 3, currentDepth, 0);
            C.setValue(2, 0, currentDepth, 0);
            C.setValue(2, 1, currentDepth, -1);
            C.setValue(2, 2, currentDepth, 1);
            C.setValue(2, 3, currentDepth, 0);
            C.setValue(3, 0, currentDepth, 0);
            C.setValue(3, 1, currentDepth, 1);
            C.setValue(3, 2, currentDepth, 0);
            C.setValue(3, 3, currentDepth, -1);
        }
        maskZeros(C);
        return C;
    }

    /**
     * Return G matrix.
     *
     * @param depth depth
     * @return G matrix.
     */
    public static Matrix getGMatrix(int depth) {
        Matrix G = new DMatrix(4, 3, depth).copy(true);
        for (int currentDepth = 0; currentDepth < depth; currentDepth++) {
            G.setValue(0, 0, currentDepth, 1);
            G.setValue(0, 1, currentDepth, 0);
            G.setValue(0, 2, currentDepth, 0);
            G.setValue(1, 0, currentDepth, 1/(double)2);
            G.setValue(1, 1, currentDepth, 1/(double)2);
            G.setValue(1, 2, currentDepth, 1/(double)2);
            G.setValue(2, 0, currentDepth, 1/(double)2);
            G.setValue(2, 1, currentDepth, -1/(double)2);
            G.setValue(2, 2, currentDepth, 1/(double)2);
            G.setValue(3, 0, currentDepth, 0);
            G.setValue(3, 1, currentDepth, 0);
            G.setValue(3, 2, currentDepth, 1);
        }
        maskZeros(G);
        return G;
    }

    /**
     * Constructor for Winograd convolution.
     *
     * @param rows rows
     * @param columns columns
     * @param depth depth.
     * @param A A matrix
     * @param AT A transposed matrix
     * @param C C matrix
     * @param CT C transposed matrix
     * @param G G matrix
     * @param GT G transposed matrix
     */
    public WinogradConvolutionMatrixOperation(int rows, int columns, int depth, Matrix A, Matrix AT, Matrix C, Matrix CT, Matrix G, Matrix GT) {
        super(rows, columns, depth, false, 2);
        this.A = A;
        this.AT = AT;
        this.C = C;
        this.CT = CT;
        this.G = G;
        this.GT = GT;
    }

    /**
     * Constructor for Winograd convolution.
     *
     * @param rows rows
     * @param columns columns
     * @param depth depth.
     * @param A A matrix
     * @param AT A transposed matrix
     * @param C C matrix
     * @param CT C transposed matrix
     */
    public WinogradConvolutionMatrixOperation(int rows, int columns, int depth, Matrix A, Matrix AT, Matrix C, Matrix CT) {
        super(rows, columns, depth, false, 2);
        this.A = A;
        this.AT = AT;
        this.C = C;
        this.CT = CT;
        this.G = null;
        this.GT = null;
    }

    /**
     * Masks matrix positions with zero value to avoid unnecessary calculations.
     *
     * @param matrix matrix to be masked.
     */
    private static void maskZeros(Matrix matrix) {
        matrix.setMask();
        int matrixRows = matrix.getRows();
        int matrixColumns = matrix.getColumns();
        int matrixDepth = matrix.getDepth();
        Mask matrixMask = matrix.getMask();
        for (int depth = 0; depth < matrixDepth; depth++) {
            for (int row = 0; row < matrixRows; row++) {
                for (int column = 0; column < matrixColumns; column++) {
                    if (matrix.getValue(row, column, depth) == 0) matrixMask.setMask(row, column, depth, true);
                }
            }
        }
    }

    /**
     * Applies operation.
     *
     * @param input input matrix.
     * @param filter filter matrix.
     * @return result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix apply(Matrix input, Matrix filter) throws MatrixException {
        this.input = input;
        this.filter = filter;
        this.result = input.getNewMatrix(getRows(), getColumns(), getDepth()).copy(true);
        applyMatrixOperation(input, null, result);
        return result;
    }

    /**
     * Applies operation.
     *
     * @param row    current row.
     * @param column current column.
     * @param depth  current depth.
     * @param value  current value.
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void apply(int row, int column, int depth, double value, Matrix result) throws MatrixException {
        Matrix currentFilter = filter.copy(true);
        currentFilter.slice(0, 0, depth, filter.getTotalRows() - 1, filter.getTotalColumns() - 1, depth);
        Matrix Gprime;
        if (G != null) {
            G.slice(0, 0, depth, 3, 2, depth);
            GT.slice(0, 0, depth, 2, 3, depth);
            Matrix G1 = G.dot(currentFilter);
            Gprime = G1.dot(GT);
        }
        else Gprime = currentFilter;
        Matrix C1 = null;
        Matrix currentInput = input.copy(true);
        for (int inputDepth = 0; inputDepth < currentInput.getDepth(); inputDepth++) {
            currentInput.slice(row, column, inputDepth, row + 3, column + 3, inputDepth);
            CT.slice(0, 0, depth, 3, 3, depth);
            C1 = C1 == null ? CT.dot(currentInput) : C1.add(CT.dot(currentInput));
        }

        C.slice(0, 0, depth, 3, 3, depth);
        assert C1 != null;
        Matrix CPrime = C1.dot(C);
        Matrix GCprime = Gprime.dot(CPrime);
        AT.slice(0, 0, depth, 1, 3, depth);
        Matrix AT1 = AT.dot(GCprime);
        this.result.slice(row, column, depth, row + 1, column + 1, depth);
        A.slice(0, 0, depth, 3, 1, depth);
        this.result.setEqualTo(AT1.dot(A));
        currentInput.unslice();
        this.result.unslice();
    }

}
