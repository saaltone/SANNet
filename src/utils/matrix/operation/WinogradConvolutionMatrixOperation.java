/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix.operation;

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
    private Matrix input;

    /**
     * Filter matrix.
     *
     */
    private Matrix filter;

    /**
     * Result matrix.
     *
     */
    private Matrix result;

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
     */
    public WinogradConvolutionMatrixOperation(int rows, int columns, int depth) throws MatrixException {
        super(rows, columns, depth, false, 2);
        AT = new DMatrix(2, 4, 1);
        AT.setValue(0, 0, 0, 1);
        AT.setValue(0, 1, 0, 1);
        AT.setValue(0, 2, 0, 1);
        AT.setValue(0, 3, 0, 0);
        AT.setValue(1, 0, 0, 0);
        AT.setValue(1, 1, 0, 1);
        AT.setValue(1, 2, 0, -1);
        AT.setValue(1, 3, 0, -1);
        maskZeros(AT);
        A = AT.transpose();

        C = new DMatrix(4, 4, 1);
        C.setValue(0, 0, 0, 1);
        C.setValue(0, 1, 0, 0);
        C.setValue(0, 2, 0, -1);
        C.setValue(0, 3, 0, 0);
        C.setValue(1, 0, 0, 0);
        C.setValue(1, 1, 0, 1);
        C.setValue(1, 2, 0, 1);
        C.setValue(1, 3, 0, 0);
        C.setValue(2, 0, 0, 0);
        C.setValue(2, 1, 0, -1);
        C.setValue(2, 2, 0, 1);
        C.setValue(2, 3, 0, 0);
        C.setValue(3, 0, 0, 0);
        C.setValue(3, 1, 0, 1);
        C.setValue(3, 2, 0, 0);
        C.setValue(3, 3, 0, -1);
        maskZeros(C);
        CT = C.transpose();

        G = new DMatrix(4, 3, 1);
        G.setValue(0, 0, 0, 1);
        G.setValue(0, 1, 0, 0);
        G.setValue(0, 2, 0, 0);
        G.setValue(1, 0, 0, 1/(double)2);
        G.setValue(1, 1, 0, 1/(double)2);
        G.setValue(1, 2, 0, 1/(double)2);
        G.setValue(2, 0, 0, 1/(double)2);
        G.setValue(2, 1, 0, -1/(double)2);
        G.setValue(2, 2, 0, 1/(double)2);
        G.setValue(3, 0, 0, 0);
        G.setValue(3, 1, 0, 0);
        G.setValue(3, 2, 0, 1);
        maskZeros(G);
        GT = G.transpose();
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
    private void maskZeros(Matrix matrix) {
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
        this.result = input.getNewMatrix(getRows(), getColumns(), getDepth());
        applyMatrixOperation();
        return result;
    }

    /**
     * Returns target matrix.
     *
     * @return target matrix.
     */
    protected Matrix getTargetMatrix() {
        return input;
    }

    /**
     * Returns another matrix used in operation.
     *
     * @return another matrix used in operation.
     */
    public Matrix getOther() {
        return null;
    }

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param depth current depth.
     * @param value current value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void apply(int row, int column, int depth, double value) throws MatrixException {
        input.slice(row, column, depth, row + 3, column + 3, depth);
        Matrix Gprime;
        if (G != null) {
            Matrix G1 = G.dot(filter);
            Gprime = G1.dot(GT);
        }
        else Gprime = filter;
        Matrix C1 = CT.dot(input);
        Matrix CPrime = C1.dot(C);
        Matrix GCprime = Gprime.dot(CPrime);
        Matrix AT1 = AT.dot(GCprime);
        result.slice(row, column, depth, row + 1, column + 1, depth);
        result.setEqualTo(AT1.dot(A));
        input.unslice();
        result.unslice();
    }

}
