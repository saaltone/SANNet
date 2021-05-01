package utils.matrix.operation;

import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Class that implements F(2x2, 3x3) Winograd convolution.<br>
 *
 * Reference: http://cs231n.stanford.edu/reports/2016/pdfs/117_Report.pdf <br>
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
     */
    public WinogradConvolutionMatrixOperation(int rows, int columns) {
        super(rows, columns, false);
        AT = new DMatrix(2, 4);
        AT.setValue(0, 0, 1);
        AT.setValue(0, 1, 1);
        AT.setValue(0, 2, 1);
        AT.setValue(0, 3, 0);
        AT.setValue(1, 0, 0);
        AT.setValue(1, 1, 1);
        AT.setValue(1, 2, -1);
        AT.setValue(1, 3, -1);
        maskZeros(AT);
        A = AT.transpose();

        C = new DMatrix(4, 4);
        C.setValue(0, 0, 1);
        C.setValue(0, 1, 0);
        C.setValue(0, 2, -1);
        C.setValue(0, 3, 0);
        C.setValue(1, 0, 0);
        C.setValue(1, 1, 1);
        C.setValue(1, 2, 1);
        C.setValue(1, 3, 0);
        C.setValue(2, 0, 0);
        C.setValue(2, 1, -1);
        C.setValue(2, 2, 1);
        C.setValue(2, 3, 0);
        C.setValue(3, 0, 0);
        C.setValue(3, 1, 1);
        C.setValue(3, 2, 0);
        C.setValue(3, 3, -1);
        maskZeros(C);
        CT = C.transpose();

        G = new DMatrix(4, 3);
        G.setValue(0, 0, 1);
        G.setValue(0, 1, 0);
        G.setValue(0, 2, 0);
        G.setValue(1, 0, 1/(double)2);
        G.setValue(1, 1, 1/(double)2);
        G.setValue(1, 2, 1/(double)2);
        G.setValue(2, 0, 1/(double)2);
        G.setValue(2, 1, -1/(double)2);
        G.setValue(2, 2, 1/(double)2);
        G.setValue(3, 0, 0);
        G.setValue(3, 1, 0);
        G.setValue(3, 2, 1);
        maskZeros(G);
        GT = G.transpose();
    }

    /**
     * Constructor for Winograd convolution.
     *
     * @param rows rows
     * @param columns columns
     * @param A A matrix
     * @param AT A transposed matrix
     * @param C C matrix
     * @param CT C transposed matrix
     * @param G G matrix
     * @param GT G transposed matrix
     */
    public WinogradConvolutionMatrixOperation(int rows, int columns, Matrix A, Matrix AT, Matrix C, Matrix CT, Matrix G, Matrix GT) {
        super(rows, columns, false);
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
     * @param A A matrix
     * @param AT A transposed matrix
     * @param C C matrix
     * @param CT C transposed matrix
     */
    public WinogradConvolutionMatrixOperation(int rows, int columns, Matrix A, Matrix AT, Matrix C, Matrix CT) {
        super(rows, columns, false);
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
        for (int row = 0; row < matrix.getRows(); row++) {
            for (int column = 0; column < matrix.getColumns(); column++) {
                if (matrix.getValue(row, column) == 0) matrix.getMask().setMask(row, column, true);
            }
        }
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

    /**
     * Applies operation.
     *
     * @param row current row.
     * @param column current column.
     * @param value current value.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void apply(int row, int column, double value) throws MatrixException {
        input.sliceAt(row, column, row + 3, column + 3);
        Matrix Gprime = G != null ? G.dot(filter).dot(GT) : filter;
        Matrix Cprime = CT.dot(input).dot(C);
        Matrix GCprime = Gprime.multiply(Cprime);
        result.sliceAt(row, column, row + 1, column + 1);
        AT.dot(GCprime).dot(A, result);
        input.unslice();
        result.unslice();
    }

}
