package utils.procedure.expression;

import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.procedure.node.Node;

import java.io.Serializable;

/**
 * Class that describes expression for Winograd convolution operation.<br>
 *
 */
public class WinogradConvolutionExpression extends AbstractBinaryExpression implements Serializable {

    /**
     * Stride of crosscorrelation operation.
     *
     */
    private final int stride;

    /**
     * Dilation step size for crosscorrelation operation.
     *
     */
    private final int dilation;

    /**
     * Filter row size;
     *
     */
    private final int filterRowSize;

    /**
     * Filter column size;
     *
     */
    private final int filterColumnSize;

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
     * Preprocessed filter.
     *
     */
    private Matrix preprocessedFilter;

    /**
     * Constructor for Winograd convolution operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param argument2 second argument.
     * @param result result of expression.
     * @param stride stride of crosscorrelation operation.
     * @param dilation dilation step size for crosscorrelation operation.
     * @param filterRowSize filter row size.
     * @param filterColumnSize filter column size.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public WinogradConvolutionExpression(int expressionID, Node argument1, Node argument2, Node result, int stride, int dilation, int filterRowSize, int filterColumnSize) throws MatrixException {
        super("WINOGRAD_CONVOLUTION", "WINOGRAD_CONVOLUTION", expressionID, argument1, argument2, result);
        this.stride = stride;
        this.dilation = dilation;
        this.filterRowSize = filterRowSize;
        this.filterColumnSize = filterColumnSize;
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
     * Calculates expression.
     *
     */
    public void calculateExpression() {
    }

    /**
     * Calculates expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpression(int index) throws MatrixException {
        if (argument1.getMatrix(index) == null || argument2.getMatrix(index) == null) throw new MatrixException(getExpressionName() + ": Arguments for operation not defined");
        argument1.getMatrix(index).setStride(stride);
        argument1.getMatrix(index).setDilation(dilation);
        argument1.getMatrix(index).setFilterRowSize(filterRowSize);
        argument1.getMatrix(index).setFilterColumnSize(filterColumnSize);
        if (preprocessedFilter == null) preprocessedFilter = G.dot(argument2.getMatrix(index)).dot(GT);
        result.setMatrix(index, argument1.getMatrix(index).winogradConvolve(preprocessedFilter, A, AT, C, CT));
//        result.setMatrix(index, argument1.getMatrix(index).winogradConvolve(argument2.getMatrix(index), A, AT, C, CT, G, GT));
    }

    /**
     * Calculates gradient of expression.
     *
     */
    public void calculateGradient() {
    }

    /**
     * Calculates gradient of expression.
     *
     * @param index data index.
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    public void calculateGradient(int index) throws MatrixException {
        if (result.getGradient(index) == null) throw new MatrixException(getExpressionName() + ": Result gradient not defined.");
        result.getGradient(index).setStride(stride);
        result.getGradient(index).setDilation(dilation);
        result.getGradient(index).setFilterRowSize(filterRowSize);
        result.getGradient(index).setFilterColumnSize(filterColumnSize);
        argument1.cumulateGradient(index, result.getGradient(index).crosscorrelateInputGradient(argument2.getMatrix(index)), false);
        argument2.cumulateGradient(index, result.getGradient(index).crosscorrelateFilterGradient(argument1.getMatrix(index)), false);
        preprocessedFilter = null;
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        printSpecificBinaryExpression();
    }

    /**
     * Prints gradient.
     *
     */
    public void printGradient() {
        printArgument1Gradient(false, getExpressionName() + "_GRADIENT(d" + result.getName() + ", " + argument2.getName() + ")");
        printArgument2Gradient(false, false, getExpressionName() + "_GRADIENT(d" + result.getName() + ", " + argument1.getName() + ")");
    }

}
