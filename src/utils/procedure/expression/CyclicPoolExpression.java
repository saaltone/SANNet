package utils.procedure.expression;

import utils.matrix.MatrixException;
import utils.matrix.operation.CyclicPoolGradientMatrixOperation;
import utils.matrix.operation.CyclicPoolMatrixOperation;
import utils.procedure.node.Node;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Stack;

/**
 * Defines cyclic pool expression.
 *
 */
public class CyclicPoolExpression extends AbstractUnaryExpression implements Serializable {

    /**
     * Reference to cyclic pool matrix operation.
     *
     */
    private final CyclicPoolMatrixOperation cyclicPoolMatrixOperation;

    /**
     * Reference to cyclic pool gradient matrix operation.
     *
     */
    private final CyclicPoolGradientMatrixOperation cyclicPoolGradientMatrixOperation;

    /**
     * Input positions for cyclic pool operation.
     *
     */
    private final HashMap<Integer, HashMap<Integer, Integer>> inputPos = new HashMap<>();

    /**
     * Stack for caching input position instances.
     *
     */
    private final Stack<HashMap<Integer, Integer>> inputPosCache = new Stack<>();

    /**
     * Constructor for cyclic pooling operation.
     *
     * @param expressionID unique ID for expression.
     * @param argument1 first argument.
     * @param result result of expression.
     * @param stride stride of pooling operation.
     * @param filterRowSize filter row size for operation.
     * @param filterColumnSize filter column size for operation.
     * @throws MatrixException throws exception if expression arguments are not defined.
     */
    public CyclicPoolExpression(int expressionID, Node argument1, Node result, int stride, int filterRowSize, int filterColumnSize) throws MatrixException {
        super("RANDOM_POOL", "RANDOM_POOL", expressionID, argument1, result);
        cyclicPoolMatrixOperation = new CyclicPoolMatrixOperation(result.getRows(), result.getColumns(), argument1.getColumns(), filterRowSize, filterColumnSize, stride);
        cyclicPoolGradientMatrixOperation = new CyclicPoolGradientMatrixOperation(result.getRows(), result.getColumns(), argument1.getColumns(), stride);
    }

    /**
     * Returns true is expression is executed as single step otherwise false.
     *
     * @return true is expression is executed as single step otherwise false.
     */
    protected boolean executeAsSingleStep() {
        return false;
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
     * @param sampleIndex sample index
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpression(int sampleIndex) throws MatrixException {
        if (argument1.getMatrix(sampleIndex) == null) throw new MatrixException(getExpressionName() + ": Arguments for operation not defined");
        if (!inputPosCache.empty()) inputPos.put(sampleIndex, inputPosCache.pop());
        else inputPos.put(sampleIndex, new HashMap<>());
        cyclicPoolMatrixOperation.apply(argument1.getMatrix(sampleIndex), inputPos.get(sampleIndex), result.getNewMatrix(sampleIndex));
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
     * @param sampleIndex sample index
     * @throws MatrixException throws exception if calculation of gradient fails.
     */
    public void calculateGradient(int sampleIndex) throws MatrixException {
        if (result.getGradient(sampleIndex) == null) throw new MatrixException(getExpressionName() + ": Result gradient not defined.");
        if (!inputPos.containsKey(sampleIndex)) throw new MatrixException("Input positions for gradient calculation are not defined.");
        if (!argument1.isStopGradient()) argument1.cumulateGradient(sampleIndex, cyclicPoolGradientMatrixOperation.apply(result.getGradient(sampleIndex), inputPos.get(sampleIndex), argument1.getEmptyMatrix()), false);
        inputPosCache.push(inputPos.remove(sampleIndex));
    }

    /**
     * Prints expression.
     *
     */
    public void printExpression() {
        print();
        System.out.println(getExpressionName() + "(" + argument1.getName() + ") = " + result.getName());
    }

    /**
     * Prints gradient.
     *
     */
    public void printGradient() {
        printArgument1Gradient(false, getExpressionName() + "_GRADIENT(" + getResultGradientName() + ", ARGMAX(" + argument1.getName() +"))");
    }

}