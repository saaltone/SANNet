package utils.procedure.expression;

import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;
import utils.procedure.node.Node;

import java.util.Set;

/**
 * Defines expression
 *
 */
public interface Expression {

    /**
     * Returns name of expression.
     *
     * @return name of expression.
     */
    String getExpressionName();

    /**
     * Returns signature of operation.
     *
     * @return signature of operation.
     */
    String getOperationSignature();

    /**
     * Returns expression ID
     *
     * @return expression ID
     */
    int getExpressionID();

    /**
     * Returns first argument of expression.
     *
     * @return first argument of expression.
     */
    Node getArgument1();

    /**
     * Returns second argument of expression.
     *
     * @return returns null unless overloaded by abstract binary expression class.
     */
    Node getArgument2();

    /**
     * Returns result of expression.
     *
     * @return result of expression.
     */
    Node getResult();

    /**
     * Sets next expression for expression calculation chain.
     *
     * @param nextExpression next expression.
     */
    void setNextExpression(Expression nextExpression);

    /**
     * Sets previous expression for gradient calculation chain.
     *
     * @param previousExpression previous expression.
     */
    void setPreviousExpression(Expression previousExpression);

    /**
     * Resets expression.
     *
     */
    void reset();

    /**
     * Calculates entire expression chain including normalization and regulation.
     *
     * @param sampleIndex sample index
     * @param firstSampleIndex first sample index
     * @param lastSampleIndex last key of inputs
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void calculateExpressionStep(int sampleIndex, int firstSampleIndex, int lastSampleIndex) throws MatrixException, DynamicParamException;

    /**
     * Calculates entire expression chain including normalization and regulation.
     *
     * @param sampleIndices sample indices
     * @param firstSampleIndex first sample index
     * @param lastSampleIndex last sample index
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void calculateExpressionStep(Set<Integer> sampleIndices, int firstSampleIndex, int lastSampleIndex) throws MatrixException, DynamicParamException;

    /**
     * Calculates entire gradient expression chain including normalization and regulation.
     *
     * @param sampleIndex sample index
     * @param lastSampleIndex last sample index
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void calculateGradientStep(int sampleIndex, int lastSampleIndex) throws MatrixException, DynamicParamException;

    /**
     * Calculates entire gradient expression chain including normalization and regulation.
     *
     * @param sampleIndices sample indices
     * @param lastSampleIndex last sample index
     * @param numberOfGradientSteps number of gradient steps taken
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void calculateGradientStep(Set<Integer> sampleIndices, int lastSampleIndex, int numberOfGradientSteps) throws MatrixException, DynamicParamException;

    /**
     * Cumulates error from regularization.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return updated error value.
     */
    double cumulateRegularizationError() throws MatrixException, DynamicParamException;

    /**
     * Prints expression chain.
     *
     */
    void printExpressionChain();

    /**
     * Prints expression chain.
     *
     */
    void invokePrintExpressionChain();

    /**
     * Prints gradient chain.
     *
     */
    void printGradientChain();

    /**
     * Prints gradient chain.
     *
     */
    void invokePrintGradientChain();

    /**
     * Prints expression.
     *
     */
    void printExpression();

    /**
     * Prints gradient.
     *
     */
    void printGradient();

}
