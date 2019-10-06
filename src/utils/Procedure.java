/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package utils;

import java.io.Serializable;
import java.util.*;

/**
 * Defines computable procedure that has chain of forward computable expressions and backward computable gradient expressions (automatic gradient).
 *
 */
public class Procedure implements Serializable {

    private static final long serialVersionUID = 9207418704022664014L;

    /**
     * Input node.
     *
     */
    private final Node inputNode;

    /**
     * Output node.
     *
     */
    private final Node outputNode;

    /**
     * List of expressions for forward calculation.
     *
     */
    private final LinkedList<Expression> expressions;

    /**
     * List of expressions for backward gradient calculation.
     *
     */
    private final LinkedList<Expression> gradientExpressions;

    /**
     * Set of dependent output input node pairs as node links.
     *
     */
    private final HashSet<NodeLink> dependentNodes;

    /**
     * Matrices attached to a specific node. Used to acquire gradients of related matrices.
     *
     */
    private final HashMap<Matrix, Node> registeredMatrixMap;

    /**
     * Constructor for procedure.
     *
     * @param inputNode input node for procedure.
     * @param outputNode input node for procedure.
     * @param expressions expressions for forward calculation.
     * @param gradientExpressions gradient expressions for backward gradient calculation.
     * @param dependentNodes node dependencies as node links for output input pair updates.
     * @param registeredMatrixMap map of registered matrices.
     */
    public Procedure(Node inputNode, Node outputNode, LinkedList<Expression> expressions, LinkedList<Expression> gradientExpressions, HashSet<NodeLink> dependentNodes, HashMap<Matrix, Node> registeredMatrixMap) {
        this.inputNode = inputNode;
        this.outputNode = outputNode;
        this.expressions = expressions;
        this.gradientExpressions = gradientExpressions;
        this.dependentNodes = dependentNodes;
        this.registeredMatrixMap = registeredMatrixMap;
    }

    /**
     * Returns number of expressions in procedure.
     *
     * @return number of expressions in procedure.
     */
    public int getSize() {
        return expressions.size();
    }

    /**
     * Resets data for every index in nodes of procedure.
     *
     * @throws MatrixException throws exception if reset operation fails.
     */
    public void reset() throws MatrixException {
        for (Expression expression : expressions) expression.resetExpression();
    }

    /**
     * Resets data for specific index in nodes of procedure.
     *
     * @param index data index is node.
     * @throws MatrixException throws exception if reset operation fails.
     */
    public void reset(int index) throws MatrixException {
        for (Expression expression : expressions) expression.resetExpression(index);
    }

    /**
     * Returns node corresponding specific matrix.
     *
     * @param matrix matrix.
     * @return node corresponding specific matrix
     */
    public Node getNode(Matrix matrix) {
        return registeredMatrixMap.get(matrix);
    }

    /**
     * Returns expression by ID.
     *
     * @param expressionID expression ID.
     * @return returned expression.
     */
    public Expression getExpression(int expressionID) {
        return expressions.get(expressionID);
    }

    /**
     * Gets input node.
     *
     * @return input node.
     */
    public Node getInputNode() {
        return inputNode;
    }

    /**
     * Gets output node.
     *
     * @return output node.
     */
    public Node getOutputNode() {
        return outputNode;
    }

    /**
     * Calculates chain of forward expressions.
     *
     * @param index specific data index.
     * @param inputMatrix input matrix.
     * @return output node.
     * @throws MatrixException throws exception if calculation fails.
     */
    public Node calculateExpression(int index, Matrix inputMatrix) throws MatrixException {
        updateDependencies(index);
        getInputNode().setMatrix(index, inputMatrix);
        for (Expression expression : expressions) expression.calculateExpression(index);
        return outputNode;
    }

    /**
     * Checks if procedure has dependencies between output and input nodes.
     *
     * @return returns true if there are dependencies otherwise returns false.
     */
    public boolean hasDependencies() {
        return dependentNodes.size() > 0;
    }

    /**
     * Resets dependencies between output and input nodes.
     *
     */
    public void resetDependencies() {
        for (NodeLink nodeLink : dependentNodes) nodeLink.reset();
    }

    /**
     * Updates data of node dependencies for expression calculation phase.
     *
     * @param index index to data for which dependencies are updates.
     */
    private void updateDependencies(int index) {
        for (NodeLink nodeLink : dependentNodes) nodeLink.updateExpression(index);
    }

    /**
     * Calculates backwards chain of gradient expressions.
     *
     * @param index specific data index.
     * @param outputGradient output gradient for procedure.
     * @return input node.
     * @throws MatrixException throws exception if calculation fails.
     */
    public Node calculateGradient(int index, Matrix outputGradient) throws MatrixException {
        getOutputNode().setGradient(index, outputGradient);
        updateGradientDependencies(index);
        for (Expression expression : gradientExpressions) expression.calculateGradient(index);
        return inputNode;
    }

    /**
     * Updates data of node dependencies for gradient calculation phase.
     *
     * @param index index to data for which dependencies are updates.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private void updateGradientDependencies(int index) throws MatrixException {
        for (NodeLink nodeLink : dependentNodes) nodeLink.updateGradient(index);
    }

    /**
     * Prints procedure.
     *
     */
    public void printProcedure() {
        Iterator iterator = expressions.iterator();
        while (iterator.hasNext()) {
            Expression expression = (Expression)iterator.next();
            expression.printExpression();
            if (iterator.hasNext()) System.out.print(" -> ");
        }
        System.out.println();
    }

}
