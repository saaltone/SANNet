/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package utils;

import java.io.Serializable;
import java.util.*;

/**
 * Class that builds computable procedures from chain of matrix operations including automated differentation (automatic gradient) as backward operation.<br>
 * Procedure factory records matrix operations in matrix instances having attachment to procedure factory.<br>
 *
 */
public class ProcedureFactory implements Serializable {

    private static final long serialVersionUID = -4961334078305757207L;

    /**
     * Procedure data to construct single procedure.
     *
     */
    private static class ProcedureData {

        /**
         * List of expressions for forward calculation.
         *
         */
        private final LinkedList<Expression> expressions = new LinkedList<>();

        /**
         * List of expressions for backward gradient calculation.
         *
         */
        private final LinkedList<Expression> gradientExpressions = new LinkedList<>();

        /**
         * Map for expressions for backward (gradient) calculation.<br>
         * This temporary map is used to build list of backward gradient expressions.<br>
         *
         */
        private final HashMap<Node, Expression> reverseExpressions = new HashMap<>();

        /**
         * Set of dependent output and input node pairs are links.
         *
         */
        private final HashSet<NodeLink> dependentNodes = new HashSet<>();

        /**
         * Input matrix.
         *
         */
        private Matrix inputMatrix;

        /**
         * Input node.
         *
         */
        private Node inputNode;

        /**
         * Output node.
         *
         */
        private Node outputNode;
    }

    /**
     * Node register.
     *
     */
    private final NodeRegister nodeRegister = new NodeRegister();

    /**
     * Current procedure ID.
     *
     */
    private int currentProcedureID = 0;

    /**
     * Current expression ID.
     *
     */
    private int currentExpressionID = 0;

    /**
     * Previous procedure data.
     *
     */
    private ProcedureData previousProcedureData = null;

    /**
     * Current procedure data.
     *
     */
    private ProcedureData currentProcedureData = null;

    /**
     * Set of registered matrices attached to procedure factory.
     *
     */
    private final HashSet<Matrix> registeredMatrixSet = new HashSet<>();

    /**
     * Map that links registered matrices to respective nodes.
     *
     */
    private final HashMap<Matrix, Node> registeredMatrixMap = new HashMap<>();

    /**
     * Default constructor for procedure factory.
     *
     */
    public ProcedureFactory() {
    }

    /**
     * Registers matrix and as needed attaches to this procedure factory.
     *
     * @param matrix matrix to be registered.
     * @param attachToProcedureFactory if true will attach matrix to procedure factory.
     */
    public void registerMatrix(Matrix matrix, boolean attachToProcedureFactory) {
        registeredMatrixSet.add(matrix);
        if (attachToProcedureFactory) matrix.setProcedureFactory(this);
    }

    /**
     * Registers set of matrices.
     *
     * @param matrices matrices to be registered.
     * @param attachToProcedureFactory if true will attach matrix to procedure factory.
     */
    public void registerMatrix(Set<Matrix> matrices, boolean attachToProcedureFactory) {
        for (Matrix matrix : matrices) registerMatrix(matrix, attachToProcedureFactory);
    }

    /**
     * Starts building of new procedure.
     *
     * @param inputMatrix input matrix.
     */
    public void newProcedure(Matrix inputMatrix) {
        currentExpressionID = 0;
        previousProcedureData = currentProcedureData;
        currentProcedureData = new ProcedureData();
        currentProcedureData.inputMatrix = inputMatrix;
        inputMatrix.setProcedureFactory(this);
    }

    /**
     * Ends building of current procedure.
     *
     * @param outputMatrix output matrix.
     * @throws MatrixException throws exception if setting of output matrix and node fails.
     * @return constructed current procedure.
     */
    public Procedure endProcedure(Matrix outputMatrix) throws MatrixException {
        if (!nodeRegister.contains(outputMatrix)) throw new MatrixException("Setting of output node failed. No node corresponding output matrix is found.");
        currentProcedureData.outputNode = nodeRegister.getNode(outputMatrix);
        defineGradientPath();
        Procedure thisProcedure = new Procedure(currentProcedureData.inputNode, currentProcedureData.outputNode, currentProcedureData.expressions, currentProcedureData.gradientExpressions, currentProcedureData.dependentNodes, registeredMatrixMap);
        if (currentProcedureID > 0) {
            analyzeDependencies();
            nodeRegister.removeProcedureFactory();
            return thisProcedure;
        }
        currentProcedureID++;
        return null;
    }

    /**
     * Analyzes and records dependencies between previous procedure and current procedure.
     *
     */
    private void analyzeDependencies() {
        currentProcedureData.dependentNodes.clear();
        if (previousProcedureData.expressions.size() != currentProcedureData.expressions.size()) return;
        for (int expressionID = 0; expressionID < currentProcedureData.expressions.size(); expressionID++) {
            analyzeDependency(previousProcedureData.expressions.get(expressionID).getArg1(), currentProcedureData.expressions.get(expressionID).getArg1());
            analyzeDependency(previousProcedureData.expressions.get(expressionID).getArg2(), currentProcedureData.expressions.get(expressionID).getArg2());
        }
    }

    /**
     * Analyzes dependency between previous (output) and current (input) arg node.<br>
     * Records dependencies to current procedure data as node links.<br>
     *
     * @param previousArgNode previous arg node.
     * @param currentArgNode current arg node.
     */
    private void analyzeDependency(Node previousArgNode, Node currentArgNode) {
        int argExpressionID = nodeRegister.getExpressionID(previousArgNode);
        int otherArgExpressionID = nodeRegister.getExpressionID(currentArgNode);
        if (argExpressionID != otherArgExpressionID) {
            Node previousResultNode = previousProcedureData.expressions.get(argExpressionID).getResult();
            currentProcedureData.dependentNodes.add(new NodeLink(previousResultNode, currentArgNode));
        }
    }

    /**
     * Records bi (two) argument expression to procedure factory.
     *
     * @param arg1 first argument of expression.
     * @param arg2 second argument of expression.
     * @param result result of expression.
     * @param type type of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void addExpression(Matrix arg1, Matrix arg2, Matrix result, Expression.Type type) throws MatrixException {
        Node node1 = defineNode(arg1, false);
        Node node2 = defineNode(arg2, false);
        Node resultNode = defineNode(result, true);
        Expression expression = new Expression(currentExpressionID++, node1, node2, resultNode, type);
        currentProcedureData.expressions.add(expression);
        currentProcedureData.reverseExpressions.put(resultNode, expression);
    }

    /**
     * Records uni (single) argument expression to procedure factory.
     *
     * @param arg1 first argument of expression.
     * @param result result of expression.
     * @param uniFunction UniFunction of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void addExpression(Matrix arg1, Matrix result, UniFunction uniFunction) throws MatrixException {
        Node node1 = defineNode(arg1, false);
        Node resultNode = defineNode(result, true);
        Expression expression = new Expression(currentExpressionID++, node1, resultNode, uniFunction);
        currentProcedureData.expressions.add(expression);
        currentProcedureData.reverseExpressions.put(resultNode, expression);
    }

    /**
     * Records bi (two) argument expression to procedure factory.
     *
     * @param arg1 first argument of expression.
     * @param arg2 second argument of expression.
     * @param result result of expression.
     * @param biFunction BiFunction of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void addExpression(Matrix arg1, Matrix arg2, Matrix result, BiFunction biFunction) throws MatrixException {
        Node node1 = defineNode(arg1, false);
        Node node2 = defineNode(arg2, false);
        Node resultNode = defineNode(result, true);
        Expression expression = new Expression(currentExpressionID++, node1, node2, resultNode, biFunction);
        currentProcedureData.expressions.add(expression);
        currentProcedureData.reverseExpressions.put(resultNode, expression);
    }

    /**
     * Defines node for procedure. Sets input and result nodes as non-constant nodes.
     *
     * @param matrix matrix for node.
     * @param resultNode if true node is result node.
     * @return defined node.
     */
    private Node defineNode(Matrix matrix, boolean resultNode) throws MatrixException {
        boolean isConstantNode = !(matrix == currentProcedureData.inputMatrix || resultNode);
        Node node = nodeRegister.defineNode(matrix, isConstantNode, currentProcedureID, currentExpressionID);
        if (matrix == currentProcedureData.inputMatrix) currentProcedureData.inputNode = node;
        if (registeredMatrixSet.contains(matrix)) registeredMatrixMap.put(matrix, node);
        return node;
    }

    /**
     * Defines backward gradient calculation path for expressions.<br>
     * Records gradient path to current procedure data.<br>
     *
     */
    private void defineGradientPath() {
        Stack<Node> resultNodes = new Stack<>();
        resultNodes.push(currentProcedureData.outputNode);
        while (!resultNodes.empty()) {
            Expression expression = currentProcedureData.reverseExpressions.get(resultNodes.pop());
            if (expression != null && !currentProcedureData.gradientExpressions.contains(expression)) {
                currentProcedureData.gradientExpressions.add(expression);
                Node arg1 = expression.getArg1();
                if (arg1 != null) resultNodes.push(arg1);
                Node arg2 = expression.getArg2();
                if (arg2 != null) resultNodes.push(arg2);
            }
        }
    }

}
