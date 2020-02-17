/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package utils.procedure;

import core.normalization.Normalization;
import utils.Sample;
import utils.matrix.BinaryFunction;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.UnaryFunction;

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
        private final LinkedList<AbstractExpression> expressions = new LinkedList<>();

        /**
         * List of expressions for backward gradient calculation.
         *
         */
        private final LinkedList<AbstractExpression> gradientExpressions = new LinkedList<>();

        /**
         * Map for expressions for backward (gradient) calculation.<br>
         * This temporary map is used to build list of backward gradient expressions.<br>
         *
         */
        private final HashMap<Node, AbstractExpression> reverseExpressions = new HashMap<>();

        /**
         * Set of dependent output and input node pairs are links.
         *
         */
        private final HashSet<NodeLink> dependentNodes = new HashSet<>();

        /**
         * Input sample.
         *
         */
        private Sample inputSample;

        /**
         * Input node.
         *
         */
        private final HashMap<Integer, Node> inputNodes = new HashMap<>();

        /**
         * Output node.
         *
         */
        private final HashMap<Integer, Node> outputNodes = new HashMap<>();
    }

    /**
     * Node register.
     *
     */
    private final NodeRegister nodeRegister = new NodeRegister();

    /**
     * Current expression ID.
     *
     */
    private int currentExpressionID = 0;

    /**
     * List of procedure data entries.
     *
     */
    private final LinkedList<ProcedureData> procedureDataList = new LinkedList<>();

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
     * @param inputSample input sample.
     */
    public void newProcedure(Sample inputSample) {
        currentExpressionID = 0;
        procedureDataList.add(currentProcedureData = new ProcedureData());
        currentProcedureData.inputSample = inputSample;
        for (Matrix inputMatrix : inputSample.values()) inputMatrix.setProcedureFactory(this);
    }

    /**
     * Ends building of current procedure.
     *
     * @param outputMatrices output matrices.
     * @throws MatrixException throws exception if setting of output matrix and node fails.
     * @return constructed current procedure.
     */
    public LinkedList<Procedure> endProcedure(Sample outputMatrices) throws MatrixException {
        for (Integer index : outputMatrices.keySet()) {
            if (!nodeRegister.contains(outputMatrices.get(index))) throw new MatrixException("Setting of output node failed. No node corresponding output matrix is found.");
            currentProcedureData.outputNodes.put(index, nodeRegister.getNode(outputMatrices.get(index)));
        }
        defineGradientPath();
        if (analyzeDependencies()) {
            procedureDataList.remove(0);
            nodeRegister.removeProcedureFactory();
            LinkedList<Procedure> procedureList = new LinkedList<>();
            for (ProcedureData procedureData : procedureDataList) {
                procedureList.add(new Procedure(procedureData.inputNodes, procedureData.outputNodes, procedureData.expressions, procedureData.gradientExpressions, procedureData.dependentNodes, registeredMatrixMap));
            }
            return procedureList;
        }
        return null;
    }

    /**
     * Analyzes and records dependencies between previous procedure and current procedure.
     *
     * @return returns true if procedures are dependent otherwise returns false.
     */
    private boolean analyzeDependencies() {
        if (procedureDataList.size() < 2) return false;
        currentProcedureData.dependentNodes.clear();
        ProcedureData previousProcedureData = procedureDataList.get(0);
        if (previousProcedureData.expressions.size() != currentProcedureData.expressions.size()) return false;
        boolean areIndependent = true;
        for (int expressionID = 0; expressionID < currentProcedureData.expressions.size(); expressionID++) {
            AbstractExpression previousExpression = previousProcedureData.expressions.get(expressionID);
            AbstractExpression currentExpression = currentProcedureData.expressions.get(expressionID);
            updateNodeLink(previousExpression.getArg1(), currentExpression.getArg1());
            updateNodeLink(previousExpression.getArg2(), currentExpression.getArg2());
            int dependencyArg1 = analyzeDependency(previousExpression.getArg1(), currentExpression.getArg1());
            int dependencyArg2 = analyzeDependency(previousExpression.getArg2(), currentExpression.getArg2());
            if (dependencyArg1 == -1 || dependencyArg2 == -1) areIndependent = false;
        }
        return areIndependent;
    }

    /**
     * Updates dependencies between previous (output) and current (input) arg node.<br>
     * Records dependencies to current procedure data as node links.<br>
     *
     * @param previousArgNode previous arg node.
     * @param currentArgNode current arg node.
     */
    private void updateNodeLink(Node previousArgNode, Node currentArgNode) {
        int argExpressionID = nodeRegister.getExpressionID(previousArgNode);
        int otherArgExpressionID = nodeRegister.getExpressionID(currentArgNode);
        if (argExpressionID != otherArgExpressionID) {
            ProcedureData previousProcedureData = procedureDataList.get(0);
            Node previousResultNode = previousProcedureData.expressions.get(argExpressionID).getResult();
            currentProcedureData.dependentNodes.add(new NodeLink(previousResultNode, currentArgNode));
        }
    }

    /**
     * Analyzes dependencies between nodes. Nodes are independent if nodes are constant and same.
     *
     * @param previousArgNode previous arg node.
     * @param currentArgNode current arg node.
     * @return returns zero if nodes are non-constant, 1 if they are independent and same otherwise -1.
     */
    private int analyzeDependency(Node previousArgNode, Node currentArgNode) {
        if (previousArgNode == null || currentArgNode == null) return 0;
        if (previousArgNode.isConstantNode() || currentArgNode.isConstantNode()) return 0;
        else return previousArgNode != currentArgNode ? -1 : 1;
    }

    /**
     * Records add expression to procedure factory.
     *
     * @param arg1 first argument of expression.
     * @param arg2 second argument of expression.
     * @param result result of expression.
     * @param normalizers normalizers.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createAddExpression(Matrix arg1, Matrix arg2, Matrix result, HashSet<Normalization> normalizers) throws MatrixException {
        Node node1 = defineNode(arg1, false, normalizers);
        Node node2 = defineNode(arg2, false, normalizers);
        Node resultNode = defineNode(result, true, normalizers);
        AddExpression expression = new AddExpression(currentExpressionID++, node1, node2, resultNode);
        currentProcedureData.expressions.add(expression);
        currentProcedureData.reverseExpressions.put(resultNode, expression);
    }

    /**
     * Records subtract expression to procedure factory.
     *
     * @param arg1 first argument of expression.
     * @param arg2 second argument of expression.
     * @param result result of expression.
     * @param normalizers normalizers.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createSubtractExpression(Matrix arg1, Matrix arg2, Matrix result, HashSet<Normalization> normalizers) throws MatrixException {
        Node node1 = defineNode(arg1, false, normalizers);
        Node node2 = defineNode(arg2, false, normalizers);
        Node resultNode = defineNode(result, true, normalizers);
        SubtractExpression expression = new SubtractExpression(currentExpressionID++, node1, node2, resultNode);
        currentProcedureData.expressions.add(expression);
        currentProcedureData.reverseExpressions.put(resultNode, expression);
    }

    /**
     * Records dot expression to procedure factory.
     *
     * @param arg1 first argument of expression.
     * @param arg2 second argument of expression.
     * @param result result of expression.
     * @param normalizers normalizers.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createDotExpression(Matrix arg1, Matrix arg2, Matrix result, HashSet<Normalization> normalizers) throws MatrixException {
        Node node1 = defineNode(arg1, false, normalizers);
        Node node2 = defineNode(arg2, false, normalizers);
        Node resultNode = defineNode(result, true, normalizers);
        DotExpression expression = new DotExpression(currentExpressionID++, node1, node2, resultNode);
        currentProcedureData.expressions.add(expression);
        currentProcedureData.reverseExpressions.put(resultNode, expression);
    }

    /**
     * Records multiply expression to procedure factory.
     *
     * @param arg1 first argument of expression.
     * @param arg2 second argument of expression.
     * @param result result of expression.
     * @param normalizers normalizers.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createMultiplyExpression(Matrix arg1, Matrix arg2, Matrix result, HashSet<Normalization> normalizers) throws MatrixException {
        Node node1 = defineNode(arg1, false, normalizers);
        Node node2 = defineNode(arg2, false, normalizers);
        Node resultNode = defineNode(result, true, normalizers);
        MultiplyExpression expression = new MultiplyExpression(currentExpressionID++, node1, node2, resultNode);
        currentProcedureData.expressions.add(expression);
        currentProcedureData.reverseExpressions.put(resultNode, expression);
    }

    /**
     * Records divide expression to procedure factory.
     *
     * @param arg1 first argument of expression.
     * @param arg2 second argument of expression.
     * @param result result of expression.
     * @param normalizers normalizers.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createDivideExpression(Matrix arg1, Matrix arg2, Matrix result, HashSet<Normalization> normalizers) throws MatrixException {
        Node node1 = defineNode(arg1, false, normalizers);
        Node node2 = defineNode(arg2, false, normalizers);
        Node resultNode = defineNode(result, true, normalizers);
        DivideExpression expression = new DivideExpression(currentExpressionID++, node1, node2, resultNode);
        currentProcedureData.expressions.add(expression);
        currentProcedureData.reverseExpressions.put(resultNode, expression);
    }

    /**
     * Records convolve expression to procedure factory.
     *
     * @param arg1 first argument of expression.
     * @param arg2 second argument of expression.
     * @param result result of expression.
     * @param stride stride of convolution operation.
     * @param normalizers normalizers.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createConvolveExpression(Matrix arg1, Matrix arg2, Matrix result, int stride, HashSet<Normalization> normalizers) throws MatrixException {
        Node node1 = defineNode(arg1, false, normalizers);
        Node node2 = defineNode(arg2, false, normalizers);
        Node resultNode = defineNode(result, true, normalizers);
        ConvolveExpression expression = new ConvolveExpression(currentExpressionID++, node1, node2, resultNode, stride);
        currentProcedureData.expressions.add(expression);
        currentProcedureData.reverseExpressions.put(resultNode, expression);
    }

    /**
     * Records crosscorrelate expression to procedure factory.
     *
     * @param arg1 first argument of expression.
     * @param arg2 second argument of expression.
     * @param result result of expression.
     * @param stride stride for operation.
     * @param normalizers normalizers.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createCrosscorrelateExpression(Matrix arg1, Matrix arg2, Matrix result, int stride, HashSet<Normalization> normalizers) throws MatrixException {
        Node node1 = defineNode(arg1, false, normalizers);
        Node node2 = defineNode(arg2, false, normalizers);
        Node resultNode = defineNode(result, true, normalizers);
        CrosscorrelateExpression expression = new CrosscorrelateExpression(currentExpressionID++, node1, node2, resultNode, stride);
        currentProcedureData.expressions.add(expression);
        currentProcedureData.reverseExpressions.put(resultNode, expression);
    }

    /**
     * Records max pool expression to procedure factory.
     *
     * @param arg1 first argument of expression.
     * @param result result of expression.
     * @param stride stride for operation.
     * @param poolSize pool size for operation.
     * @param normalizers normalizers.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createMaxPoolExpression(Matrix arg1, Matrix result, int stride, int poolSize, HashSet<Normalization> normalizers) throws MatrixException {
        Node node1 = defineNode(arg1, false, normalizers);
        Node resultNode = defineNode(result, true, normalizers);
        MaxPoolExpression expression = new MaxPoolExpression(currentExpressionID++, node1, resultNode, stride, poolSize);
        currentProcedureData.expressions.add(expression);
        currentProcedureData.reverseExpressions.put(resultNode, expression);
    }

    /**
     * Records average pool expression to procedure factory.
     *
     * @param arg1 first argument of expression.
     * @param result result of expression.
     * @param stride stride for operation.
     * @param poolSize pool size for operation.
     * @param normalizers normalizers.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createAveragePoolExpression(Matrix arg1, Matrix result, int stride, int poolSize, HashSet<Normalization> normalizers) throws MatrixException {
        Node node1 = defineNode(arg1, false, normalizers);
        Node resultNode = defineNode(result, true, normalizers);
        AveragePoolExpression expression = new AveragePoolExpression(currentExpressionID++, node1, resultNode, stride, poolSize);
        currentProcedureData.expressions.add(expression);
        currentProcedureData.reverseExpressions.put(resultNode, expression);
    }

    /**
     * Records uni (single) argument expression to procedure factory.
     *
     * @param arg1 first argument of expression.
     * @param result result of expression.
     * @param unaryFunction UnaryFunction of expression.
     * @param normalizers normalizers.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createUnaryFunctionExpression(Matrix arg1, Matrix result, UnaryFunction unaryFunction, HashSet<Normalization> normalizers) throws MatrixException {
        Node node1 = defineNode(arg1, false, normalizers);
        Node resultNode = defineNode(result, true, normalizers);
        UnaryFunctionExpression expression = new UnaryFunctionExpression(currentExpressionID++, node1, resultNode, unaryFunction);
        currentProcedureData.expressions.add(expression);
        currentProcedureData.reverseExpressions.put(resultNode, expression);
    }

    /**
     * Records binary (two) argument expression to procedure factory.
     *
     * @param arg1 first argument of expression.
     * @param arg2 second argument of expression.
     * @param result result of expression.
     * @param binaryFunction BinaryFunction of expression.
     * @param normalizers normalizers.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createBinaryFunctionExpression(Matrix arg1, Matrix arg2, Matrix result, BinaryFunction binaryFunction, HashSet<Normalization> normalizers) throws MatrixException {
        Node node1 = defineNode(arg1, false, normalizers);
        Node node2 = defineNode(arg2, false, normalizers);
        Node resultNode = defineNode(result, true, normalizers);
        BinaryFunctionExpression expression = new BinaryFunctionExpression(currentExpressionID++, node1, node2, resultNode, binaryFunction);
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
    private Node defineNode(Matrix matrix, boolean resultNode, HashSet<Normalization> normalizers) throws MatrixException {
        boolean isConstantNode = !(currentProcedureData.inputSample.contains(matrix) || resultNode);
        Node node = nodeRegister.defineNode(matrix, isConstantNode, resultNode, normalizers, procedureDataList.size() - 1, currentExpressionID);
        for (Integer index : currentProcedureData.inputSample.keySet()) {
            if (currentProcedureData.inputSample.get(index) == matrix) currentProcedureData.inputNodes.put(index, node);
        }
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
        for (Node outputNode : currentProcedureData.outputNodes.values()) resultNodes.push(outputNode);
        while (!resultNodes.empty()) {
            AbstractExpression expression = currentProcedureData.reverseExpressions.get(resultNodes.pop());
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
