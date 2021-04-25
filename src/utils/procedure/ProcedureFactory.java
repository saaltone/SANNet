/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.procedure;

import utils.DynamicParamException;
import utils.matrix.*;
import utils.procedure.expression.*;
import utils.procedure.node.Node;
import utils.procedure.node.NodeRegister;

import java.io.Serial;
import java.io.Serializable;
import java.util.*;

/**
 * Class that builds computable procedures from chain of matrix operations including automated differentiation (automatic gradient) as backward operation.<br>
 * Procedure factory records matrix operations in matrix instances having attachment to procedure factory.<br>
 *
 */
public class ProcedureFactory implements Serializable {

    @Serial
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
        private final HashMap<Node, AbstractExpression> reverseExpressionMap = new HashMap<>();

        /**
         * if true procedure has dependent nodes.
         *
         */
        private boolean hasDependentNodes = false;

        /**
         * Input sample.
         *
         */
        private MMatrix inputSample;

        /**
         * Input nodes.
         *
         */
        private final HashMap<Integer, Node> inputNodes = new HashMap<>();

        /**
         * Output nodes.
         *
         */
        private final HashMap<Integer, Node> outputNodes = new HashMap<>();

        /**
         * Nodes of procedure.
         *
         */
        private final HashSet<Node> nodes = new HashSet<>();

    }

    /**
     * Reference to node register.
     *
     */
    private final NodeRegister nodeRegister = new NodeRegister();

    /**
     * Current expression ID.
     *
     */
    private transient int currentExpressionID = 0;

    /**
     * Current procedure data.
     *
     */
    private transient ProcedureData currentProcedureData = null;

    /**
     * Unique expression lock to reserve procedure factory.
     *
     */
    private double expressionLock = 0;

    /**
     * If true silently continues creation of existing procedure even new one is attempted.
     *
     */
    private boolean silentlyContinue = false;

    /**
     * Random function.
     *
     */
    private final Random random = new Random();

    /**
     * Default constructor for procedure factory.
     *
     */
    public ProcedureFactory() {
    }

    /**
     * Returns procedure
     *
     * @param forwardProcedure reference to class that defines forward procedure.
     * @param weights weights to be registered.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @return resulting procedure.
     */
    public Procedure getProcedure(ForwardProcedure forwardProcedure, HashSet<Matrix> weights) throws MatrixException, DynamicParamException {
        registerConstantMatrices(weights);

        ProcedureData previousProcedureData = new ProcedureData();
        newProcedure(previousProcedureData, forwardProcedure.getInputMatrices(true));
        endProcedure(previousProcedureData, forwardProcedure.getForwardProcedure());

        ProcedureData nextProcedureData = new ProcedureData();
        newProcedure(nextProcedureData, forwardProcedure.getInputMatrices(false));
        endProcedure(nextProcedureData, forwardProcedure.getForwardProcedure());

        updateDependencies(previousProcedureData, nextProcedureData);

        nodeRegister.removeProcedureFactory();

        AbstractExpression previousExpression = null;
        for (AbstractExpression expression : nextProcedureData.expressions) {
            if (previousExpression != null) previousExpression.setNextExpression(expression);
            previousExpression = expression;
        }
        previousExpression = null;
        for (AbstractExpression expression : nextProcedureData.gradientExpressions) {
            if (previousExpression != null) previousExpression.setPreviousExpression(expression);
            previousExpression = expression;
        }

        return new Procedure(nextProcedureData.inputNodes, nextProcedureData.outputNodes, nextProcedureData.nodes, nextProcedureData.expressions.get(0), nextProcedureData.gradientExpressions.get(0), nextProcedureData.hasDependentNodes);
    }

    /**
     * Registers set of constant matrices.
     *
     * @param matrices matrices to be registered.
     */
    private void registerConstantMatrices(Set<Matrix> matrices) {
        if (matrices == null) return;
        for (Matrix matrix : matrices) matrix.setProcedureFactory(this);
    }

    /**
     * Starts building new procedure.
     *
     * @param inputSample input sample.
     */
    private void newProcedure(ProcedureData procedureData, MMatrix inputSample) {
        procedureData.inputSample = inputSample;
        inputSample.setProcedureFactory(this);
        for (Matrix matrix : inputSample.get().values()) matrix.setProcedureFactory(this);
        currentExpressionID = 0;
        currentProcedureData = procedureData;
    }

    /**
     * Finalizes building current procedure.
     *
     * @param outputMatrices output matrices.
     * @throws MatrixException throws exception if setting of output matrix and node fails.
     */
    private void endProcedure(ProcedureData procedureData, MMatrix outputMatrices) throws MatrixException {
        if (!nodeRegister.contains(outputMatrices)) {
            for (Integer index : outputMatrices.keySet()) {
                if (!nodeRegister.contains(outputMatrices.get(index))) throw new MatrixException("Setting of output node failed. No node corresponding output matrix is found.");
                procedureData.outputNodes.put(index, nodeRegister.getNode(outputMatrices.get(index)));
            }
        } else procedureData.outputNodes.put(0, nodeRegister.getNode(outputMatrices));
        defineGradientPath(procedureData);
        currentProcedureData = null;
    }

    /**
     * Defines backward gradient calculation path for expressions.<br>
     * Records gradient path to current procedure data.<br>
     *
     */
    private void defineGradientPath(ProcedureData procedureData) {
        Stack<Node> resultNodes = new Stack<>();
        for (Node outputNode : procedureData.outputNodes.values()) resultNodes.push(outputNode);
        while (!resultNodes.empty()) {
            AbstractExpression expression = procedureData.reverseExpressionMap.get(resultNodes.pop());
            if (expression != null && !procedureData.gradientExpressions.contains(expression)) {
                procedureData.gradientExpressions.add(expression);
                Node argument1 = expression.getArgument1();
                if (argument1 != null) resultNodes.push(argument1);
                Node argument2 = expression.getArgument2();
                if (argument2 != null) resultNodes.push(argument2);
            }
        }
    }

    /**
     * Analyzes and records dependencies between previous procedure and current procedure.
     *
     */
    private void updateDependencies(ProcedureData previousProcedureData, ProcedureData nextProcedureData) {
        for (int expressionID = 0; expressionID < nextProcedureData.expressions.size() - 1; expressionID++) {
            updateNodeLink(previousProcedureData, nextProcedureData, previousProcedureData.expressions.get(expressionID).getArgument1(), nextProcedureData.expressions.get(expressionID).getArgument1());
            updateNodeLink(previousProcedureData, nextProcedureData, previousProcedureData.expressions.get(expressionID).getArgument2(), nextProcedureData.expressions.get(expressionID).getArgument2());
        }
    }

    /**
     * Updates dependencies between previous (output) and current (input) node.<br>
     * Records dependencies to respective nodes.<br>
     *
     * @param previousArgumentNode previous node.
     * @param nextArgumentNode current node.
     */
    private void updateNodeLink(ProcedureData previousProcedureData, ProcedureData nextProcedureData, Node previousArgumentNode, Node nextArgumentNode) {
        int previousArgumentExpressionID = nodeRegister.getExpressionID(previousArgumentNode);
        int nextArgumentExpressionID = nodeRegister.getExpressionID(nextArgumentNode);
        if (previousArgumentExpressionID != nextArgumentExpressionID) {
            Node previousResultNode = previousProcedureData.expressions.get(previousArgumentExpressionID).getResult();
            nextProcedureData.hasDependentNodes = true;
            nextArgumentNode.setFromNode(previousResultNode);
            previousResultNode.setToNode(nextArgumentNode);
        }
    }

    /**
     * Defines node for procedure. Sets input and result nodes as non-constant nodes.
     *
     * @param matrix matrix for node.
     * @param resultNode if true node is result node.
     * @return defined node.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Node defineNode(MMatrix matrix, boolean resultNode) throws MatrixException {
        boolean isConstantNode = !(currentProcedureData.inputSample == matrix || resultNode);
        Node node = nodeRegister.defineNode(matrix, isConstantNode, currentExpressionID);
        for (Integer index : currentProcedureData.inputSample.keySet()) {
            if (currentProcedureData.inputSample == matrix) currentProcedureData.inputNodes.put(index, node);
        }
        currentProcedureData.nodes.add(node);
        return node;
    }

    /**
     * Defines node for procedure. Sets input and result nodes as non-constant nodes.
     *
     * @param matrix matrix for node.
     * @param resultNode if true node is result node.
     * @return defined node.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Node defineNode(Matrix matrix, boolean resultNode) throws MatrixException {
        boolean isConstantNode = !(currentProcedureData.inputSample.contains(matrix) || resultNode);
        Node node = nodeRegister.defineNode(matrix, isConstantNode, currentExpressionID);
        for (Integer index : currentProcedureData.inputSample.keySet()) {
            if (currentProcedureData.inputSample.get(index) == matrix) currentProcedureData.inputNodes.put(index, node);
        }
        currentProcedureData.nodes.add(node);
        return node;
    }

    /**
     * Starts new expression and reserves procedure factory with expression lock.
     *
     * @param originator originator of procedure request.
     * @throws MatrixException throws exception if procedure factory is already reserved by another request
     * @return unique expression lock key.
     */
    public double startExpression(Object originator) throws MatrixException {
        return startExpression(originator, true);
    }

    /**
     * Starts new expression and reserves procedure factory with expression lock.
     *
     * @param originator originator of procedure request.
     * @param silentlyContinue if true silently returns and continues creation of existing procedure without throwing exception.
     * @throws MatrixException throws exception if procedure factory is already reserved by another request
     * @return unique expression lock key.
     */
    public double startExpression(Object originator, boolean silentlyContinue) throws MatrixException {
        if (expressionLock != 0) {
            if (silentlyContinue) return 0;
            else throw new MatrixException("Procedure factory is reserved by: " + originator);
        }
        this.silentlyContinue = silentlyContinue;
        expressionLock = random.nextDouble();
        return expressionLock;
    }

    /**
     * Internally finishes creation of expression and frees expression lock.
     *
     */
    private void finishExpression() {
        expressionLock = 0;
    }

    /**
     * Checks if there is ongoing procedure. Silently continues with existing expression if flag is set otherwise throws exception.
     *
     * @param expressionLock unique expression lock key.
     * @param originator originator of procedure request.
     * @return returns true is existing expression creation is ongoing otherwise false.
     * @throws MatrixException throws exception if procedure factory is already reserved by another request
     */
    public boolean checkOngoingExpression(double expressionLock, Object originator) throws MatrixException {
        if (this.expressionLock != expressionLock) {
            if (silentlyContinue) return true;
            else throw new MatrixException("Procedure factory is reserved by: " + originator);
        }
        else return false;
    }

    /**
     * Records add expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param argument2 second argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createAddExpression(double expressionLock, Matrix argument1, Matrix argument2, Matrix result) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node node2 = defineNode(argument2, false);
        Node resultNode = defineNode(result, true);
        AddExpression expression = new AddExpression(currentExpressionID++, node1, node2, resultNode);
        storeExpression(expression, resultNode);
    }

    /**
     * Records add expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param argument2 second argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createAddExpression(double expressionLock, MMatrix argument1, Matrix argument2, MMatrix result) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node node2 = defineNode(argument2, false);
        Node resultNode = defineNode(result, true);
        AddExpression expression = new AddExpression(currentExpressionID++, node1, node2, resultNode);
        storeExpression(expression, resultNode);
    }

    /**
     * Records add expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param argument2 second argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createAddExpression(double expressionLock, MMatrix argument1, MMatrix argument2, MMatrix result) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node node2 = defineNode(argument2, false);
        Node resultNode = defineNode(result, true);
        AddExpression expression = new AddExpression(currentExpressionID++, node1, node2, resultNode);
        storeExpression(expression, resultNode);
    }

    /**
     * Records subtract expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param argument2 second argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createSubtractExpression(double expressionLock, Matrix argument1, Matrix argument2, Matrix result) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node node2 = defineNode(argument2, false);
        Node resultNode = defineNode(result, true);
        SubtractExpression expression = new SubtractExpression(currentExpressionID++, node1, node2, resultNode);
        storeExpression(expression, resultNode);
    }

    /**
     * Records subtract expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param argument2 second argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createSubtractExpression(double expressionLock, MMatrix argument1, Matrix argument2, MMatrix result) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node node2 = defineNode(argument2, false);
        Node resultNode = defineNode(result, true);
        SubtractExpression expression = new SubtractExpression(currentExpressionID++, node1, node2, resultNode);
        storeExpression(expression, resultNode);
    }

    /**
     * Records subtract expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param argument2 second argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createSubtractExpression(double expressionLock, MMatrix argument1, MMatrix argument2, MMatrix result) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node node2 = defineNode(argument2, false);
        Node resultNode = defineNode(result, true);
        SubtractExpression expression = new SubtractExpression(currentExpressionID++, node1, node2, resultNode);
        storeExpression(expression, resultNode);
    }

    /**
     * Records dot expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param argument2 second argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createDotExpression(double expressionLock, Matrix argument1, Matrix argument2, Matrix result) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node node2 = defineNode(argument2, false);
        Node resultNode = defineNode(result, true);
        DotExpression expression = new DotExpression(currentExpressionID++, node1, node2, resultNode);
        storeExpression(expression, resultNode);
    }

    /**
     * Records dot expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param argument2 second argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createDotExpression(double expressionLock, MMatrix argument1, Matrix argument2, MMatrix result) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node node2 = defineNode(argument2, false);
        Node resultNode = defineNode(result, true);
        DotExpression expression = new DotExpression(currentExpressionID++, node1, node2, resultNode);
        storeExpression(expression, resultNode);
    }

    /**
     * Records dot expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param argument2 second argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createDotExpression(double expressionLock, MMatrix argument1, MMatrix argument2, MMatrix result) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node node2 = defineNode(argument2, false);
        Node resultNode = defineNode(result, true);
        DotExpression expression = new DotExpression(currentExpressionID++, node1, node2, resultNode);
        storeExpression(expression, resultNode);
    }

    /**
     * Records multiply expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param argument2 second argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createMultiplyExpression(double expressionLock, Matrix argument1, Matrix argument2, Matrix result) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node node2 = defineNode(argument2, false);
        Node resultNode = defineNode(result, true);
        MultiplyExpression expression = new MultiplyExpression(currentExpressionID++, node1, node2, resultNode);
        storeExpression(expression, resultNode);
    }

    /**
     * Records multiply expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param argument2 second argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createMultiplyExpression(double expressionLock, MMatrix argument1, Matrix argument2, MMatrix result) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node node2 = defineNode(argument2, false);
        Node resultNode = defineNode(result, true);
        MultiplyExpression expression = new MultiplyExpression(currentExpressionID++, node1, node2, resultNode);
        storeExpression(expression, resultNode);
    }

    /**
     * Records multiply expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param argument2 second argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createMultiplyExpression(double expressionLock, MMatrix argument1, MMatrix argument2, MMatrix result) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node node2 = defineNode(argument2, false);
        Node resultNode = defineNode(result, true);
        MultiplyExpression expression = new MultiplyExpression(currentExpressionID++, node1, node2, resultNode);
        storeExpression(expression, resultNode);
    }

    /**
     * Records divide expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param argument2 second argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createDivideExpression(double expressionLock, Matrix argument1, Matrix argument2, Matrix result) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node node2 = defineNode(argument2, false);
        Node resultNode = defineNode(result, true);
        DivideExpression expression = new DivideExpression(currentExpressionID++, node1, node2, resultNode);
        storeExpression(expression, resultNode);
    }

    /**
     * Records divide expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param argument2 second argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createDivideExpression(double expressionLock, MMatrix argument1, Matrix argument2, MMatrix result) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node node2 = defineNode(argument2, false);
        Node resultNode = defineNode(result, true);
        DivideExpression expression = new DivideExpression(currentExpressionID++, node1, node2, resultNode);
        storeExpression(expression, resultNode);
    }

    /**
     * Records divide expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param argument2 second argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createDivideExpression(double expressionLock, MMatrix argument1, MMatrix argument2, MMatrix result) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node node2 = defineNode(argument2, false);
        Node resultNode = defineNode(result, true);
        DivideExpression expression = new DivideExpression(currentExpressionID++, node1, node2, resultNode);
        storeExpression(expression, resultNode);
    }

    /**
     * Records convolve expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param argument2 second argument of expression.
     * @param result result of expression.
     * @param stride stride of convolution operation.
     * @param dilation dilation step size.
     * @param filterRowSize filter row size.
     * @param filterColumnSize filter column size.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createConvolveExpression(double expressionLock, Matrix argument1, Matrix argument2, Matrix result, int stride, int dilation, int filterRowSize, int filterColumnSize) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node node2 = defineNode(argument2, false);
        Node resultNode = defineNode(result, true);
        ConvolveExpression expression = new ConvolveExpression(currentExpressionID++, node1, node2, resultNode, stride, dilation, filterRowSize, filterColumnSize);
        storeExpression(expression, resultNode);
    }

    /**
     * Records crosscorrelate expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param argument2 second argument of expression.
     * @param result result of expression.
     * @param stride stride for operation.
     * @param dilation dilation step size.
     * @param filterRowSize filter row size.
     * @param filterColumnSize filter column size.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createCrosscorrelateExpression(double expressionLock, Matrix argument1, Matrix argument2, Matrix result, int stride, int dilation, int filterRowSize, int filterColumnSize) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node node2 = defineNode(argument2, false);
        Node resultNode = defineNode(result, true);
        CrosscorrelateExpression expression = new CrosscorrelateExpression(currentExpressionID++, node1, node2, resultNode, stride, dilation, filterRowSize, filterColumnSize);
        storeExpression(expression, resultNode);
    }

    /**
     * Records max pool expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @param stride stride for operation.
     * @param filterRowSize filter row size for operation.
     * @param filterColumnSize filter column size for operation.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createMaxPoolExpression(double expressionLock, Matrix argument1, Matrix result, int stride, int filterRowSize, int filterColumnSize) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node resultNode = defineNode(result, true);
        MaxPoolExpression expression = new MaxPoolExpression(currentExpressionID++, node1, resultNode, stride, filterRowSize, filterColumnSize);
        storeExpression(expression, resultNode);
    }

    /**
     * Records average pool expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @param stride stride for operation.
     * @param filterRowSize filter row size for operation.
     * @param filterColumnSize filter column size for operation.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createAveragePoolExpression(double expressionLock, Matrix argument1, Matrix result, int stride, int filterRowSize, int filterColumnSize) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node resultNode = defineNode(result, true);
        AveragePoolExpression expression = new AveragePoolExpression(currentExpressionID++, node1, resultNode, stride, filterRowSize, filterColumnSize);
        storeExpression(expression, resultNode);
    }

    /**
     * Records sum expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createSumExpression(double expressionLock, Matrix argument1, Matrix result) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node resultNode = defineNode(result, true);
        SumExpression expression = new SumExpression(currentExpressionID++, node1, resultNode, false);
        storeExpression(expression, resultNode);
    }

    /**
     * Records sum expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createSumExpression(double expressionLock, MMatrix argument1, Matrix result) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node resultNode = defineNode(result, true);
        SumExpression expression = new SumExpression(currentExpressionID++, node1, resultNode, true);
        storeExpression(expression, resultNode);
    }

    /**
     * Records mean expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createMeanExpression(double expressionLock, Matrix argument1, Matrix result) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node resultNode = defineNode(result, true);
        MeanExpression expression = new MeanExpression(currentExpressionID++, node1, resultNode, false);
        storeExpression(expression, resultNode);
    }

    /**
     * Records mean expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createMeanExpression(double expressionLock, MMatrix argument1, Matrix result) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node resultNode = defineNode(result, true);
        MeanExpression expression = new MeanExpression(currentExpressionID++, node1, resultNode, true);
        storeExpression(expression, resultNode);
    }

    /**
     * Records variance expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createVarianceExpression(double expressionLock, Matrix argument1, Matrix result) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node resultNode = defineNode(result, true);
        VarianceExpression expression = new VarianceExpression(currentExpressionID++, node1, resultNode, false);
        storeExpression(expression, resultNode);
    }

    /**
     * Records variance expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createVarianceExpression(double expressionLock, MMatrix argument1, Matrix result) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node resultNode = defineNode(result, true);
        VarianceExpression expression = new VarianceExpression(currentExpressionID++, node1, resultNode, true);
        storeExpression(expression, resultNode);
    }

    /**
     * Records standard deviation expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void createStandardDeviationExpression(double expressionLock, Matrix argument1, Matrix result) throws MatrixException, DynamicParamException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node resultNode = defineNode(result, true);
        StandardDeviationExpression expression = new StandardDeviationExpression(currentExpressionID++, node1, resultNode, false);
        storeExpression(expression, resultNode);
    }

    /**
     * Records standard deviation expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void createStandardDeviationExpression(double expressionLock, MMatrix argument1, Matrix result) throws MatrixException, DynamicParamException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node resultNode = defineNode(result, true);
        StandardDeviationExpression expression = new StandardDeviationExpression(currentExpressionID++, node1, resultNode, true);
        storeExpression(expression, resultNode);
    }

    /**
     * Records norm expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @param p power of norm.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createNormExpression(double expressionLock, Matrix argument1, Matrix result, int p) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node resultNode = defineNode(result, true);
        NormExpression expression = new NormExpression(currentExpressionID++, node1, resultNode, p);
        storeExpression(expression, resultNode);
    }

    /**
     * Records unary (single argument) expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @param unaryFunction UnaryFunction of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createUnaryFunctionExpression(double expressionLock, Matrix argument1, Matrix result, UnaryFunction unaryFunction) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node resultNode = defineNode(result, true);
        UnaryFunctionExpression expression = new UnaryFunctionExpression(currentExpressionID++, node1, resultNode, unaryFunction);
        storeExpression(expression, resultNode);
    }

    /**
     * Records unary (single argument) expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @param unaryFunction UnaryFunction of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createUnaryFunctionExpression(double expressionLock, MMatrix argument1, MMatrix result, UnaryFunction unaryFunction) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node resultNode = defineNode(result, true);
        UnaryFunctionExpression expression = new UnaryFunctionExpression(currentExpressionID++, node1, resultNode, unaryFunction);
        storeExpression(expression, resultNode);
    }

    /**
     * Records binary (two argument) expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param argument2 second argument of expression.
     * @param result result of expression.
     * @param binaryFunction BinaryFunction of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createBinaryFunctionExpression(double expressionLock, Matrix argument1, Matrix argument2, Matrix result, BinaryFunction binaryFunction) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node node2 = defineNode(argument2, false);
        Node resultNode = defineNode(result, true);
        BinaryFunctionExpression expression = new BinaryFunctionExpression(currentExpressionID++, node1, node2, resultNode, binaryFunction);
        storeExpression(expression, resultNode);
    }

    /**
     * Records binary (two argument) expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param argument2 second argument of expression.
     * @param result result of expression.
     * @param binaryFunction BinaryFunction of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createBinaryFunctionExpression(double expressionLock, MMatrix argument1, Matrix argument2, MMatrix result, BinaryFunction binaryFunction) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node node2 = defineNode(argument2, false);
        Node resultNode = defineNode(result, true);
        BinaryFunctionExpression expression = new BinaryFunctionExpression(currentExpressionID++, node1, node2, resultNode, binaryFunction);
        storeExpression(expression, resultNode);
    }

    /**
     * Records binary (two argument) expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param argument2 second argument of expression.
     * @param result result of expression.
     * @param binaryFunction BinaryFunction of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createBinaryFunctionExpression(double expressionLock, MMatrix argument1, MMatrix argument2, MMatrix result, BinaryFunction binaryFunction) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        Node node1 = defineNode(argument1, false);
        Node node2 = defineNode(argument2, false);
        Node resultNode = defineNode(result, true);
        BinaryFunctionExpression expression = new BinaryFunctionExpression(currentExpressionID++, node1, node2, resultNode, binaryFunction);
        storeExpression(expression, resultNode);
    }

    /**
     * Stores expression into procedure chain
     *
     * @param expression expression.
     * @param resultNode result node
     */
    private void storeExpression(AbstractExpression expression, Node resultNode) {
        currentProcedureData.expressions.add(expression);
        currentProcedureData.reverseExpressionMap.put(resultNode, expression);
        finishExpression();
    }

}
