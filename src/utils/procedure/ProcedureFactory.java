/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure;

import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.*;
import utils.procedure.expression.*;
import utils.procedure.node.Node;
import utils.procedure.node.NodeRegister;

import java.io.Serial;
import java.io.Serializable;
import java.util.*;

/**
 * Builds computable procedures from chain of matrix operations including automated differentiation (automatic gradient) as backward operation.<br>
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
        private final LinkedList<Expression> expressions = new LinkedList<>();

        /**
         * List of expressions for backward gradient calculation.
         *
         */
        private final LinkedList<Expression> gradients = new LinkedList<>();

        /**
         * Map for expressions for backward (gradient) calculation.<br>
         * This temporary map is used to build list of backward gradient expressions.<br>
         *
         */
        private final HashMap<Node, Expression> reverseExpressionMap = new HashMap<>();

        /**
         * if true procedure has dependent nodes.
         *
         */
        private final HashSet<Node> dependentNodes = new HashSet<>();

        /**
         * Input matrices.
         *
         */
        private TreeMap<Integer, MMatrix> inputMatrices;

        /**
         * Input nodes.
         *
         */
        private final TreeMap<Integer, Node> inputNodes = new TreeMap<>();

        /**
         * Output nodes.
         *
         */
        private final TreeMap<Integer, Node> outputNodes = new TreeMap<>();

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
     * Constant matrices.
     *
     */
    private final HashSet<Matrix> constantMatrices = new HashSet<>();

    /**
     * Unique expression lock to reserve procedure factory.
     *
     */
    private transient int expressionLock = 0;

    /**
     * Count for expression lock reservations.
     *
     */
    private int expressionLockCount = 0;

    /**
     * Current node ID.
     *
     */
    private int currentNodeID = 0;

    /**
     * If true silently continues creation of existing procedure even new one is attempted.
     *
     */
    private transient boolean silentlyContinue = false;

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
     * @return resulting procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if operation fails.
     */
    public Procedure getProcedure(ForwardProcedure forwardProcedure) throws MatrixException, DynamicParamException, NeuralNetworkException {
        registerConstantMatrices(forwardProcedure.getParameterMatrices());
        registerConstantMatrices(forwardProcedure.getConstantMatrices());

        ProcedureData previousProcedureData = new ProcedureData();
        newProcedure(previousProcedureData, forwardProcedure.getInputMatrices(true));
        endProcedure(previousProcedureData, forwardProcedure.getForwardProcedure());

        ProcedureData nextProcedureData = new ProcedureData();
        newProcedure(nextProcedureData, forwardProcedure.getInputMatrices(false));
        endProcedure(nextProcedureData, forwardProcedure.getForwardProcedure());

        updateDependencies(previousProcedureData, nextProcedureData);

        nodeRegister.removeProcedureFactory();

        Expression previousExpression = null;
        for (Expression expression : nextProcedureData.expressions) {
            if (previousExpression != null) previousExpression.setNextExpression(expression);
            previousExpression = expression;
        }
        previousExpression = null;
        for (Expression expression : nextProcedureData.gradients) {
            if (previousExpression != null) previousExpression.setPreviousExpression(expression);
            previousExpression = expression;
        }

        return new Procedure(forwardProcedure.getProcedureName(), nextProcedureData.inputNodes, nextProcedureData.outputNodes, nextProcedureData.nodes, nextProcedureData.expressions.get(0), nextProcedureData.gradients.get(0), nextProcedureData.dependentNodes, forwardProcedure.getParameterMatrices(), forwardProcedure.getStopGradients(), forwardProcedure.isReversedInput(), forwardProcedure.isJoinedInput());
    }

    /**
     * Registers set of constant matrices.
     *
     * @param constantMatrices constant matrices to be registered.
     */
    private void registerConstantMatrices(Set<Matrix> constantMatrices) {
        if (constantMatrices == null) return;
        for (Matrix matrix : constantMatrices) matrix.setProcedureFactory(this);
        this.constantMatrices.addAll(constantMatrices);
    }

    /**
     * Starts building new procedure.
     *
     * @param inputMatrices input matrices.
     */
    private void newProcedure(ProcedureData procedureData, TreeMap<Integer, MMatrix> inputMatrices) {
        procedureData.inputMatrices = inputMatrices;
        for (MMatrix mMatrix : inputMatrices.values()) mMatrix.setProcedureFactory(this, true);
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
            int depth = outputMatrices.getDepth();
            for (int depthIndex = 0; depthIndex < depth; depthIndex++) {
                Matrix matrix = outputMatrices.get(depthIndex);
                if (!nodeRegister.contains(matrix)) throw new MatrixException("Setting of output node failed. No node corresponding output matrix is found.");
                procedureData.outputNodes.put(depthIndex, nodeRegister.getNode(matrix));
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
        HashMap<Node, Expression> reverseExpressionMap = new HashMap<>(procedureData.reverseExpressionMap);
        for (Node outputNode : procedureData.outputNodes.values()) resultNodes.push(outputNode);
        while (!resultNodes.empty()) {
            Expression expression = reverseExpressionMap.remove(resultNodes.pop());
            if (expression != null) {
                procedureData.gradients.add(expression);
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
        int expressionIDSize = nextProcedureData.expressions.size();
        for (int expressionID = 0; expressionID < expressionIDSize; expressionID++) {
            Expression previousExpression1 = previousProcedureData.reverseExpressionMap.get(nextProcedureData.expressions.get(expressionID).getArgument1());
            if (previousExpression1 != null) {
                updateNodeLink(nextProcedureData, nextProcedureData.expressions.get(previousExpression1.getExpressionID()).getResult(), nextProcedureData.expressions.get(expressionID).getArgument1());
            }
            Expression previousExpression2 = previousProcedureData.reverseExpressionMap.get(nextProcedureData.expressions.get(expressionID).getArgument2());
            if (previousExpression2 != null) {
                updateNodeLink(nextProcedureData, nextProcedureData.expressions.get(previousExpression2.getExpressionID()).getResult(), nextProcedureData.expressions.get(expressionID).getArgument2());
            }
        }
    }

    /**
     * Records dependencies between previous time step result node and next time step argument nodes.<br>
     *
     * @param nextProcedureData next procedure data.
     * @param fromResultNode from result node.
     * @param toArgumentNode to argument node.
     */
    private void updateNodeLink(ProcedureData nextProcedureData, Node fromResultNode, Node toArgumentNode) {
        fromResultNode.setToArgumentNode(toArgumentNode);
        nextProcedureData.dependentNodes.add(fromResultNode);

        toArgumentNode.setFromResultNode(fromResultNode);
        nextProcedureData.dependentNodes.add(toArgumentNode);
    }

    /**
     * Defines node for procedure. Sets input and result nodes as non-constant nodes.
     *
     * @param matrix matrix for node.
     * @return defined node.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Node defineNode(Matrix matrix) throws MatrixException {
        return defineNode (matrix, false);
    }

    /**
     * Defines node for procedure. Sets input and result nodes as non-constant nodes.
     *
     * @param matrix matrix for node.
     * @return defined node.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Node defineSingleNode(Matrix matrix) throws MatrixException {
        return defineNode (matrix, true);
    }

    /**
     * Defines node for procedure. Sets input and result nodes as non-constant nodes.
     *
     * @param matrix matrix for node.
     * @return defined node.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Node defineNode(Matrix matrix, boolean asSingleNode) throws MatrixException {
        currentNodeID += nodeRegister.nodeExists(matrix) ? 0 : 1;

        Node node = nodeRegister.defineNode(matrix, asSingleNode || constantMatrices.contains(matrix), currentExpressionID, currentNodeID);

        attachMatrixToInputNode(currentProcedureData.inputMatrices, matrix, currentProcedureData.inputNodes, node);

        currentProcedureData.nodes.add(node);

        return node;
    }

    /**
     * Attaches matrix to input node.
     *
     * @param inputMatrices input matrices
     * @param matrix matrix to be attached to node.
     * @param inputNodes input nodes.
     * @param node node.
     */
    private void attachMatrixToInputNode(TreeMap<Integer, MMatrix> inputMatrices, Matrix matrix, TreeMap<Integer, Node> inputNodes, Node node) {
        for (Map.Entry<Integer, MMatrix> entry : inputMatrices.entrySet()) {
            int depthIndex = entry.getKey();
            MMatrix inputMMatrix = entry.getValue();
            if (inputMMatrix.contains(matrix)) inputNodes.put(depthIndex, node);
            int depth = inputMMatrix.getDepth();
            for (int depthIndex1 = 0; depthIndex1 < depth; depthIndex1++) {
                Matrix inputMatrix = inputMMatrix.get(depthIndex1);
                if (inputMatrix == matrix) inputNodes.put(depthIndex, node);
            }
        }
    }

    /**
     * Defines node for procedure. Sets input and result nodes as non-constant nodes.
     *
     * @param mMatrix multi-matrix for node.
     * @return defined node.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Node defineNode(MMatrix mMatrix) throws MatrixException {
        boolean isSingleNode = false;

        int matrixDepth = mMatrix.getDepth();
        for (int inputDepth = 0; inputDepth < matrixDepth; inputDepth++) {
            if (constantMatrices.contains(mMatrix.get(inputDepth))) {
                isSingleNode = true;
                break;
            }
        }

        currentNodeID += nodeRegister.nodeExists(mMatrix) ? 0 : 1;

        Node node = nodeRegister.defineNode(mMatrix, isSingleNode, currentExpressionID, currentNodeID);

        attachMatrixToInputNode(currentProcedureData.inputMatrices, mMatrix, currentProcedureData.inputNodes, node);

        currentProcedureData.nodes.add(node);

        return node;
    }

    /**
     * Attaches multi-matrix to input node.
     *
     * @param inputMatrices input matrices
     * @param mMatrix multi-matrix to be attached to input node.
     * @param node node.
     */
    private void attachMatrixToInputNode(TreeMap<Integer, MMatrix> inputMatrices, MMatrix mMatrix, TreeMap<Integer, Node> inputNodes, Node node) {
        for (Map.Entry<Integer, MMatrix> entry : inputMatrices.entrySet()) {
            int depthIndex = entry.getKey();
            MMatrix inputMMatrix = entry.getValue();
            if (inputMMatrix == mMatrix) inputNodes.put(depthIndex, node);
        }
    }

    /**
     * Starts new expression and reserves procedure factory with expression lock.
     *
     * @param originator originator of procedure request.
     * @throws MatrixException throws exception if procedure factory is already reserved by another request
     * @return unique expression lock key.
     */
    public int startExpression(Object originator) throws MatrixException {
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
    public int startExpression(Object originator, boolean silentlyContinue) throws MatrixException {
        if (expressionLock > 0) {
            if (silentlyContinue) return 0;
            else throw new MatrixException("Procedure factory is reserved by: " + originator);
        }
        this.silentlyContinue = silentlyContinue;
        expressionLock = ++expressionLockCount;
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
        storeExpression(new AddExpression(currentExpressionID++, defineNode(argument1), defineNode(argument2), defineNode(result)));
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
        storeExpression(new AddExpression(currentExpressionID++, defineNode(argument1), defineNode(argument2), defineNode(result)));
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
        storeExpression(new AddExpression(currentExpressionID++, defineNode(argument1), defineNode(argument2), defineNode(result)));
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
        storeExpression(new SubtractExpression(currentExpressionID++, defineNode(argument1), defineNode(argument2), defineNode(result)));
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
        storeExpression(new SubtractExpression(currentExpressionID++, defineNode(argument1), defineNode(argument2), defineNode(result)));
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
        storeExpression(new SubtractExpression(currentExpressionID++, defineNode(argument1), defineNode(argument2), defineNode(result)));
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
        storeExpression(new DotExpression(currentExpressionID++, defineNode(argument1), defineNode(argument2), defineNode(result)));
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
        storeExpression(new DotExpression(currentExpressionID++, defineNode(argument1), defineNode(argument2), defineNode(result)));
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
        storeExpression(new DotExpression(currentExpressionID++, defineNode(argument1), defineNode(argument2), defineNode(result)));
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
        storeExpression(new MultiplyExpression(currentExpressionID++, defineNode(argument1), defineNode(argument2), defineNode(result)));
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
        storeExpression(new MultiplyExpression(currentExpressionID++, defineNode(argument1), defineNode(argument2), defineNode(result)));
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
        storeExpression(new MultiplyExpression(currentExpressionID++, defineNode(argument1), defineNode(argument2), defineNode(result)));
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
        storeExpression(new DivideExpression(currentExpressionID++, defineNode(argument1), defineNode(argument2), defineNode(result)));
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
        storeExpression(new DivideExpression(currentExpressionID++, defineNode(argument1), defineNode(argument2), defineNode(result)));
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
        storeExpression(new DivideExpression(currentExpressionID++, defineNode(argument1), defineNode(argument2), defineNode(result)));
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
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createConvolveExpression(double expressionLock, Matrix argument1, Matrix argument2, Matrix result, int stride, int dilation) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        storeExpression(new ConvolveExpression(currentExpressionID++, defineNode(argument1), defineNode(argument2), defineNode(result), stride, dilation));
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
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createCrosscorrelateExpression(double expressionLock, Matrix argument1, Matrix argument2, Matrix result, int stride, int dilation) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        storeExpression(new CrosscorrelateExpression(currentExpressionID++, defineNode(argument1), defineNode(argument2), defineNode(result), stride, dilation));
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
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createWinogradConvolveExpression(double expressionLock, Matrix argument1, Matrix argument2, Matrix result, int stride, int dilation) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        storeExpression(new WinogradConvolutionExpression(currentExpressionID++, defineNode(argument1), defineNode(argument2), defineNode(result), stride, dilation));
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
        storeExpression(new MaxPoolExpression(currentExpressionID++, defineNode(argument1), defineNode(result), stride, filterRowSize, filterColumnSize));
    }

    /**
     * Records random pool expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @param stride stride for operation.
     * @param filterRowSize filter row size for operation.
     * @param filterColumnSize filter column size for operation.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createRandomPoolExpression(double expressionLock, Matrix argument1, Matrix result, int stride, int filterRowSize, int filterColumnSize) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        storeExpression(new RandomPoolExpression(currentExpressionID++, defineNode(argument1), defineNode(result), stride, filterRowSize, filterColumnSize));
    }

    /**
     * Records cyclic pool expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @param stride stride for operation.
     * @param filterRowSize filter row size for operation.
     * @param filterColumnSize filter column size for operation.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createCyclicPoolExpression(double expressionLock, Matrix argument1, Matrix result, int stride, int filterRowSize, int filterColumnSize) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        storeExpression(new CyclicPoolExpression(currentExpressionID++, defineNode(argument1), defineNode(result), stride, filterRowSize, filterColumnSize));
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
        storeExpression(new AveragePoolExpression(currentExpressionID++, defineNode(argument1), defineNode(result), stride, filterRowSize, filterColumnSize));
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
        storeExpression(new SumExpression(currentExpressionID++, defineNode(argument1), defineNode(result), false));
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
        storeExpression(new SumExpression(currentExpressionID++, defineNode(argument1), defineSingleNode(result), true));
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
        storeExpression(new MeanExpression(currentExpressionID++, defineNode(argument1), defineNode(result), false));
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
        storeExpression(new MeanExpression(currentExpressionID++, defineNode(argument1), defineSingleNode(result), true));
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
        storeExpression(new VarianceExpression(currentExpressionID++, defineNode(argument1), defineNode(result), false));
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
        storeExpression(new VarianceExpression(currentExpressionID++, defineNode(argument1), defineSingleNode(result), true));
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
        storeExpression(new StandardDeviationExpression(currentExpressionID++, defineNode(argument1), defineNode(result), false));
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
        storeExpression(new StandardDeviationExpression(currentExpressionID++, defineNode(argument1), defineSingleNode(result), true));
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
        storeExpression(new NormExpression(currentExpressionID++, defineNode(argument1), defineNode(result), p));
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
        storeExpression(new UnaryFunctionExpression(currentExpressionID++, defineNode(argument1), defineNode(result), unaryFunction));
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
        storeExpression(new UnaryFunctionExpression(currentExpressionID++, defineNode(argument1), defineNode(result), unaryFunction));
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
        storeExpression(new BinaryFunctionExpression(currentExpressionID++, defineNode(argument1), defineNode(argument2), defineNode(result), binaryFunction));
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
        storeExpression(new BinaryFunctionExpression(currentExpressionID++, defineNode(argument1), defineNode(argument2), defineNode(result), binaryFunction));
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
        storeExpression(new BinaryFunctionExpression(currentExpressionID++, defineNode(argument1), defineNode(argument2), defineNode(result), binaryFunction));
    }

    /**
     * Records join expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param argument2 second argument of expression.
     * @param result result of expression.
     * @param joinedVertically if true joined vertically otherwise horizontally
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createJoinExpression(double expressionLock, Matrix argument1, Matrix argument2, Matrix result, boolean joinedVertically) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        storeExpression(new JoinExpression(currentExpressionID++, defineNode(argument1), defineNode(argument2), defineNode(result), joinedVertically));
    }

    /**
     * Records unjoin expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @param unjoinAtRow unjoins at row.
     * @param unjoinAtColumn unjoins at column.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createUnjoinExpression(double expressionLock, Matrix argument1, Matrix result, int unjoinAtRow, int unjoinAtColumn) throws MatrixException {
        if (checkOngoingExpression(expressionLock, argument1)) return;
        storeExpression(new UnjoinExpression(currentExpressionID++, defineNode(argument1), defineNode(result), unjoinAtRow, unjoinAtColumn));
    }

    /**
     * Stores expression into procedure chain
     *
     * @param expression expression.
     */
    private void storeExpression(Expression expression) {
        currentProcedureData.expressions.add(expression);
        currentProcedureData.reverseExpressionMap.put(expression.getResult(), expression);
        finishExpression();
    }

}
