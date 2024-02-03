/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package utils.procedure;

import utils.configurable.DynamicParamException;
import utils.matrix.BinaryFunction;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.UnaryFunction;
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
        private TreeMap<Integer, Matrix> inputMatrices;

        /**
         * Input nodes.
         *
         */
        private final HashMap<Integer, Node> inputNodes = new HashMap<>();

        /**
         * Output node.
         *
         */
        private Node outputNode;

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
     * Default constructor for procedure factory.
     */
    public ProcedureFactory() {
    }

    /**
     * Synchronizes procedure factories for matrices.
     *
     * @param first first matrix
     * @param result result matrix
     * @throws MatrixException throws exception if matrices has conflicting procedure factories.
     */
    public static void synchronize(Matrix first, Matrix result) throws MatrixException {
        ProcedureFactory firstProcedureFactory = first.getProcedureFactory();
        ProcedureFactory resultProcedureFactory = result.getProcedureFactory();
        HashSet<ProcedureFactory> procedureFactories = new HashSet<>();
        if (firstProcedureFactory != null) procedureFactories.add(firstProcedureFactory);
        if (resultProcedureFactory != null) procedureFactories.add(resultProcedureFactory);
        if (procedureFactories.size() == 0) return;
        if (procedureFactories.size() > 1) throw new MatrixException("Matrices have conflicting procedure factories.");
        ProcedureFactory procedureFactory = (ProcedureFactory)procedureFactories.toArray()[0];
        first.setProcedureFactory(procedureFactory);
        result.setProcedureFactory(procedureFactory);
    }

    /**
     * Synchronizes procedure factories for matrices.
     *
     * @param first first matrix
     * @param second second matrix
     * @param result result matrix
     * @throws MatrixException throws exception if matrices has conflicting procedure factories.
     */
    public static void synchronize(Matrix first, Matrix second, Matrix result) throws MatrixException {
        ProcedureFactory firstProcedureFactory = first.getProcedureFactory();
        ProcedureFactory secondProcedureFactory = second.getProcedureFactory();
        ProcedureFactory resultProcedureFactory = result.getProcedureFactory();
        HashSet<ProcedureFactory> procedureFactories = new HashSet<>();
        if (firstProcedureFactory != null) procedureFactories.add(firstProcedureFactory);
        if (secondProcedureFactory != null) procedureFactories.add(secondProcedureFactory);
        if (resultProcedureFactory != null) procedureFactories.add(resultProcedureFactory);
        if (procedureFactories.size() == 0) return;
        if (procedureFactories.size() > 1) throw new MatrixException("Matrices have conflicting procedure factories.");
        ProcedureFactory procedureFactory = (ProcedureFactory)procedureFactories.toArray()[0];
        first.setProcedureFactory(procedureFactory);
        second.setProcedureFactory(procedureFactory);
        result.setProcedureFactory(procedureFactory);
    }

    /**
     * Returns procedure
     *
     * @param forwardProcedure reference to class that defines forward procedure.
     * @return resulting procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Procedure getProcedure(ForwardProcedure forwardProcedure) throws MatrixException, DynamicParamException {
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

        return new Procedure(nextProcedureData.inputNodes, nextProcedureData.outputNode, nextProcedureData.nodes, nextProcedureData.expressions.get(0), nextProcedureData.gradients.get(0), nextProcedureData.dependentNodes, forwardProcedure.getParameterMatrices(), forwardProcedure.getStopGradients(), forwardProcedure.isReversedInput(), forwardProcedure.isJoinedInput());
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
    private void newProcedure(ProcedureData procedureData, TreeMap<Integer, Matrix> inputMatrices) {
        procedureData.inputMatrices = inputMatrices;
        for (Matrix matrix : inputMatrices.values()) matrix.setProcedureFactory(this);
        currentExpressionID = 0;
        currentProcedureData = procedureData;
    }

    /**
     * Finalizes building current procedure.
     *
     * @param outputMatrix output matrix.
     * @throws MatrixException throws exception if setting of output matrix and node fails.
     */
    private void endProcedure(ProcedureData procedureData, Matrix outputMatrix) throws MatrixException {
        if (!nodeRegister.contains(outputMatrix)) throw new MatrixException("Setting of output node failed. No node corresponding output matrix is found.");
        procedureData.outputNode = nodeRegister.getNode(outputMatrix);
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
        resultNodes.push(procedureData.outputNode);
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
     * @throws MatrixException throws exception if dimensions of from result node and to argument node are not matching.
     */
    private void updateDependencies(ProcedureData previousProcedureData, ProcedureData nextProcedureData) throws MatrixException {
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
     * @throws MatrixException throws exception if dimensions of from result node and to argument node are not matching.
     */
    private void updateNodeLink(ProcedureData nextProcedureData, Node fromResultNode, Node toArgumentNode) throws MatrixException {
        if (fromResultNode.getRows() != toArgumentNode.getRows() || fromResultNode.getColumns() != toArgumentNode.getColumns()) throw new MatrixException("Dimensions of from result node " + fromResultNode.getRows() + "x" + fromResultNode.getColumns() + " and to argument node " + toArgumentNode.getRows() + "x" + toArgumentNode.getColumns() + " are not matching.");

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
    private Node defineNode(Matrix matrix, boolean isSingleNode) throws MatrixException {
        currentNodeID += nodeRegister.nodeExists(matrix) ? 0 : 1;

        Node node = nodeRegister.defineNode(matrix, isSingleNode || constantMatrices.contains(matrix), currentNodeID);

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
    private void attachMatrixToInputNode(TreeMap<Integer, Matrix> inputMatrices, Matrix matrix, HashMap<Integer, Node> inputNodes, Node node) {
        for (Map.Entry<Integer, Matrix> entry : inputMatrices.entrySet()) {
            if (entry.getValue() == matrix) inputNodes.put(entry.getKey(), node);
        }
    }

    /**
     * Starts new expression and reserves procedure factory with expression lock.
     *
     * @return unique expression lock key.
     */
    public int startExpression() {
        if (expressionLock > 0) return 0;
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
     * @return returns true is existing expression creation is ongoing otherwise false.
     */
    public boolean checkOngoingExpression(double expressionLock) {
        return this.expressionLock != expressionLock;
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
        if (checkOngoingExpression(expressionLock)) return;
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
        if (checkOngoingExpression(expressionLock)) return;
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
        if (checkOngoingExpression(expressionLock)) return;
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
        if (checkOngoingExpression(expressionLock)) return;
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
        if (checkOngoingExpression(expressionLock)) return;
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
     * @param isDepthSeparable if true convolution is depth separable
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createConvolveExpression(double expressionLock, Matrix argument1, Matrix argument2, Matrix result, int stride, int dilation, boolean isDepthSeparable) throws MatrixException {
        if (checkOngoingExpression(expressionLock)) return;
        storeExpression(new ConvolveExpression(currentExpressionID++, defineNode(argument1), defineNode(argument2), defineNode(result), stride, dilation, isDepthSeparable));
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
     * @param isDepthSeparable if true convolution is depth separable
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createCrosscorrelateExpression(double expressionLock, Matrix argument1, Matrix argument2, Matrix result, int stride, int dilation, boolean isDepthSeparable) throws MatrixException {
        if (checkOngoingExpression(expressionLock)) return;
        storeExpression(new CrosscorrelateExpression(currentExpressionID++, defineNode(argument1), defineNode(argument2), defineNode(result), stride, dilation, isDepthSeparable));
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
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void createWinogradConvolveExpression(double expressionLock, Matrix argument1, Matrix argument2, Matrix result, int stride, int dilation) throws MatrixException, DynamicParamException {
        if (checkOngoingExpression(expressionLock)) return;
        storeExpression(new WinogradConvolutionExpression(currentExpressionID++, defineNode(argument1), defineNode(argument2), defineNode(result), stride, dilation));
    }

    /**
     * Records max pool expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @param dilation dilation for operation.
     * @param stride stride for operation.
     * @param filterRowSize filter row size for operation.
     * @param filterColumnSize filter column size for operation.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createMaxPoolExpression(double expressionLock, Matrix argument1, Matrix result, int dilation, int stride, int filterRowSize, int filterColumnSize) throws MatrixException {
        if (checkOngoingExpression(expressionLock)) return;
        storeExpression(new MaxPoolExpression(currentExpressionID++, defineNode(argument1), defineNode(result), dilation, stride, filterRowSize, filterColumnSize));
    }

    /**
     * Records random pool expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @param dilation dilation for operation.
     * @param stride stride for operation.
     * @param filterRowSize filter row size for operation.
     * @param filterColumnSize filter column size for operation.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createRandomPoolExpression(double expressionLock, Matrix argument1, Matrix result, int dilation, int stride, int filterRowSize, int filterColumnSize) throws MatrixException {
        if (checkOngoingExpression(expressionLock)) return;
        storeExpression(new RandomPoolExpression(currentExpressionID++, defineNode(argument1), defineNode(result), dilation, stride, filterRowSize, filterColumnSize));
    }

    /**
     * Records cyclic pool expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @param dilation dilation for operation.
     * @param stride stride for operation.
     * @param filterRowSize filter row size for operation.
     * @param filterColumnSize filter column size for operation.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createCyclicPoolExpression(double expressionLock, Matrix argument1, Matrix result, int dilation, int stride, int filterRowSize, int filterColumnSize) throws MatrixException {
        if (checkOngoingExpression(expressionLock)) return;
        storeExpression(new CyclicPoolExpression(currentExpressionID++, defineNode(argument1), defineNode(result), dilation, stride, filterRowSize, filterColumnSize));
    }

    /**
     * Records average pool expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @param dilation dilation for operation.
     * @param stride stride for operation.
     * @param filterRowSize filter row size for operation.
     * @param filterColumnSize filter column size for operation.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createAveragePoolExpression(double expressionLock, Matrix argument1, Matrix result, int dilation, int stride, int filterRowSize, int filterColumnSize) throws MatrixException {
        if (checkOngoingExpression(expressionLock)) return;
        storeExpression(new AveragePoolExpression(currentExpressionID++, defineNode(argument1), defineNode(result), dilation, stride, filterRowSize, filterColumnSize));
    }

    /**
     * Records sum expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createSumExpression(double expressionLock, Matrix argument1, Matrix result, int direction) throws MatrixException {
        if (checkOngoingExpression(expressionLock)) return;
        storeExpression(new SumExpression(currentExpressionID++, defineNode(argument1), defineNode(result), false, direction));
    }

    /**
     * Records sum expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @param executeAsSingleStep true if calculation is done per index otherwise over all indices.
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createSumExpression(double expressionLock, Matrix argument1, Matrix result, boolean executeAsSingleStep, int direction) throws MatrixException {
        if (checkOngoingExpression(expressionLock)) return;
        storeExpression(new SumExpression(currentExpressionID++, defineNode(argument1), defineNode(result), executeAsSingleStep, direction));
    }

    /**
     * Records mean expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createMeanExpression(double expressionLock, Matrix argument1, Matrix result, int direction) throws MatrixException {
        if (checkOngoingExpression(expressionLock)) return;
        storeExpression(new MeanExpression(currentExpressionID++, defineNode(argument1), defineNode(result), false, direction));
    }

    /**
     * Records mean expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @param executeAsSingleStep true if calculation is done per index otherwise over all indices.
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createMeanExpression(double expressionLock, Matrix argument1, Matrix result, boolean executeAsSingleStep, int direction) throws MatrixException {
        if (checkOngoingExpression(expressionLock)) return;
        storeExpression(new MeanExpression(currentExpressionID++, defineNode(argument1), defineSingleNode(result), executeAsSingleStep, direction));
    }

    /**
     * Records variance expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createVarianceExpression(double expressionLock, Matrix argument1, Matrix result, int direction) throws MatrixException {
        if (checkOngoingExpression(expressionLock)) return;
        storeExpression(new VarianceExpression(currentExpressionID++, defineNode(argument1), defineNode(result), false, direction));
    }

    /**
     * Records variance expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @param executeAsSingleStep true if calculation is done per index otherwise over all indices.
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createVarianceExpression(double expressionLock, Matrix argument1, Matrix result, boolean executeAsSingleStep, int direction) throws MatrixException {
        if (checkOngoingExpression(expressionLock)) return;
        storeExpression(new VarianceExpression(currentExpressionID++, defineNode(argument1), executeAsSingleStep ? defineSingleNode(result) : defineNode(result), executeAsSingleStep, direction));
    }

    /**
     * Records standard deviation expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @param executeAsSingleStep true if calculation is done per index otherwise over all indices.
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @throws MatrixException throws exception if adding of expression fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void createStandardDeviationExpression(double expressionLock, Matrix argument1, Matrix result, boolean executeAsSingleStep, int direction) throws MatrixException, DynamicParamException {
        if (checkOngoingExpression(expressionLock)) return;
        storeExpression(new StandardDeviationExpression(currentExpressionID++, defineNode(argument1), executeAsSingleStep ? defineSingleNode(result) : defineNode(result), executeAsSingleStep, direction));
    }

    /**
     * Records standard deviation expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @param direction if value is one normalizes over row direction, if two normalizes over column direction, if three normalizes over depth direction, otherwise normalized over all directions.
     * @throws MatrixException throws exception if adding of expression fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void createStandardDeviationExpression(double expressionLock, Matrix argument1, Matrix result, int direction) throws MatrixException, DynamicParamException {
        if (checkOngoingExpression(expressionLock)) return;
        storeExpression(new StandardDeviationExpression(currentExpressionID++, defineNode(argument1), defineNode(result), false, direction));
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
        if (checkOngoingExpression(expressionLock)) return;
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
        if (checkOngoingExpression(expressionLock)) return;
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
        if (checkOngoingExpression(expressionLock)) return;
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
        if (checkOngoingExpression(expressionLock)) return;
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
     * @param unjoinAtDepth unjoins at depth.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createUnjoinExpression(double expressionLock, Matrix argument1, Matrix result, int unjoinAtRow, int unjoinAtColumn, int unjoinAtDepth) throws MatrixException {
        if (checkOngoingExpression(expressionLock)) return;
        storeExpression(new UnjoinExpression(currentExpressionID++, defineNode(argument1), defineNode(result), unjoinAtRow, unjoinAtColumn, unjoinAtDepth));
    }

    /**
     * Records flatten expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createFlattenExpression(double expressionLock, Matrix argument1, Matrix result) throws MatrixException {
        if (checkOngoingExpression(expressionLock)) return;
        storeExpression(new FlattenExpression(currentExpressionID++, defineNode(argument1), defineNode(result)));
    }

    /**
     * Records dropout expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @param probability probability.
     * @param monte_carlo if true is monte carlo dropout otherwise normal dropout.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createDropoutExpression(double expressionLock, Matrix argument1, Matrix result, double probability, boolean monte_carlo) throws MatrixException {
        if (checkOngoingExpression(expressionLock)) return;
        storeExpression(new DropoutExpression(currentExpressionID++, defineNode(argument1), defineNode(result), probability, monte_carlo));
    }

    /**
     * Records gradient clipping expression to procedure factory.
     *
     * @param expressionLock unique expression lock key.
     * @param argument1 first argument of expression.
     * @param result result of expression.
     * @param threshold threshold.
     * @throws MatrixException throws exception if adding of expression fails.
     */
    public void createGradientClippingExpression(double expressionLock, Matrix argument1, Matrix result, double threshold) throws MatrixException {
        if (checkOngoingExpression(expressionLock)) return;
        storeExpression(new GradientClippingExpression(currentExpressionID++, defineNode(argument1), defineNode(result), threshold));
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
