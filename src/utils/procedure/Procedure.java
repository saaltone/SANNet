/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package utils.procedure;

import core.normalization.Normalization;
import core.regularization.Regularization;
import utils.DynamicParamException;
import utils.Sequence;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.procedure.expression.Expression;
import utils.procedure.node.Node;

import java.io.Serial;
import java.io.Serializable;
import java.util.*;

/**
 * Defines computable procedure that has chain of forward computable expressions and backward computable gradient expressions (based on automatic gradient).<br>
 *
 */
public class Procedure implements Serializable {

    @Serial
    private static final long serialVersionUID = 9207418704022664014L;

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

    /**
     * Chain of expressions.
     *
     */
    private final Expression expressionChain;

    /**
     * Chain of gradients.
     *
     */
    private final Expression gradientChain;

    /**
     * True if procedure has dependent nodes.
     *
     */
    private final boolean hasDependentNodes;

    /**
     * Constructor for procedure.
     *
     * @param inputNodes input nodes for procedure.
     * @param outputNodes input nodes for procedure.
     * @param nodes all nodes for procedure.
     * @param expressionChain chain of expressions describing procedure.
     * @param gradientChain chain of gradients for procedure.
     * @param hasDependentNodes true if procedure has dependent nodes.
     */
    public Procedure(HashMap<Integer, Node> inputNodes, HashMap<Integer, Node> outputNodes, HashSet<Node> nodes, Expression expressionChain, Expression gradientChain, boolean hasDependentNodes) {
        this.inputNodes.putAll(inputNodes);
        this.outputNodes.putAll(outputNodes);
        this.nodes.addAll(nodes);
        this.expressionChain = expressionChain;
        this.gradientChain = gradientChain;
        this.hasDependentNodes = hasDependentNodes;
    }

    /**
     * Sets normalizers for node.
     *
     * @param normalizers normalizers for node.
     */
    public void setNormalizers(HashSet<Normalization> normalizers) {
        for (Node node : nodes) node.setNormalizers(normalizers);
    }

    /**
     * Sets regularizers for node.
     *
     * @param regularizers regularizers for node.
     */
    public void setRegularizers(HashSet<Regularization> regularizers) {
        for (Node node : nodes) node.setRegularizers(regularizers);
    }

    /**
     * Resets data for every index in nodes of procedure.
     *
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public void reset() throws MatrixException {
        reset(true);
    }

    /**
     * Resets data for every index in nodes of procedure.
     *
     * @param resetDependentNodes if true resets also dependent nodes.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public void reset(boolean resetDependentNodes) throws MatrixException {
        for (Node node : nodes) node.resetNode(resetDependentNodes);
        expressionChain.reset();
    }

    /**
     * Initializes normalization for every node.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void initialize() throws MatrixException, DynamicParamException {
        for (Node node : nodes) node.initializeNormalization();
    }

    /**
     * Returns node corresponding specific matrix.
     *
     * @param matrix matrix.
     * @return node corresponding specific matrix
     */
    public Node getNode(Matrix matrix) {
        for (Node node : nodes) if (node.contains(matrix)) return node;
        return null;
    }

    /**
     * Returns input nodes.
     *
     * @return input nodes.
     */
    public HashMap<Integer, Node> getInputNodes() {
        return inputNodes;
    }

    /**
     * Returns output nodes.
     *
     * @return output nodes.
     */
    public HashMap<Integer, Node> getOutputNodes() {
        return outputNodes;
    }

    /**
     * Returns nodes.
     *
     * @return nodes.
     */
    public HashSet<Node> getNodes() {
        return nodes;
    }

    /**
     * Checks if procedure has dependencies between output and input nodes.
     *
     * @return returns true if there are dependencies otherwise returns false.
     */
    public boolean hasDependencies() {
        return hasDependentNodes;
    }

    /**
     * Stores matrix dependency
     *
     * @param backupIndex backup index
     * @throws MatrixException throws exception if storing dependencies fails.
     */
    public void storeDependencies(int backupIndex) throws MatrixException {
        for (Node node : nodes) node.storeMatrixDependency(backupIndex);
    }

    /**
     * Restores matrix dependency.
     *
     * @param backupIndex backup index.
     * @throws MatrixException throws exception if restoring of backup fails.
     */
    public void restoreDependencies(int backupIndex) throws MatrixException {
        for (Node node : nodes) node.restoreMatrixDependency(backupIndex);
    }

    /**
     * Calculates chain of forward expressions.
     *
     * @param inputSequence input sequence.
     * @param outputSequence output sequence.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void calculateExpression(Sequence inputSequence, Sequence outputSequence) throws MatrixException, DynamicParamException {
        if (hasDependencies()) calculateExpressionPerSample(inputSequence, outputSequence);
        else calculateExpressionPerStep(inputSequence, outputSequence);
    }

    /**
     * Calculates chain of forward expressions sample by sample.
     *
     * @param inputSequence input sequence.
     * @param outputSequence output sequence.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void calculateExpressionPerSample(Sequence inputSequence, Sequence outputSequence) throws MatrixException, DynamicParamException {
        for (Integer sampleIndex : inputSequence.keySet()) {
            setInputSample(sampleIndex, inputSequence.get(sampleIndex));

            expressionChain.calculateExpressionStep(sampleIndex, inputSequence.firstKey(), inputSequence.lastKey());

            outputSequence.put(sampleIndex, setOutputSample(sampleIndex));
        }
    }

    /**
     * Calculates chain of forward expressions for all samples.
     *
     * @param inputSequence input sequence.
     * @param outputSequence output sequence.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void calculateExpressionPerStep(Sequence inputSequence, Sequence outputSequence) throws MatrixException, DynamicParamException {
        for (Integer sampleIndex : inputSequence.keySet()) setInputSample(sampleIndex, inputSequence.get(sampleIndex));

        expressionChain.calculateExpressionStep(inputSequence.keySet(), inputSequence.firstKey(), inputSequence.lastKey());

        for (Integer sampleIndex : inputSequence.keySet()) outputSequence.put(sampleIndex, setOutputSample(sampleIndex));
    }

    /**
     * Sets input sample.
     *
     * @param sampleIndex sample index
     * @param inputSample input sample
     * @throws MatrixException throws exception if calculation fails.
     */
    private void setInputSample(int sampleIndex, MMatrix inputSample) throws MatrixException {
        for (Integer nodeIndex : inputSample.keySet()) getInputNodes().get(nodeIndex).setMatrix(sampleIndex, inputSample.get(nodeIndex));
    }

    /**
     * Sets output sample.
     *
     * @param sampleIndex sample index
     * @return outputSample.
     * @throws MatrixException throws exception if calculation fails.
     */
    private MMatrix setOutputSample(int sampleIndex) throws MatrixException {
        MMatrix outputSample = new MMatrix(getOutputNodes().size());
        for (Integer nodeIndex : getOutputNodes().keySet()) outputSample.put(nodeIndex, getOutputNodes().get(nodeIndex).getMatrix(sampleIndex));
        return outputSample;
    }

    /**
     * Calculates chain of forward expressions.
     *
     * @param inputMatrix input matrices.
     * @return output matrix.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public MMatrix calculateExpression(MMatrix inputMatrix) throws MatrixException, DynamicParamException {
        Sequence inputSequence = new Sequence(1);
        for (Integer sampleIndex : inputMatrix.keySet()) inputSequence.put(sampleIndex, new MMatrix(inputMatrix.get(sampleIndex)));

        Sequence outputSequence = new Sequence(1);
        calculateExpressionPerStep(inputSequence, outputSequence);

        MMatrix outputMatrix = new MMatrix();
        for (Integer sampleIndex : outputSequence.keySet()) outputMatrix.put(sampleIndex, outputSequence.get(sampleIndex, 0));
        return outputMatrix;
    }

    /**
     * Calculates chain of forward expressions.
     *
     * @param inputMatrix input matrices.
     * @param sampleIndex sample index.
     * @return output matrix.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix calculateExpression(Matrix inputMatrix, int sampleIndex) throws MatrixException, DynamicParamException {
        setInputSample(sampleIndex, new MMatrix(inputMatrix));
        expressionChain.calculateExpressionStep(sampleIndex, 0, 0);
        return getOutputNodes().get(0).getMatrix(sampleIndex);
    }

    /**
     * Returns regularization error.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return regularization error.
     */
    public double getRegularizationError() throws MatrixException, DynamicParamException {
        return expressionChain.cumulateRegularizationError();
    }

    /**
     * Calculates chain of backward expressions for multiple inputs per gradient expression step.
     *
     * @param outputGradientSequence output gradients.
     * @param inputGradientSequence input gradients.
     * @param steps number of steps calculated backwards.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void calculateGradient(Sequence outputGradientSequence, Sequence inputGradientSequence, int steps) throws MatrixException, DynamicParamException {
        if (hasDependencies()) calculateGradientPerSample(outputGradientSequence, inputGradientSequence, steps);
        else calculateGradientPerStep(outputGradientSequence, inputGradientSequence, steps);
    }

    /**
     * Calculates chain of backward expressions for multiple inputs per gradient expression step per sample.
     *
     * @param outputGradientSequence output gradients.
     * @param inputGradientSequence input gradients.
     * @param numberOfGradientSteps number of steps calculated backwards.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void calculateGradientPerSample(Sequence outputGradientSequence, Sequence inputGradientSequence, int numberOfGradientSteps) throws MatrixException, DynamicParamException {
        int gradientStepCount = 0;
        for (Integer sampleIndex : outputGradientSequence.descendingKeySet()) {
            setOutputSampleGradient(sampleIndex, outputGradientSequence.get(sampleIndex));

            gradientChain.calculateGradientStep(sampleIndex, outputGradientSequence.firstKey());

            inputGradientSequence.put(sampleIndex, setInputSampleGradient(sampleIndex));
            if (numberOfGradientSteps > 0 && ++gradientStepCount >= numberOfGradientSteps) break;
        }
    }

    /**
     * Calculates chain of backward expressions for multiple inputs per gradient expression step.
     *
     * @param outputGradientSequence output gradients.
     * @param inputGradientSequence input gradients.
     * @param numberOfGradientSteps number of steps calculated backwards.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void calculateGradientPerStep(Sequence outputGradientSequence, Sequence inputGradientSequence, int numberOfGradientSteps) throws MatrixException, DynamicParamException {
        int gradientStepCount = 0;
        for (Integer sampleIndex : outputGradientSequence.keySet()) {
            setOutputSampleGradient(sampleIndex, outputGradientSequence.get(sampleIndex));
            if (numberOfGradientSteps > 0 && ++gradientStepCount >= numberOfGradientSteps) break;
        }

        gradientChain.calculateGradientStep(outputGradientSequence.keySet(), outputGradientSequence.lastKey(), numberOfGradientSteps);

        gradientStepCount = 0;
        for (Integer sampleIndex : outputGradientSequence.keySet()) {
            inputGradientSequence.put(sampleIndex, setInputSampleGradient(sampleIndex));
            if (numberOfGradientSteps > 0 && ++gradientStepCount >= numberOfGradientSteps) break;
        }
    }

    /**
     * Sets output sample gradient.
     *
     * @param sampleIndex sample index
     * @param outputSampleGradient output sample gradient
     * @throws MatrixException throws exception if calculation fails.
     */
    private void setOutputSampleGradient(int sampleIndex, MMatrix outputSampleGradient) throws MatrixException {
        for (Integer nodeIndex : outputSampleGradient.keySet()) getOutputNodes().get(nodeIndex).setGradient(sampleIndex, outputSampleGradient.get(nodeIndex));
    }

    /**
     * Sets input sample gradient.
     *
     * @param sampleIndex input sample
     * @return inputSampleGradient input sample gradient
     * @throws MatrixException throws exception if calculation fails.
     */
    private MMatrix setInputSampleGradient(int sampleIndex) throws MatrixException {
        MMatrix inputSampleGradient = new MMatrix(getInputNodes().size());
        for (Integer nodeIndex : getInputNodes().keySet()) inputSampleGradient.put(nodeIndex, getInputNodes().get(nodeIndex).getGradient(sampleIndex));
        return inputSampleGradient;
    }

    /**
     * Calculates backward chain of gradient expressions.
     *
     * @param outputGradient output gradient for procedure.
     * @return input gradient.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public MMatrix calculateGradient(MMatrix outputGradient) throws MatrixException, DynamicParamException {
        Sequence outputGradientSequence = new Sequence(1);
        for (Integer sampleIndex : outputGradient.keySet()) outputGradientSequence.put(sampleIndex, new MMatrix(outputGradient.get(sampleIndex)));

        Sequence inputGradientSequence = new Sequence(1);
        calculateGradientPerStep(outputGradientSequence, inputGradientSequence, -1);

        MMatrix inputGradient = new MMatrix();
        for (Integer sampleIndex : inputGradientSequence.keySet()) inputGradient.put(sampleIndex, inputGradientSequence.get(sampleIndex, 0));
        return inputGradient;
    }

    /**
     * Calculates backward chain of gradient expressions.
     *
     * @param outputGradient output gradient for procedure.
     * @param sampleIndex sample index.
     * @return input gradient.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix calculateGradient(Matrix outputGradient, int sampleIndex) throws MatrixException, DynamicParamException {
        setOutputSampleGradient(sampleIndex, new MMatrix(outputGradient));
        gradientChain.calculateGradientStep(sampleIndex, 0);
        return getInputNodes().get(0).getGradient(sampleIndex);
    }


    /**
     * Returns gradient for specific constant matrix.
     *
     * @param constantMatrix constant matrix.
     * @return gradient corresponding specific constant matrix.
     * @throws MatrixException throws exception is no node corresponding reference matrix is found.
     */
    public Matrix getGradient(Matrix constantMatrix) throws MatrixException {
        Node node = getNode(constantMatrix);
        if (node == null) throw new MatrixException("No such reference matrix registered.");
        return node.getGradientMean();
    }

    /**
     * Sets if gradient is updated for nodes of this expression. If true gradient is not updated otherwise it is updated.
     *
     * @param referenceMatrices reference matrices of nodes.
     * @param stopGradient if true gradient is not updated otherwise it is updated.
     */
    public void setStopGradient(HashSet<Matrix> referenceMatrices, boolean stopGradient) {
        for (Matrix referenceMatrix : referenceMatrices) setStopGradient(referenceMatrix, stopGradient);
    }

    /**
     * Sets if gradient is updated for nodes of this expression. If true gradient is not updated otherwise it is updated.
     *
     * @param referenceMatrix reference matrix of node.
     * @param stopGradient if true gradient is not updated otherwise it is updated.
     */
    public void setStopGradient(Matrix referenceMatrix, boolean stopGradient) {
        for (Node node : nodes) {
            if (node.isReferenceOf(referenceMatrix)) node.setStopGradient(stopGradient);
        }
    }

    /**
     * Prints expression chain.
     *
     */
    public void printExpressionChain() {
        expressionChain.printExpressionChain();
    }

    /**
     * Prints gradient chain.
     *
     */
    public void printGradientChain() {
        gradientChain.printGradientChain();
    }

}
