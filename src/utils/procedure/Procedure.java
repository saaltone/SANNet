/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.procedure;

import core.normalization.Normalization;
import core.regularization.Regularization;
import utils.Sequence;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

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
    private final HashMap<Integer, Node> inputNodes = new HashMap<>();

    /**
     * Output node.
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
    private final AbstractExpression expressionChain;

    /**
     * Chain of gradient expressions.
     *
     */
    private final AbstractExpression gradientExpressionChain;

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
     * @param gradientExpressionChain chain of gradient expressions for procedure.
     * @param hasDependentNodes true if procedure has dependent nodes.
     */
    public Procedure(HashMap<Integer, Node> inputNodes, HashMap<Integer, Node> outputNodes, HashSet<Node> nodes, AbstractExpression expressionChain, AbstractExpression gradientExpressionChain, boolean hasDependentNodes) {
        this.inputNodes.putAll(inputNodes);
        this.outputNodes.putAll(outputNodes);
        this.nodes.addAll(nodes);
        this.expressionChain = expressionChain;
        this.gradientExpressionChain = gradientExpressionChain;
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
     */
    public void reset() {
        reset(true);
    }

    /**
     * Resets data for every index in nodes of procedure.
     *
     * @param resetDependentNodes if true resets also dependent nodes.
     */
    public void reset(boolean resetDependentNodes) {
        for (Node node : nodes) node.resetNode(resetDependentNodes);
    }

    /**
     * Initializes normalization for every node.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void initialize() throws MatrixException {
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
     * Calculates chain of forward expressions for multiple inputs.
     *
     * @param inputSequence input sequence.
     * @param outputSequence output sequence.
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpression(Sequence inputSequence, Sequence outputSequence) throws MatrixException {
        if (hasDependencies()) calculateExpressionPerSample(inputSequence, outputSequence);
        else calculateExpressionPerStep(inputSequence, outputSequence);
    }

    /**
     * Calculates chain of forward expressions for multiple inputs.
     *
     * @param inputSequence input sequence.
     * @param outputSequence output sequence.
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpressionPerSample(Sequence inputSequence, Sequence outputSequence) throws MatrixException {
        for (Integer sampleIndex : inputSequence.keySet()) {
            setInputSample(sampleIndex, inputSequence.get(sampleIndex));

            expressionChain.calculateExpressionStep(sampleIndex, inputSequence.firstKey());

            MMatrix outputSample = new MMatrix(getOutputNodes().size());
            setOutputSample(sampleIndex, outputSample);
            outputSequence.put(sampleIndex, outputSample);
        }
    }

    /**
     * Calculates chain of forward expressions for multiple inputs per expression step.
     *
     * @param inputSequence input sequence.
     * @param outputSequence output sequence.
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpressionPerStep(Sequence inputSequence, Sequence outputSequence) throws MatrixException {
        for (Integer sampleIndex : inputSequence.keySet()) setInputSample(sampleIndex, inputSequence.get(sampleIndex));

        expressionChain.calculateExpressionStep(inputSequence.keySet(), inputSequence.firstKey());

        for (Integer sampleIndex : inputSequence.keySet()) {
            MMatrix outputSample = new MMatrix(getOutputNodes().size());
            setOutputSample(sampleIndex, outputSample);
            outputSequence.put(sampleIndex, outputSample);
        }
    }

    /**
     * Sets input sample.
     *
     * @param sampleIndex sample index
     * @param inputSample input sample
     * @throws MatrixException throws exception if calculation fails.
     */
    private void setInputSample(int sampleIndex, MMatrix inputSample) throws MatrixException {
        for (Integer entryIndex : inputSample.keySet()) getInputNodes().get(entryIndex).setMatrix(sampleIndex, inputSample.get(entryIndex));
    }

    /**
     * Sets output sample.
     *
     * @param sampleIndex sample index
     * @param outputSample output sample
     * @throws MatrixException throws exception if calculation fails.
     */
    private void setOutputSample(int sampleIndex, MMatrix outputSample) throws MatrixException {
        for (Integer entryIndex : outputNodes.keySet()) outputSample.put(entryIndex, outputNodes.get(entryIndex).getMatrix(sampleIndex));
    }

    /**
     * Calculates chain of forward expressions.
     *
     * @param inputMatrix input matrices.
     * @return output matrix.
     * @throws MatrixException throws exception if calculation fails.
     */
    public MMatrix calculateExpression(MMatrix inputMatrix) throws MatrixException {
        Sequence inputSequence = new Sequence(1);
        for (Integer index : inputMatrix.keySet()) inputSequence.put(index, new MMatrix(inputMatrix.get(index)));

        Sequence outputSequence = new Sequence(1);
        calculateExpressionPerStep(inputSequence, outputSequence);

        MMatrix outputMatrix = new MMatrix();
        for (Integer index : outputSequence.keySet()) outputMatrix.put(index, outputSequence.get(index, 0));
        return outputMatrix;
    }

    /**
     * Calculates chain of forward expressions.
     *
     * @param inputMatrix input matrices.
     * @return output matrix.
     * @throws MatrixException throws exception if calculation fails.
     */
    public Matrix calculateExpression(Matrix inputMatrix) throws MatrixException {
        Sequence inputSequence = new Sequence(1);
        inputSequence.put(0, 0, inputMatrix);
        Sequence outputSequence = new Sequence(1);
        calculateExpressionPerStep(inputSequence, outputSequence);
        return outputSequence.get(0,0);
    }

    /**
     * Returns regularization error.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return regularization error.
     */
    public double getRegularizationError() throws MatrixException {
        return expressionChain.cumulateRegularizationError();
    }

    /**
     * Calculates chain of forward expressions for multiple inputs per gradient expression step.
     *
     * @param outputGradientSequence output gradients.
     * @param inputGradientSequence input gradients.
     * @param steps number of steps calculated backwards.
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateGradient(Sequence outputGradientSequence, Sequence inputGradientSequence, int steps) throws MatrixException {
        if (hasDependencies()) calculateGradientPerSample(outputGradientSequence, inputGradientSequence, steps);
        else calculateGradientPerStep(outputGradientSequence, inputGradientSequence, steps);
    }

    /**
     * Calculates chain of forward expressions for multiple inputs per gradient expression step per sample.
     *
     * @param outputGradientSequence output gradients.
     * @param inputGradientSequence input gradients.
     * @param steps number of steps calculated backwards.
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateGradientPerSample(Sequence outputGradientSequence, Sequence inputGradientSequence, int steps) throws MatrixException {
        int step = 0;
        for (Integer sampleIndex : outputGradientSequence.descendingKeySet()) {
            setOutputSampleGradient(sampleIndex, outputGradientSequence.get(sampleIndex));

            gradientExpressionChain.calculateGradientStep(sampleIndex, outputGradientSequence.firstKey());

            MMatrix inputSampleGradient = new MMatrix(inputNodes.size());
            setInputSampleGradient(sampleIndex, inputSampleGradient);
            inputGradientSequence.put(sampleIndex, inputSampleGradient);

            if (steps > 0 && ++step >= steps) break;
        }
    }

    /**
     * Calculates chain of forward expressions for multiple inputs per gradient expression step.
     *
     * @param outputGradientSequence output gradients.
     * @param inputGradientSequence input gradients.
     * @param steps number of steps calculated backwards.
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateGradientPerStep(Sequence outputGradientSequence, Sequence inputGradientSequence, int steps) throws MatrixException {
        int step = 0;
        for (Integer sampleIndex : outputGradientSequence.keySet()) {
            setOutputSampleGradient(sampleIndex, outputGradientSequence.get(sampleIndex));

            if (steps > 0 && ++step >= steps) break;
        }

        gradientExpressionChain.calculateGradientStep(outputGradientSequence.keySet(), outputGradientSequence.lastKey(), steps);

        step = 0;
        for (Integer sampleIndex : outputGradientSequence.keySet()) {
            MMatrix inputSampleGradient = new MMatrix(getInputNodes().size());
            setInputSampleGradient(sampleIndex, inputSampleGradient);
            inputGradientSequence.put(sampleIndex, inputSampleGradient);

            if (steps > 0 && ++step >= steps) break;
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
        for (Integer entryIndex : outputSampleGradient.keySet()) getOutputNodes().get(entryIndex).setGradient(sampleIndex, outputSampleGradient.get(entryIndex));
    }

    /**
     * Sets input sample gradient.
     *
     * @param sampleIndex input sample
     * @param inputSampleGradient input sample gradient
     * @throws MatrixException throws exception if calculation fails.
     */
    private void setInputSampleGradient(int sampleIndex, MMatrix inputSampleGradient) throws MatrixException {
        for (Integer entryIndex : inputNodes.keySet()) inputSampleGradient.put(entryIndex, inputNodes.get(entryIndex).getGradient(sampleIndex));
    }

    /**
     * Calculates backwards chain of gradient expressions.
     *
     * @param outputGradient output gradient for procedure.
     * @return input gradient.
     * @throws MatrixException throws exception if calculation fails.
     */
    public MMatrix calculateGradient(MMatrix outputGradient) throws MatrixException {
        Sequence outputGradientSequence = new Sequence(1);
        for (Integer index : outputGradient.keySet()) outputGradientSequence.put(index, new MMatrix(outputGradient.get(index)));

        Sequence inputGradientSequence = new Sequence(1);
        calculateGradientPerStep(outputGradientSequence, inputGradientSequence, -1);

        MMatrix inputGradient = new MMatrix();
        for (Integer index : inputGradientSequence.keySet()) inputGradient.put(index, inputGradientSequence.get(index, 0));
        return inputGradient;
    }

    /**
     * Calculates backwards chain of gradient expressions.
     *
     * @param outputGradient output gradient for procedure.
     * @return input gradient.
     * @throws MatrixException throws exception if calculation fails.
     */
    public Matrix calculateGradient(Matrix outputGradient) throws MatrixException {
        Sequence outputSequence = new Sequence(1);
        outputSequence.put(0, 0, outputGradient);
        Sequence inputSequence = new Sequence(1);
        calculateGradientPerStep(outputSequence, inputSequence, -1);
        return inputSequence.get(0,0);
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
        return node.getGradient().divide(node.getEntryCount());
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
        gradientExpressionChain.printGradientChain();
    }

}
