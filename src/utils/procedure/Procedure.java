/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package utils.procedure;

import utils.configurable.DynamicParamException;
import utils.sampling.Sequence;
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
     * If true input is reversed otherwise not.
     *
     */
    private final boolean reversedInput;

    /**
     * Parameter matrices.
     *
     */
    private final HashSet<Matrix> parameterMatrices;

    /**
     * Constructor for procedure.
     * @param inputNodes input nodes for procedure.
     * @param outputNodes input nodes for procedure.
     * @param nodes all nodes for procedure.
     * @param expressionChain chain of expressions describing procedure.
     * @param gradientChain chain of gradients for procedure.
     * @param hasDependentNodes true if procedure has dependent nodes.
     * @param parameterMatrices parameter matrices.
     * @param stopGradientMatrices matrices for which gradient is not updated.
     * @param reversedInput if true input is reversed other not.
     * @throws MatrixException throws exception if node does not contain all constant and parameter matrices.
     */
    public Procedure(HashMap<Integer, Node> inputNodes, HashMap<Integer, Node> outputNodes, HashSet<Node> nodes, Expression expressionChain, Expression gradientChain, boolean hasDependentNodes, HashSet<Matrix> parameterMatrices, HashSet<Matrix> stopGradientMatrices, boolean reversedInput) throws MatrixException {
        this.inputNodes.putAll(inputNodes);
        this.outputNodes.putAll(outputNodes);
        this.nodes.addAll(nodes);
        this.expressionChain = expressionChain;
        this.gradientChain = gradientChain;
        this.hasDependentNodes = hasDependentNodes;
        this.parameterMatrices = parameterMatrices;
        if (parameterMatrices != null) checkParameterMatrices();
        if (stopGradientMatrices != null) setStopGradient(stopGradientMatrices, true);
        this.reversedInput = reversedInput;
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
        Set<Integer> inputKeySet = reversedInput ? inputSequence.descendingKeySet() : inputSequence.keySet();
        int firstKey = reversedInput ? inputSequence.lastKey() : inputSequence.firstKey();

        for (Integer sampleIndex : inputKeySet) {
            setInputSample(sampleIndex, inputSequence.get(sampleIndex));

            expressionChain.calculateExpressionStep(sampleIndex, firstKey);

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
        Set<Integer> inputKeySet = reversedInput ? inputSequence.descendingKeySet() : inputSequence.keySet();

        for (Integer sampleIndex : inputKeySet) setInputSample(sampleIndex, inputSequence.get(sampleIndex));

        expressionChain.calculateExpressionStep(inputKeySet);

        for (Integer sampleIndex : inputKeySet) outputSequence.put(sampleIndex, setOutputSample(sampleIndex));
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
     * @param sampleIndex sample index.
     * @return output matrix.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix calculateExpression(Matrix inputMatrix, int sampleIndex) throws MatrixException, DynamicParamException {
        setInputSample(sampleIndex, new MMatrix(inputMatrix));
        expressionChain.calculateExpressionStep(sampleIndex, 0);
        return getOutputNodes().get(0).getMatrix(sampleIndex);
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
        Set<Integer> inputKeySet = reversedInput ? outputGradientSequence.keySet() : outputGradientSequence.descendingKeySet();
        int lastKey = reversedInput ? outputGradientSequence.firstKey() : outputGradientSequence.lastKey();

        int gradientStepCount = 0;
        for (Integer sampleIndex : inputKeySet) {
            setOutputSampleGradient(sampleIndex, outputGradientSequence.get(sampleIndex));

            gradientChain.calculateGradientStep(sampleIndex, lastKey);

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
        Set<Integer> inputKeySet = reversedInput ? outputGradientSequence.keySet() : outputGradientSequence.descendingKeySet();

        int gradientStepCount = 0;
        for (Integer sampleIndex : inputKeySet) {
            setOutputSampleGradient(sampleIndex, outputGradientSequence.get(sampleIndex));
            if (numberOfGradientSteps > 0 && ++gradientStepCount >= numberOfGradientSteps) break;
        }

        gradientChain.calculateGradientStep(inputKeySet, numberOfGradientSteps);

        gradientStepCount = 0;
        for (Integer sampleIndex : inputKeySet) {
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
     * Check that procedure contains all parameter matrices.
     *
     * @throws MatrixException throws exception if node does not contain all parameter matrices.
     */
    private void checkParameterMatrices() throws MatrixException {
        for (Matrix parameterMatrix : parameterMatrices) {
            boolean containsParameterMatrix = false;
            for (Node node : nodes) {
                if (node.isReferenceOf(parameterMatrix)) {
                    containsParameterMatrix = true;
                    break;
                }
            }
            if (!containsParameterMatrix) {
                System.out.println("Fail: " + this + " " + parameterMatrix + " " + parameterMatrix.getName());
                throw new MatrixException("Procedure does not contain all parameter matrices.");
            }
        }
    }

    /**
     * Gets gradients for parameter matrices
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @return gradients
     */
    public HashMap<Matrix, Matrix> getGradients() throws MatrixException {
        HashMap<Matrix, Matrix> gradients = new HashMap<>();
        getGradients(gradients);
        return gradients;
    }

    /**
     * Gets gradients for parameter matrices
     *
     * @param gradients gradients
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void getGradients(HashMap<Matrix, Matrix> gradients) throws MatrixException {
        if (parameterMatrices == null) return;
        for (Matrix parameterMatrix : parameterMatrices) {
            Node node = getNode(parameterMatrix);
            if (node != null) gradients.put(parameterMatrix, node.getGradientMean());
        }
    }

    /**
     * Sets if gradient is updated for nodes of this expression. If true gradient is not updated otherwise it is updated.
     *
     * @param referenceMatrices reference matrices of nodes.
     * @param stopGradient if true gradient is not updated otherwise it is updated.
     * @throws MatrixException throws exception if procedure does not contain reference matrix.
     */
    public void setStopGradient(HashSet<Matrix> referenceMatrices, boolean stopGradient) throws MatrixException {
        for (Matrix referenceMatrix : referenceMatrices) setStopGradient(referenceMatrix, stopGradient);
    }

    /**
     * Sets if gradient is updated for nodes of this expression. If true gradient is not updated otherwise it is updated.
     *
     * @param referenceMatrix reference matrix of node.
     * @param stopGradient if true gradient is not updated otherwise it is updated.
     * @throws MatrixException throws exception if procedure does not contain reference matrix.
     */
    public void setStopGradient(Matrix referenceMatrix, boolean stopGradient) throws MatrixException {
        boolean containsReferenceMatrix = false;
        for (Node node : nodes) {
            if (node.isReferenceOf(referenceMatrix)) {
                node.setStopGradient(stopGradient);
                containsReferenceMatrix = true;
            }
        }
        if (!containsReferenceMatrix) throw new MatrixException("Procedure does not contain reference matrix.");
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
