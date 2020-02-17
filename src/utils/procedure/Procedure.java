/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package utils.procedure;

import utils.Sample;
import utils.Sequence;
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
     * List of expressions for forward calculation.
     *
     */
    private final LinkedList<AbstractExpression> expressions;

    /**
     * List of expressions for backward gradient calculation.
     *
     */
    private final LinkedList<AbstractExpression> gradientExpressions;

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
     * @param inputNodes input nodes for procedure.
     * @param outputNodes input nodes for procedure.
     * @param expressions expressions for forward calculation.
     * @param gradientExpressions gradient expressions for backward gradient calculation.
     * @param dependentNodes node dependencies as node links for output input pair updates.
     * @param registeredMatrixMap map of registered matrices.
     */
    public Procedure(HashMap<Integer, Node> inputNodes, HashMap<Integer, Node> outputNodes, LinkedList<AbstractExpression> expressions, LinkedList<AbstractExpression> gradientExpressions, HashSet<NodeLink> dependentNodes, HashMap<Matrix, Node> registeredMatrixMap) {
        this.inputNodes.putAll(inputNodes);
        this.outputNodes.putAll(outputNodes);
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
     */
    public void reset() {
        for (AbstractExpression expression : expressions) expression.resetExpression();
    }

    /**
     * Resets data for specific index in nodes of procedure.
     *
     * @param index data index is node.
     * @throws MatrixException throws exception if reset operation fails.
     */
    public void reset(int index) throws MatrixException {
        for (AbstractExpression expression : expressions) expression.resetExpression(index);
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
    public AbstractExpression getExpression(int expressionID) {
        return expressions.get(expressionID);
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
     * Calculates chain of forward expressions for multiple inputs.
     *
     * @param inputSequence input sequence.
     * @param outputSequence output sequence.
     * @param reset if true removes procedure data after calculating each index.
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpression(Sequence inputSequence, Sequence outputSequence, boolean reset) throws MatrixException {
        if (hasDependencies()) calculateExpressionPerSample(inputSequence, outputSequence, reset);
        else calculateExpressionPerStep(inputSequence, outputSequence, reset);
    }

    /**
     * Calculates chain of forward expressions for multiple inputs.
     *
     * @param inputSequence input sequence.
     * @param outputSequence output sequence.
     * @param reset if true removes procedure data after calculating each index.
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpressionPerSample(Sequence inputSequence, Sequence outputSequence, boolean reset) throws MatrixException {
        boolean constantCallbackExecuted = false;
        for (Integer sampleIndex : inputSequence.keySet()) {
            if (hasDependencies()) updateDependencies(sampleIndex);

            Sample inputSample = inputSequence.get(sampleIndex);

            for (Integer entryIndex : inputSample.keySet()) getInputNodes().get(entryIndex).setMatrix(sampleIndex, inputSample.get(entryIndex));

            for (AbstractExpression expression : expressions) {
                if (!constantCallbackExecuted) expression.forwardCallbackConstant();
                expression.forwardCallback(sampleIndex);
                expression.calculateExpression(sampleIndex);
            }
            constantCallbackExecuted = true;

            Sample outputSample = new Sample(getOutputNodes().size());
            for (Integer entryIndex : outputNodes.keySet()) outputSample.put(entryIndex, outputNodes.get(entryIndex).getMatrix(sampleIndex));
            outputSequence.put(sampleIndex, outputSample);

            if (reset) reset(sampleIndex);
        }
    }

    /**
     * Calculates chain of forward expressions for multiple inputs per expression step.
     *
     * @param inputSequence input sequence.
     * @param outputSequence output sequence.
     * @param reset if true removes procedure data after calculating each index.
     * @throws MatrixException throws exception if calculation fails.
     */
    public void calculateExpressionPerStep(Sequence inputSequence, Sequence outputSequence, boolean reset) throws MatrixException {
        for (Integer sampleIndex : inputSequence.keySet()) {
            Sample inputSample = inputSequence.get(sampleIndex);
            for (Integer entryIndex : inputSample.keySet()) getInputNodes().get(entryIndex).setMatrix(sampleIndex, inputSample.get(entryIndex));
        }

        for (AbstractExpression expression : expressions) {
            expression.forwardCallbackConstant();
            expression.forwardCallback();
            for (Integer sampleIndex : inputSequence.keySet()) {
                expression.forwardCallback(sampleIndex);
                expression.calculateExpression(sampleIndex);
            }
        }

        for (Integer sampleIndex : inputSequence.keySet()) {
            Sample outputSample = new Sample(getOutputNodes().size());
            for (Integer entryIndex : outputNodes.keySet()) outputSample.put(entryIndex, outputNodes.get(entryIndex).getMatrix(sampleIndex));
            outputSequence.put(sampleIndex, outputSample);
            if (reset) reset(sampleIndex);
        }

    }

    /**
     * Calculates chain of forward expressions.
     *
     * @param sampleIndex specific sample index.
     * @param inputMatrix input matrices.
     * @return output matrix.
     * @throws MatrixException throws exception if calculation fails.
     */
    public Matrix calculateExpression(int sampleIndex, Matrix inputMatrix) throws MatrixException {
        if (hasDependencies()) updateDependencies(sampleIndex);

        getInputNodes().get(0).setMatrix(sampleIndex, inputMatrix);

        for (AbstractExpression expression : expressions) expression.calculateExpression(sampleIndex);

        return outputNodes.get(0).getMatrix(sampleIndex);
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
    private void updateDependencies(int index) throws MatrixException {
        for (NodeLink nodeLink : dependentNodes) nodeLink.updateExpression(index);
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

        boolean constantCallbackExecuted = false;
        for (Integer sampleIndex : outputGradientSequence.descendingKeySet()) {
            Sample outputGradientSample = outputGradientSequence.get(sampleIndex);

            for (Integer entryIndex : outputGradientSample.keySet()) getOutputNodes().get(entryIndex).setGradient(sampleIndex, outputGradientSample.get(entryIndex));
            updateGradientDependencies(sampleIndex);

            for (AbstractExpression expression : gradientExpressions) {
                expression.calculateGradient(sampleIndex);
                expression.backwardCallback(sampleIndex);
                if (!constantCallbackExecuted) expression.backwardCallbackConstant();
            }
            constantCallbackExecuted = true;

            Sample inputGradientSample = new Sample(inputNodes.size());
            for (Integer entryIndex : inputNodes.keySet()) inputGradientSample.put(entryIndex, inputNodes.get(entryIndex).getGradient(sampleIndex));
            inputGradientSequence.put(sampleIndex, inputGradientSample);

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
            Sample outputGradientSample = outputGradientSequence.get(sampleIndex);
            for (Integer entryIndex : outputGradientSample.keySet()) getOutputNodes().get(entryIndex).setGradient(sampleIndex, outputGradientSample.get(entryIndex));
            if (steps > 0 && ++step >= steps) break;
        }

        for (AbstractExpression expression : gradientExpressions) {
            step = 0;
            for (Integer sampleIndex : outputGradientSequence.keySet()) {
                expression.calculateGradient(sampleIndex);
                expression.backwardCallback(sampleIndex);
                if (steps > 0 && ++step >= steps) break;
            }
            expression.backwardCallback();
            expression.backwardCallbackConstant();
        }

        step = 0;
        for (Integer sampleIndex : outputGradientSequence.keySet()) {
            Sample inputGradientSample = new Sample(getInputNodes().size());
            for (Integer entryIndex : inputNodes.keySet()) inputGradientSample.put(entryIndex, inputNodes.get(entryIndex).getGradient(sampleIndex));
            inputGradientSequence.put(sampleIndex, inputGradientSample);
            if (steps > 0 && ++step >= steps) break;
        }

    }

    /**
     * Calculates backwards chain of gradient expressions.
     *
     * @param sampleIndex specific sample index.
     * @param outputGradient output gradient for procedure.
     * @return input gradient.
     * @throws MatrixException throws exception if calculation fails.
     */
    public Matrix calculateGradient(int sampleIndex, Matrix outputGradient) throws MatrixException {
        getOutputNodes().get(0).setGradient(sampleIndex, outputGradient);

        if (hasDependencies()) updateGradientDependencies(sampleIndex);

        for (AbstractExpression expression : gradientExpressions) expression.calculateGradient(sampleIndex);

        return inputNodes.get(0).getGradient(sampleIndex);
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
            AbstractExpression expression = (AbstractExpression)iterator.next();
            expression.printExpression();
            if (iterator.hasNext()) System.out.print(" -> ");
        }
        System.out.println();
    }

}
