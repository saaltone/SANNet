/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure;

import core.optimization.Optimizer;
import core.optimization.OptimizerFactory;
import utils.configurable.DynamicParamException;
import utils.matrix.AbstractMatrix;
import utils.sampling.Sequence;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.procedure.expression.Expression;
import utils.procedure.node.Node;

import java.io.Serial;
import java.io.Serializable;
import java.util.*;

/**
 * Implements computable procedure having chain of forward computable expressions and backward computable gradient expressions (based on automatic gradient).<br>
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
    private final Node outputNode;

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
     * Dependent nodes.
     *
     */
    private final HashSet<Node> dependentNodes = new HashSet<>();

    /**
     * If true input is reversed otherwise not.
     *
     */
    private final boolean reversedInput;

    /**
     * If true inputs are joined otherwise not.
     *
     */
    private final boolean joinedInput;

    /**
     * Parameter matrices.
     *
     */
    private final HashSet<Matrix> parameterMatrices;

    /**
     * Optimizer for procedure.
     *
     */
    protected Optimizer optimizer = OptimizerFactory.createDefault();

    /**
     * Constructor for procedure.
     *
     * @param inputNodes input nodes for procedure.
     * @param outputNode output node for procedure.
     * @param nodes all nodes for procedure.
     * @param expressionChain chain of expressions describing procedure.
     * @param gradientChain chain of gradients for procedure.
     * @param dependentNodes dependent nodes.
     * @param parameterMatrices parameter matrices.
     * @param stopGradientMatrices matrices for which gradient is not updated.
     * @param reversedInput reversed input.
     * @param joinedInput if true inputs are joined otherwise not.
     * @throws MatrixException throws exception if node does not contain all constant and parameter matrices.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Procedure(HashMap<Integer, Node> inputNodes, Node outputNode, HashSet<Node> nodes, Expression expressionChain, Expression gradientChain, HashSet<Node> dependentNodes, HashSet<Matrix> parameterMatrices, HashSet<Matrix> stopGradientMatrices, boolean reversedInput, boolean joinedInput) throws MatrixException, DynamicParamException {
        this.inputNodes.putAll(inputNodes);
        this.outputNode = outputNode;
        this.nodes.addAll(nodes);
        this.expressionChain = expressionChain;
        this.gradientChain = gradientChain;
        this.dependentNodes.addAll(dependentNodes);
        this.parameterMatrices = parameterMatrices;
        if (parameterMatrices != null) checkParameterMatrices();
        if (stopGradientMatrices != null) setStopGradient(stopGradientMatrices, true);
        this.reversedInput = reversedInput;
        this.joinedInput = joinedInput;
    }

    /**
     * Sets reset matrix dependencies flag.
     *
     * @param resetDependencies if true matrix dependencies are reset otherwise false.
     */
    public void resetDependencies(boolean resetDependencies) {
        for (Node dependentNode : dependentNodes) dependentNode.resetDependencies(resetDependencies);
    }

    /**
     * Resets data for every index in nodes of procedure.
     *
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public void reset() throws MatrixException {
        for (Node node : nodes) node.reset();
    }

    /**
     * Sets is procedure is active.
     *
     * @param isActive is true procedure is active otherwise non-active.
     */
    public void setActive(boolean isActive) {
        expressionChain.setActive(isActive);
        gradientChain.setActive(isActive);
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
    public Node getOutputNode() {
        return outputNode;
    }

    /**
     * Checks if procedure has dependencies between output and input nodes.
     *
     * @return returns true if there are dependencies otherwise returns false.
     */
    public boolean hasDependencies() {
        return !dependentNodes.isEmpty();
    }

    /**
     * Calculates chain of forward expressions.
     *
     * @param inputSequences input sequences.
     * @param outputSequence output sequence.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void calculateExpression(TreeMap<Integer, Sequence> inputSequences, Sequence outputSequence) throws MatrixException, DynamicParamException {
        expressionChain.reset();
        if (joinedInput) calculateExpressionForMultipleSequences(Sequence.join(inputSequences, true), outputSequence);
        else calculateExpressionForMultipleSequences(inputSequences, outputSequence);
    }

    /**
     * Calculates chain of forward expressions.
     *
     * @param inputSequences input sequences.
     * @param outputSequence output sequence.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private void calculateExpressionForMultipleSequences(TreeMap<Integer, Sequence> inputSequences, Sequence outputSequence) throws MatrixException, DynamicParamException {
        if (hasDependencies()) calculateExpressionPerSample(inputSequences, outputSequence);
        else calculateExpressionPerStep(inputSequences, outputSequence);
    }

    /**
     * Calculates chain of forward expressions sample by sample.
     *
     * @param inputSequences input sequences.
     * @param outputSequence output sequence.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private void calculateExpressionPerSample(TreeMap<Integer, Sequence> inputSequences, Sequence outputSequence) throws MatrixException, DynamicParamException {
        Sequence inputSequence = inputSequences.get(inputSequences.firstKey());

        int firstKey = reversedInput ? inputSequence.lastKey() : inputSequence.firstKey();
        Set<Integer> inputKeySet = reversedInput ? inputSequence.descendingKeySet() : inputSequence.keySet();

        int previousSampleIndex = -1;
        for (Integer sampleIndex : inputKeySet) {
            for (Node dependentNode : dependentNodes) dependentNode.updateMatrixDependency(sampleIndex, previousSampleIndex);

            setInputSamples(inputSequences, inputSequence, sampleIndex);

            expressionChain.calculateExpressionStep(sampleIndex, firstKey);

            outputSequence.put(sampleIndex, getOutputNode().getMatrix(sampleIndex));

            for (Node dependentNode : dependentNodes) dependentNode.updateDependencies(sampleIndex);

            previousSampleIndex = sampleIndex;
        }

    }

    /**
     * Calculates chain of forward expressions for all samples.
     *
     * @param inputSequences input sequences.
     * @param outputSequence output sequence.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private void calculateExpressionPerStep(TreeMap<Integer, Sequence> inputSequences, Sequence outputSequence) throws MatrixException, DynamicParamException {
        Sequence inputSequence = inputSequences.get(inputSequences.firstKey());

        Set<Integer> inputKeySet = reversedInput ? inputSequence.descendingKeySet() : inputSequence.keySet();

        for (Integer sampleIndex : inputKeySet) setInputSamples(inputSequences, inputSequence, sampleIndex);

        expressionChain.calculateExpressionStep(inputKeySet);

        for (Integer sampleIndex : inputKeySet) outputSequence.put(sampleIndex, getOutputNode().getMatrix(sampleIndex));
    }

    /**
     * Sets input samples for expression chain.
     *
     * @param inputSequences input sequences
     * @param inputSequence input sequence.
     * @param sampleIndex sample index.
     * @throws MatrixException throws exception if calculation fails.
     */
    private void setInputSamples(TreeMap<Integer, Sequence> inputSequences, Sequence inputSequence, int sampleIndex) throws MatrixException {
        if (inputSequences.size() == 1) {
            getInputNodes().get(0).setMatrix(sampleIndex, inputSequence.get(sampleIndex));
        }
        else {
            for (Map.Entry<Integer, Sequence> entry : inputSequences.entrySet()) {
                getInputNodes().get(entry.getKey()).setMatrix(sampleIndex, entry.getValue().get(sampleIndex));
            }
        }
    }

    /**
     * Calculates chain of forward expressions.
     *
     * @param inputMatrix input matrices.
     * @return output matrix.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix calculateExpression(Matrix inputMatrix) throws MatrixException, DynamicParamException {
        getInputNodes().get(0).setMatrix(0, inputMatrix);
        expressionChain.calculateExpressionStep(0, 0);
        return getOutputNode().getMatrix(0);
    }

    /**
     * Calculates chain of backward expressions for multiple inputs per gradient expression step.
     *
     * @param outputGradientSequence output gradients.
     * @param inputGradientSequences input gradients.
     * @param steps number of steps calculated backwards.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void calculateGradient(Sequence outputGradientSequence, TreeMap<Integer, Sequence> inputGradientSequences, int steps) throws MatrixException, DynamicParamException {
        if (joinedInput) {
            TreeMap <Integer, Sequence> joinedInputGradientSequences = new TreeMap<>() {{ put(0, new Sequence()); }};
            calculateGradientForMultipleInputs(outputGradientSequence, joinedInputGradientSequences, steps);
            Sequence sequence = joinedInputGradientSequences.get(0);
            for (Map.Entry<Integer, Matrix> entry : sequence.entrySet()) {
                Matrix[] matrices = AbstractMatrix.unjoin(entry.getValue());
                for (int index = 0; index < matrices.length; index++) {
                    inputGradientSequences.get(index).put(entry.getKey(), matrices[index]);
                }
            }
        }
        else calculateGradientForMultipleInputs(outputGradientSequence, inputGradientSequences, steps);
    }

    /**
     * Calculates chain of backward expressions for multiple inputs per gradient expression step.
     *
     * @param outputGradientSequence output gradient sequence.
     * @param inputGradientSequences input gradient sequences.
     * @param steps number of steps calculated backwards.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private void calculateGradientForMultipleInputs(Sequence outputGradientSequence, TreeMap<Integer, Sequence> inputGradientSequences, int steps) throws MatrixException, DynamicParamException {
        if (hasDependencies()) calculateGradientPerSample(outputGradientSequence, inputGradientSequences, steps);
        else calculateGradientPerStep(outputGradientSequence, inputGradientSequences, steps);
    }

    /**
     * Calculates chain of backward expressions for multiple inputs per gradient expression step per sample.
     *
     * @param outputGradientSequence output gradient sequence.
     * @param inputGradientSequences input gradient sequences.
     * @param numberOfGradientSteps number of steps calculated backwards.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private void calculateGradientPerSample(Sequence outputGradientSequence, TreeMap<Integer, Sequence> inputGradientSequences, int numberOfGradientSteps) throws MatrixException, DynamicParamException {
        int lastKey = reversedInput ? outputGradientSequence.lastKey() : outputGradientSequence.firstKey();

        int previousSampleIndex = -1;
        int gradientStepCount = 0;
        for (Map.Entry<Integer, Matrix> entry : reversedInput ? outputGradientSequence.entrySet() : outputGradientSequence.descendingEntrySet()) {
            int sampleIndex = entry.getKey();

            for (Node dependentNode : dependentNodes) dependentNode.updateGradientDependency(sampleIndex, previousSampleIndex);

            getOutputNode().setGradient(sampleIndex, entry.getValue());

            gradientChain.calculateGradientStep(sampleIndex, lastKey);

            for (Map.Entry<Integer, Node> nodeEntry : inputNodes.entrySet()) {
                inputGradientSequences.get(nodeEntry.getKey()).increment(sampleIndex, nodeEntry.getValue().getGradient(sampleIndex));
            }

            if (numberOfGradientSteps > 0 && ++gradientStepCount >= numberOfGradientSteps) break;

            previousSampleIndex = sampleIndex;
        }

    }

    /**
     * Calculates chain of backward expressions for multiple inputs per gradient expression step.
     *
     * @param outputGradientSequence output gradient sequence.
     * @param inputGradientSequences input gradient sequences.
     * @param numberOfGradientSteps number of steps calculated backwards.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private void calculateGradientPerStep(Sequence outputGradientSequence, TreeMap<Integer, Sequence> inputGradientSequences, int numberOfGradientSteps) throws MatrixException, DynamicParamException {
        Set<Integer> inputKeySet = reversedInput ? outputGradientSequence.keySet() : outputGradientSequence.descendingKeySet();

        int gradientStepCount = 0;
        for (Map.Entry<Integer, Matrix> entry : reversedInput ? outputGradientSequence.entrySet() : outputGradientSequence.descendingEntrySet()) {
            getOutputNode().setGradient(entry.getKey(), entry.getValue());
            if (numberOfGradientSteps > 0 && ++gradientStepCount >= numberOfGradientSteps) break;
        }

        gradientChain.calculateGradientStep(inputKeySet, numberOfGradientSteps);

        gradientStepCount = 0;
        for (Integer sampleIndex : inputKeySet) {
            for (Map.Entry<Integer, Node> entry : inputNodes.entrySet()) {
                inputGradientSequences.get(entry.getKey()).increment(sampleIndex, entry.getValue().getGradient(sampleIndex));
            }
            if (numberOfGradientSteps > 0 && ++gradientStepCount >= numberOfGradientSteps) break;
        }
    }

    /**
     * Calculates backward chain of gradient expressions.
     *
     * @param outputGradient output gradient for procedure.
     * @return input gradient.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix calculateGradient(Matrix outputGradient) throws MatrixException, DynamicParamException {
        getOutputNode().setGradient(0, outputGradient);
        gradientChain.calculateGradientStep(0, 0);
        return getInputNodes().get(0).getGradient(0);
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
                System.out.println("Failed to find parameter matrix: " + this + " " + parameterMatrix + " " + parameterMatrix.getName());
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
        return new HashMap<>() {{ putAll(getProcedureGradients()); }};
    }

    /**
     * Gets gradients for parameter matrices
     *
     * @return gradients
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private HashMap<Matrix, Matrix> getProcedureGradients() throws MatrixException {
        HashMap<Matrix, Matrix> gradients = new HashMap<>();
        if (parameterMatrices == null) return gradients;
        for (Matrix parameterMatrix : parameterMatrices) {
            Node node = getNode(parameterMatrix);
            if (node != null) gradients.put(parameterMatrix, node.getGradientMean());
        }
        return gradients;
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
     * Sets optimizer for layer.<br>
     * Optimizer optimizes weight parameters iteratively towards optimal solution.<br>
     *
     * @param optimizer optimizer to be added.
     */
    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
    }

    /**
     * Resets optimizer of layer.
     *
     */
    public void resetOptimizer() {
        optimizer.reset();
    }

    /**
     * Executes weight updates with regularizers and optimizer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void optimize() throws MatrixException, DynamicParamException {
        for (Map.Entry<Matrix, Matrix> entry : getGradients().entrySet()) optimizer.optimize(entry.getKey(), entry.getValue());
    }

    /**
     * Returns optimizer by name.
     *
     * @return optimizer by name.
     */
    public String getOptimizerByName() {
        return optimizer.getName();
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
