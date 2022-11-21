/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
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
 * Defines computable procedure having chain of forward computable expressions and backward computable gradient expressions (based on automatic gradient).<br>
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
     * Reversed procedure instance.
     *
     */
    private final Procedure reversedProcedure;

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
     * Constructor for procedure.
     * @param inputNodes input nodes for procedure.
     * @param outputNodes input nodes for procedure.
     * @param nodes all nodes for procedure.
     * @param expressionChain chain of expressions describing procedure.
     * @param gradientChain chain of gradients for procedure.
     * @param hasDependentNodes true if procedure has dependent nodes.
     * @param parameterMatrices parameter matrices.
     * @param stopGradientMatrices matrices for which gradient is not updated.
     * @param reversedProcedure reversedProcedure.
     * @param joinedInput if true inputs are joined otherwise not.
     * @throws MatrixException throws exception if node does not contain all constant and parameter matrices.
     */
    public Procedure(HashMap<Integer, Node> inputNodes, HashMap<Integer, Node> outputNodes, HashSet<Node> nodes, Expression expressionChain, Expression gradientChain, boolean hasDependentNodes, HashSet<Matrix> parameterMatrices, HashSet<Matrix> stopGradientMatrices, Procedure reversedProcedure, boolean joinedInput) throws MatrixException {
        this.inputNodes.putAll(inputNodes);
        this.outputNodes.putAll(outputNodes);
        this.nodes.addAll(nodes);
        this.expressionChain = expressionChain;
        this.gradientChain = gradientChain;
        this.hasDependentNodes = hasDependentNodes;
        this.parameterMatrices = parameterMatrices;
        if (parameterMatrices != null) checkParameterMatrices();
        if (stopGradientMatrices != null) setStopGradient(stopGradientMatrices, true);
        this.reversedInput = reversedProcedure != null;
        this.reversedProcedure = reversedProcedure;
        this.joinedInput = joinedInput;
    }

    /**
     * Resets data for every index in nodes of procedure.
     *
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public void reset() throws MatrixException {
        for (Node node : nodes) node.reset();
        expressionChain.reset();
        if (reversedProcedure != null) reversedProcedure.reset();
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
     * Calculates chain of forward expressions.
     *
     * @param inputSequence input sequence.
     * @param inputSequences input sequences.
     * @return output sequence.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Sequence calculateExpression(Sequence inputSequence, HashMap<Integer, Sequence> inputSequences) throws MatrixException, DynamicParamException {
        Sequence outputSequence = new Sequence();
        calculateExpression(inputSequences, inputSequence, outputSequence);
        return outputSequence;
    }

    /**
     * Calculates chain of forward expressions.
     *
     * @param inputSequences input sequences.
     * @param inputSequence input sequence.
     * @param outputSequence output sequence.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void calculateExpression(HashMap<Integer, Sequence> inputSequences, Sequence inputSequence, Sequence outputSequence) throws MatrixException, DynamicParamException {
        if (reversedProcedure == null) {
            if (inputSequences.size() == 1) {
                calculateExpression(inputSequence, outputSequence);
            }
            else {
                if (joinedInput) calculateExpression(Sequence.join(inputSequences, true), outputSequence);
                else calculateExpression(inputSequences, outputSequence);
            }
        }
        else {
            if (inputSequences.size() == 1) {
                outputSequence.putAll(Sequence.join(new Sequence[] { calculateExpression(inputSequence), reversedProcedure.calculateExpression(inputSequence) }, true));
            }
            else {
                if (joinedInput) {
                    Sequence joinedInputSequence = Sequence.join(inputSequences, true);
                    outputSequence.putAll(Sequence.join(new Sequence[] { calculateExpression(joinedInputSequence), reversedProcedure.calculateExpression(joinedInputSequence) }, true));
                }
                else {
                    outputSequence.putAll(Sequence.join(new Sequence[] { calculateExpression(inputSequences), reversedProcedure.calculateExpression(inputSequences) }, true));
                }
            }
        }
    }

    /**
     * Calculates chain of forward expressions.
     *
     * @param inputSequence input sequence.
     * @return output sequence.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Sequence calculateExpression(Sequence inputSequence) throws MatrixException, DynamicParamException {
        Sequence outputSequence = new Sequence();
        if (hasDependencies()) calculateExpressionPerSample(inputSequence, outputSequence);
        else calculateExpressionPerStep(inputSequence, outputSequence);
        return outputSequence;
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
    private void calculateExpressionPerSample(Sequence inputSequence, Sequence outputSequence) throws MatrixException, DynamicParamException {
        int firstKey = reversedInput ? inputSequence.lastKey() : inputSequence.firstKey();

        for (Map.Entry<Integer, MMatrix> entry : reversedInput ? inputSequence.descendingEntrySet() : inputSequence.entrySet()) {
            int sampleIndex = entry.getKey();
            MMatrix inputSample = entry.getValue();
            setInputSample(sampleIndex, inputSample, getInputNodes());

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
    private void calculateExpressionPerStep(Sequence inputSequence, Sequence outputSequence) throws MatrixException, DynamicParamException {
        Set<Integer> inputKeySet = reversedInput ? inputSequence.descendingKeySet() : inputSequence.keySet();

        for (Map.Entry<Integer, MMatrix> entry : reversedInput ? inputSequence.descendingEntrySet() : inputSequence.entrySet()) {
            int sampleIndex = entry.getKey();
            MMatrix inputSample = entry.getValue();
            setInputSample(sampleIndex, inputSample, getInputNodes());
        }

        expressionChain.calculateExpressionStep(inputKeySet);

        for (Integer sampleIndex : inputKeySet) {
            outputSequence.put(sampleIndex, setOutputSample(sampleIndex));
        }
    }

    /**
     * Sets input sample.
     *
     * @param sampleIndex sample index
     * @param inputSample input sample
     * @param inputNodes input nodes
     * @throws MatrixException throws exception if calculation fails.
     */
    private void setInputSample(int sampleIndex, MMatrix inputSample, HashMap<Integer, Node> inputNodes) throws MatrixException {
        int depth = inputSample.getDepth();
        for (int depthIndex = 0; depthIndex < depth; depthIndex++) {
            setInputSample(sampleIndex, depthIndex, inputSample, inputNodes.get(depthIndex));
        }
    }

    /**
     * Calculates chain of forward expressions.
     *
     * @param inputSequences input sequences.
     * @return output sequence.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Sequence calculateExpression(HashMap<Integer, Sequence> inputSequences) throws MatrixException, DynamicParamException {
        Sequence outputSequence = new Sequence();
        calculateExpression(inputSequences, outputSequence);
        return outputSequence;
    }

    /**
     * Calculates chain of forward expressions.
     *
     * @param inputSequences input sequences.
     * @param outputSequence output sequence.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void calculateExpression(HashMap<Integer, Sequence> inputSequences, Sequence outputSequence) throws MatrixException, DynamicParamException {
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
    private void calculateExpressionPerSample(HashMap<Integer, Sequence> inputSequences, Sequence outputSequence) throws MatrixException, DynamicParamException {
        for (Map.Entry<Integer, Sequence> entry : inputSequences.entrySet()) {
            int depthIndex = entry.getKey();
            Sequence inputSequence = entry.getValue();
            int firstKey = reversedInput ? inputSequence.lastKey() : inputSequence.firstKey();

            for (Map.Entry<Integer, MMatrix> entry1 : reversedInput ? inputSequence.descendingEntrySet() : inputSequence.entrySet()) {
                int sampleIndex = entry1.getKey();
                MMatrix inputSample = entry1.getValue();
                setInputSample(sampleIndex, depthIndex, inputSample, getInputNodes().get(depthIndex));

                expressionChain.calculateExpressionStep(sampleIndex, firstKey);

                outputSequence.put(sampleIndex, setOutputSample(sampleIndex));
            }
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
    private void calculateExpressionPerStep(HashMap<Integer, Sequence> inputSequences, Sequence outputSequence) throws MatrixException, DynamicParamException {
        Set<Integer> inputKeySet = null;
        for (Map.Entry<Integer, Sequence> entry : inputSequences.entrySet()) {
            int depthIndex = entry.getKey();
            Sequence inputSequence = entry.getValue();

            inputKeySet = reversedInput ? inputSequence.descendingKeySet() : inputSequence.keySet();

            for (Map.Entry<Integer, MMatrix> entry1 : reversedInput ? inputSequence.descendingEntrySet() : inputSequence.entrySet()) {
                int sampleIndex = entry1.getKey();
                MMatrix inputSample = entry1.getValue();
                setInputSample(sampleIndex, 0, inputSample, getInputNodes().get(depthIndex));
            }

        }

        if (inputKeySet != null) {
            expressionChain.calculateExpressionStep(inputKeySet);

            for (Integer sampleIndex : inputKeySet) {
                outputSequence.put(sampleIndex, setOutputSample(sampleIndex));
            }
        }

    }

    /**
     * Sets input sample.
     *
     * @param sampleIndex sample index
     * @param depthIndex depth index
     * @param inputSample input sample
     * @param inputNode input node
     * @throws MatrixException throws exception if calculation fails.
     */
    private void setInputSample(int sampleIndex, int depthIndex, MMatrix inputSample, Node inputNode) throws MatrixException {
        Matrix inputSampleEntry = inputSample.get(depthIndex);
        inputNode.setMatrix(sampleIndex, inputSampleEntry);
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
        for (Map.Entry<Integer, Node> entry : getOutputNodes().entrySet()) {
            int nodeIndex = entry.getKey();
            Node node = entry.getValue();
            outputSample.put(nodeIndex, node.getMatrix(sampleIndex));
        }
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
        setInputSample(sampleIndex, new MMatrix(inputMatrix), getInputNodes());
        expressionChain.calculateExpressionStep(sampleIndex, 0);
        return getOutputNodes().get(0).getMatrix(sampleIndex);
    }

    /**
     * Calculates chain of backward expressions for multiple inputs per gradient expression step.
     *
     * @param outputGradientSequence output gradients.
     * @param inputGradientSequences input gradients.
     * @param steps number of steps calculated backwards.
     * @return input gradient.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Sequence calculateGradient(HashMap<Integer, Sequence> inputGradientSequences, Sequence outputGradientSequence, int steps) throws MatrixException, DynamicParamException {
        Sequence inputGradientSequence = new Sequence();
        calculateGradient(outputGradientSequence, inputGradientSequences, inputGradientSequence, steps);
        return inputGradientSequence;
    }

    /**
     * Calculates chain of backward expressions for multiple inputs per gradient expression step.
     *
     * @param outputGradientSequence output gradients.
     * @param inputGradientSequences input gradients.
     * @param inputGradientSequence input gradient.
     * @param steps number of steps calculated backwards.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void calculateGradient(Sequence outputGradientSequence, HashMap<Integer, Sequence> inputGradientSequences, Sequence inputGradientSequence, int steps) throws MatrixException, DynamicParamException {
        if (reversedProcedure == null) {
            if (inputGradientSequences.size() == 1) {
                calculateGradient(outputGradientSequence, inputGradientSequence, steps);
            }
            else {
                if (joinedInput) Sequence.unjoinAsMap(calculateGradient(outputGradientSequence, steps), inputGradientSequences);
                else calculateGradient(outputGradientSequence, inputGradientSequences, steps);
            }
        }
        else {
            Sequence[] unjoinedOutputGradientSequence = Sequence.unjoin(outputGradientSequence);
            if (inputGradientSequences.size() == 1) {
                inputGradientSequence.putAll(Sequence.merge(calculateGradient(unjoinedOutputGradientSequence[0], steps), reversedProcedure.calculateGradient(unjoinedOutputGradientSequence[1], steps)));
            }
            else {
                if (joinedInput) {
                    Sequence.unjoinAsMap(Sequence.merge(calculateGradient(unjoinedOutputGradientSequence[0], steps), reversedProcedure.calculateGradient(unjoinedOutputGradientSequence[1], steps)), inputGradientSequences);
                }
                else {
                    calculateGradient(unjoinedOutputGradientSequence[0], inputGradientSequences, steps);
                    reversedProcedure.calculateGradient(unjoinedOutputGradientSequence[1], inputGradientSequences, steps);
                }
            }
        }
    }

    /**
     * Calculates chain of backward expressions for multiple inputs per gradient expression step.
     *
     * @param outputGradientSequence output gradients.
     * @param steps number of steps calculated backwards.
     * @return input gradients.
     * @throws MatrixException throws exception if calculation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Sequence calculateGradient(Sequence outputGradientSequence, int steps) throws MatrixException, DynamicParamException {
        Sequence inputGradientSequence = new Sequence();
        if (hasDependencies()) calculateGradientPerSample(outputGradientSequence, inputGradientSequence, steps);
        else calculateGradientPerStep(outputGradientSequence, inputGradientSequence, steps);
        return inputGradientSequence;
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
    private void calculateGradientPerSample(Sequence outputGradientSequence, Sequence inputGradientSequence, int numberOfGradientSteps) throws MatrixException, DynamicParamException {
        int lastKey = reversedInput ? outputGradientSequence.firstKey() : outputGradientSequence.lastKey();

        int gradientStepCount = 0;
        for (Map.Entry<Integer, MMatrix> entry : reversedInput ? outputGradientSequence.descendingEntrySet() : outputGradientSequence.entrySet()) {
            int sampleIndex = entry.getKey();
            MMatrix outputGradientSample = entry.getValue();
            setOutputSampleGradient(sampleIndex, outputGradientSample);

            gradientChain.calculateGradientStep(sampleIndex, lastKey);

            inputGradientSequence.increment(sampleIndex, setInputSampleGradient(sampleIndex, getInputNodes()));

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
    private void calculateGradientPerStep(Sequence outputGradientSequence, Sequence inputGradientSequence, int numberOfGradientSteps) throws MatrixException, DynamicParamException {
        Set<Integer> inputKeySet = reversedInput ? outputGradientSequence.keySet() : outputGradientSequence.descendingKeySet();

        int gradientStepCount = 0;
        for (Map.Entry<Integer, MMatrix> entry : reversedInput ? outputGradientSequence.descendingEntrySet() : outputGradientSequence.entrySet()) {
            int sampleIndex = entry.getKey();
            MMatrix outputGradientSample = entry.getValue();
            setOutputSampleGradient(sampleIndex, outputGradientSample);
            if (numberOfGradientSteps > 0 && ++gradientStepCount >= numberOfGradientSteps) break;
        }

        gradientChain.calculateGradientStep(inputKeySet, numberOfGradientSteps);

        gradientStepCount = 0;
        for (Integer sampleIndex : inputKeySet) {
            inputGradientSequence.increment(sampleIndex, setInputSampleGradient(sampleIndex, getInputNodes()));
            if (numberOfGradientSteps > 0 && ++gradientStepCount >= numberOfGradientSteps) break;
        }
    }

    /**
     * Sets output sample gradient.
     *
     * @param sampleIndex sample index
     * @param outputSampleGradient output sample gradient
     */
    private void setOutputSampleGradient(int sampleIndex, MMatrix outputSampleGradient) {
        int depth = outputSampleGradient.getDepth();
        for (int depthIndex = 0; depthIndex < depth; depthIndex++) {
            setOutputSampleGradient(sampleIndex, depthIndex, outputSampleGradient, getOutputNodes().get(depthIndex));
        }
    }

    /**
     * Sets output sample gradient.
     *
     * @param sampleIndex sample index
     * @param depthIndex depth index
     * @param outputSampleGradient output sample gradient
     * @param outputNode output node
     */
    private void setOutputSampleGradient(int sampleIndex, int depthIndex, MMatrix outputSampleGradient, Node outputNode) {
        Matrix outputSampleGradientEntry = outputSampleGradient.get(depthIndex);
        outputNode.setGradient(sampleIndex, outputSampleGradientEntry);
    }

    /**
     * Sets input sample gradient.
     *
     * @param sampleIndex input sample
     * @param inputNodes input nodes
     * @return inputSampleGradient input sample gradient
     * @throws MatrixException throws exception if calculation fails.
     */
    private MMatrix setInputSampleGradient(int sampleIndex, HashMap<Integer, Node> inputNodes) throws MatrixException {
        MMatrix inputSampleGradient = new MMatrix(getInputNodes().size());
        for (Map.Entry<Integer, Node> entry : inputNodes.entrySet()) {
            int nodeIndex = entry.getKey();
            Node node = entry.getValue();
            inputSampleGradient.put(nodeIndex, node.getGradient(sampleIndex));
        }
        return inputSampleGradient;
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
    public void calculateGradient(Sequence outputGradientSequence, HashMap<Integer, Sequence> inputGradientSequences, int steps) throws MatrixException, DynamicParamException {
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
    private void calculateGradientPerSample(Sequence outputGradientSequence, HashMap<Integer, Sequence> inputGradientSequences, int numberOfGradientSteps) throws MatrixException, DynamicParamException {
        int lastKey = reversedInput ? outputGradientSequence.firstKey() : outputGradientSequence.lastKey();

        int gradientStepCount = 0;
        for (Map.Entry<Integer, MMatrix> entry : reversedInput ? outputGradientSequence.descendingEntrySet() : outputGradientSequence.entrySet()) {
            int sampleIndex = entry.getKey();
            MMatrix outputGradientSample = entry.getValue();
            setOutputSampleGradient(sampleIndex, outputGradientSample);

            gradientChain.calculateGradientStep(sampleIndex, lastKey);

            setInputSampleGradient(sampleIndex, inputNodes, inputGradientSequences);

            if (numberOfGradientSteps > 0 && ++gradientStepCount >= numberOfGradientSteps) break;
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
    private void calculateGradientPerStep(Sequence outputGradientSequence, HashMap<Integer, Sequence> inputGradientSequences, int numberOfGradientSteps) throws MatrixException, DynamicParamException {
        Set<Integer> inputKeySet = reversedInput ? outputGradientSequence.keySet() : outputGradientSequence.descendingKeySet();

        int gradientStepCount = 0;
        for (Map.Entry<Integer, MMatrix> entry : reversedInput ? outputGradientSequence.descendingEntrySet() : outputGradientSequence.entrySet()) {
            int sampleIndex = entry.getKey();
            MMatrix outputGradientSample = entry.getValue();
            setOutputSampleGradient(sampleIndex, outputGradientSample);
            if (numberOfGradientSteps > 0 && ++gradientStepCount >= numberOfGradientSteps) break;
        }

        gradientChain.calculateGradientStep(inputKeySet, numberOfGradientSteps);

        gradientStepCount = 0;
        for (Integer sampleIndex : inputKeySet) {
            setInputSampleGradient(sampleIndex, inputNodes, inputGradientSequences);
            if (numberOfGradientSteps > 0 && ++gradientStepCount >= numberOfGradientSteps) break;
        }
    }

    /**
     * Sets input sample gradient.
     *
     * @param sampleIndex input sample
     * @param inputNodes input nodes
     * @throws MatrixException throws exception if calculation fails.
     */
    private void setInputSampleGradient(int sampleIndex, HashMap<Integer, Node> inputNodes, HashMap<Integer, Sequence> inputGradientSequences) throws MatrixException {
        for (Map.Entry<Integer, Node> entry : inputNodes.entrySet()) {
            int nodeIndex = entry.getKey();
            Node node = entry.getValue();
            inputGradientSequences.get(nodeIndex).increment(sampleIndex, new MMatrix(node.getGradient(sampleIndex)));
        }
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
        if (reversedProcedure != null) gradients.putAll(reversedProcedure.getGradients());
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
