/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.layer.regularization;

import core.layer.AbstractExecutionLayer;
import core.layer.NeuralNetworkLayer;
import core.layer.OutputLayer;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

import java.util.HashMap;
import java.util.HashSet;

/**
 * Implements L2 (ridge) regularization.<br>
 * <br>
 * Reference: https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c<br>
 *
 */
public class L2_Regularization extends AbstractExecutionLayer {

    /**
     * Parameter name types for L2_Regularization.
     *     - lambda: lambda value for regularization. Default value: 0.01.<br>
     *
     */
    private final static String paramNameTypes = "(lambda:DOUBLE)";

    /**
     * Regularization rate.
     *
     */
    private double lambda;

    /**
     * Regularized weight of next layer.
     *
     */
    private final HashSet<Matrix> layerRegularizedWeights = new HashSet<>();

    /**
     * Constructor for L2 regularization layer.
     *
     * @param layerIndex layer Index.
     * @param initialization initialization function for weight.
     * @param params parameters for feedforward layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public L2_Regularization(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException {
        super (layerIndex, initialization, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        lambda = 0.01;
    }

    /**
     * Returns parameters used for L2 regularization layer.
     *
     * @return parameters used for L2 regularization layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + L2_Regularization.paramNameTypes;
    }

    /**
     * Sets parameters used for L2 regularization.<br>
     * <br>
     * Supported parameters are:<br>
     *     - lambda: lambda value for regularization. Default value: 0.01.<br>
     *
     * @param params parameters used for L2 regularization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("lambda")) lambda = params.getValueAsDouble("lambda");
    }

    /**
     * Checks if layer is recurrent layer type.
     *
     * @return always false.
     */
    public boolean isRecurrentLayer() { return false; }

    /**
     * Checks if layer is convolutional layer type.
     *
     * @return always false.
     */
    public boolean isConvolutionalLayer() { return false; }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     */
    public MMatrix getInputMatrices(boolean resetPreviousInput) {
        return null;
    }

    /**
     * Defines layer procedure for forward and backward calculation (automatic gradient) by applying procedure factory.<br>
     *
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     */
    protected void defineProcedure() throws NeuralNetworkException {
        if (!(getNextLayer() instanceof OutputLayer)) throw new NeuralNetworkException("L2 Regularization must be final layer prior output layer.");

        NeuralNetworkLayer previousNeuralNetworkLayer = getPreviousLayer();
        while (previousNeuralNetworkLayer != null) {
            HashSet<Matrix> regularizedWeights = previousNeuralNetworkLayer.getRegularizedWeights();
            if (regularizedWeights != null) layerRegularizedWeights.addAll(regularizedWeights);
            previousNeuralNetworkLayer = previousNeuralNetworkLayer.getPreviousLayer();
        }
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     */
    public MMatrix getForwardProcedure() {
        return null;
    }

    /**
     * Returns matrices for which gradient is not calculated.
     *
     * @return matrices for which gradient is not calculated.
     */
    protected HashSet<Matrix> getStopGradients() {
        return new HashSet<>();
    }

    /**
     * Returns constant matrices.
     *
     * @return constant matrices.
     */
    protected HashSet<Matrix> getConstantMatrices() {
        return new HashSet<>();
    }

    /**
     * Takes single forward processing step to process layer input(s).<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forwardProcess() throws MatrixException {
        resetLayerOutputs();
        getLayerOutputs().putAll(getPreviousLayerOutputs());
    }

    /**
     * Takes single backward processing step to process layer output gradient(s) towards input.<br>
     * Applies automated backward (automatic gradient) procedure when relevant to layer.<br>
     * Additionally applies any regularization defined for layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backwardProcess() throws MatrixException {
        resetLayerGradients();
        getLayerGradients().putAll(getNextLayerGradients());

        HashMap<Matrix, Matrix> layerWeightGradients = new HashMap<>();

        NeuralNetworkLayer previousNeuralNetworkLayer = getPreviousLayer();
        while (previousNeuralNetworkLayer != null) {
            HashMap<Matrix, Matrix> currentLayerWeightGradients = previousNeuralNetworkLayer.getLayerWeightGradients();
            if (currentLayerWeightGradients != null) layerWeightGradients.putAll(currentLayerWeightGradients);
            previousNeuralNetworkLayer = previousNeuralNetworkLayer.getPreviousLayer();
        }

        for (Matrix weight : layerRegularizedWeights) {
            if (layerWeightGradients.containsKey(weight)) {
                Matrix weightGradientSum = layerWeightGradients.get(weight);
                weightGradientSum.add(weight.multiply(2 * lambda), weightGradientSum);
            }
        }
    }

    /**
     * Executes weight updates with regularizers and optimizer.
     *
     */
    public void optimize() {
    }

    /**
     * Cumulates error from (L1 / L2 / Lp) regularization.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return cumulated error from regularization.
     */
    public double error() throws MatrixException, DynamicParamException {
        double weightSum = 0;
        for (Matrix weight : layerRegularizedWeights) {
            weightSum += weight.power(2).sum();
        }
        return lambda * weightSum;
    }

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        return "Lambda: " + lambda;
    }

    /**
     * Prints forward expression chains of layer.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    public void printExpressions() throws NeuralNetworkException {
        System.out.println(getLayerName() + ": ");
        System.out.println();
    }

    /**
     * Prints backward gradient chains of layer.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    public void printGradients() throws NeuralNetworkException {
        System.out.println(getLayerName() + ": ");
        System.out.println();
    }

}
