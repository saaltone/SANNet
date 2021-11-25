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
import utils.matrix.Initialization;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashMap;
import java.util.HashSet;

/**
 * Implements Lp regularization (experimental). P here is any norm higher or equal to 1.<br>
 * Setting p = 1 this becomes L1 regularization and setting p = 2 this becomes L2 regularization.<br>
 * <br>
 * This is experimental regularization method.<br>
 *
 */
public class Lp_Regularization extends AbstractExecutionLayer {

    /**
     * Parameter name types for Lp_Regularization.
     *     - lambda: lambda value for regularization. Default value: 0.01.<br>
     *     - p: p norm of normalizer. Default 3.<br>
     *
     */
    private final static String paramNameTypes = "(lambda:DOUBLE), " +
            "(p:INT)";

    /**
     * Regularization rate.
     *
     */
    private double lambda;

    /**
     * Order of norm.
     *
     */
    private int p;

    /**
     * Regularized weight of next layer.
     *
     */
    private final HashSet<Matrix> layerRegularizedWeights = new HashSet<>();

    /**
     * Constructor for Lp regularization layer.
     *
     * @param layerIndex layer Index.
     * @param initialization initialization function for weight.
     * @param params parameters for feedforward layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Lp_Regularization(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException {
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
     * Returns parameters used for Lp regularization layer.
     *
     * @return parameters used for Lp regularization layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + Lp_Regularization.paramNameTypes;
    }

    /**
     * Sets parameters used for Lp regularization.<br>
     * <br>
     * Supported parameters are:<br>
     *     - lambda: lambda value for regularization. Default value: 0.01.<br>
     *     - p: p norm of normalizer. Default 3.<br>
     *
     * @param params parameters used for Lp regularization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("lambda")) lambda = params.getValueAsDouble("lambda");
        if (params.hasParam("p")) p = params.getValueAsInteger("p");
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
        if (!(getNextLayer() instanceof OutputLayer)) throw new NeuralNetworkException("Lp Regularization must be final layer prior output layer.");

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
                weightGradientSum.add(weight.apply((value) -> value != 0 ? p * lambda * Math.pow(Math.abs(value), p - 1) / value : 0), weightGradientSum);
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
            weightSum += weight.power(p).sum();
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
