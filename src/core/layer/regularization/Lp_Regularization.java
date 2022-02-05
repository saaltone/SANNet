/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.layer.regularization;

import core.layer.NeuralNetworkLayer;
import core.layer.OutputLayer;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Initialization;
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
public class Lp_Regularization extends AbstractRegularizationLayer {

    /**
     * Parameter name types for Lp regularization.
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
     * @param layerIndex layer index
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
     * Takes single backward processing step to process layer output gradient(s) towards input.<br>
     * Applies automated backward (automatic gradient) procedure when relevant to layer.<br>
     * Additionally applies any regularization defined for layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void backwardProcess() throws MatrixException, DynamicParamException {
        super.backwardProcess();

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

}
