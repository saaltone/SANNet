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
import utils.matrix.*;

import java.util.HashMap;
import java.util.HashSet;

/**
 * Implements L2 (ridge) regularization.<br>
 * <br>
 * Reference: https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c<br>
 *
 */
public class L2_Regularization extends AbstractRegularizationLayer {

    /**
     * Parameter name types for L2 regularization.
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
     * @param layerIndex layer index
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
     * Takes single backward processing step to process layer output gradient(s) towards input.<br>
     * Applies automated backward (automatic gradient) procedure when relevant to layer.<br>
     * Applies additionally any regularization defined for layer.<br>
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
            Matrix weightGradientSum = layerWeightGradients.get(weight);
            if (weightGradientSum != null) {
                weightGradientSum.add(weight.multiply(2 * lambda), weightGradientSum);
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

}
