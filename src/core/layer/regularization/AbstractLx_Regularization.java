/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
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
import utils.matrix.UnaryFunctionType;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Stack;

/**
 * Implements abstract regularization layer for Lx regulation.<br>
 *
 */
public abstract class AbstractLx_Regularization extends AbstractRegularizationLayer {

    /**
     * Parameter name types for AbstractLx_Regularization.
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
     * Set of previous neural network layers.
     *
     */
    private final HashSet<NeuralNetworkLayer> previousNeuralNetworkLayers = new HashSet<>();

    /**
     * Constructor for AbstractLx_Regularization layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for feedforward layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractLx_Regularization(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException {
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
     * Returns parameters used for AbstractLx_Regularization layer.
     *
     * @return parameters used for AbstractLx_Regularization layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + AbstractLx_Regularization.paramNameTypes;
    }

    /**
     * Sets parameters used for AbstractLx_Regularization.<br>
     * <br>
     * Supported parameters are:<br>
     *     - lambda: lambda value for regularization. Default value: 0.01.<br>
     *
     * @param params parameters used for AbstractLx_Regularization.
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
        for (NeuralNetworkLayer nextLayer : getNextLayers().values()) if (!(nextLayer instanceof OutputLayer)) throw new NeuralNetworkException("L1 Regularization must be final layer prior output layer.");

        Stack<NeuralNetworkLayer> neuralNetworkLayers = new Stack<>();
        for (NeuralNetworkLayer previousNeuralNetworkLayer : getPreviousLayers().values()) neuralNetworkLayers.push(previousNeuralNetworkLayer);
        while (!neuralNetworkLayers.isEmpty()) {
            NeuralNetworkLayer neuralNetworkLayer = neuralNetworkLayers.pop();
            for (NeuralNetworkLayer previousNeuralNetworkLayer : neuralNetworkLayer.getPreviousLayers().values()) neuralNetworkLayers.push(previousNeuralNetworkLayer);

            HashSet<Matrix> regularizedWeights = neuralNetworkLayer.getRegularizedWeights();
            if (regularizedWeights != null) {
                previousNeuralNetworkLayers.add(neuralNetworkLayer);
                layerRegularizedWeights.addAll(regularizedWeights);
            }
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
        for (NeuralNetworkLayer previousNeuralNetworkLayer : previousNeuralNetworkLayers) {
            layerWeightGradients.putAll(previousNeuralNetworkLayer.getLayerWeightGradients());
        }

        for (Matrix weight : layerRegularizedWeights) {
            Matrix weightGradientSum = layerWeightGradients.get(weight);
            if (weightGradientSum != null) {
                weightGradientSum.addBy(applyRegularization(weight, lambda));
            }
        }

    }

    /**
     * Applies regularization.
     *
     * @param weight weight
     * @param lambda lambda value
     * @return regularization result
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract Matrix applyRegularization(Matrix weight, double lambda) throws MatrixException;

    /**
     * Cumulates error from (L1 / L2 / Lp) regularization.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return cumulated error from regularization.
     */
    public double error() throws MatrixException, DynamicParamException {
        double weightSum = 0;
        for (NeuralNetworkLayer previousNeuralNetworkLayer : previousNeuralNetworkLayers) {
            for (Matrix weight : previousNeuralNetworkLayer.getRegularizedWeights()) {
                weightSum += weight.apply(UnaryFunctionType.ABS).sum();
            }
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
