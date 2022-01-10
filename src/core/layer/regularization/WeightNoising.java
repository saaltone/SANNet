/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.layer.regularization;

import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Initialization;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashSet;
import java.util.Random;

/**
 * Layer that adds noise to the weights during training phase.<br>
 *
 */
public class WeightNoising extends AbstractRegularizationLayer {

    /**
     * Parameter name types for WeightNoising.
     *     - initialNoise: initial noise level. Default value 0.02.<br>
     *     - minNoise: minimum noise level. Default value 0.<br>
     *     - noiseDecay: noise decay factor. Default value 0.999.<br>
     *
     */
    private final static String paramNameTypes = "(initialNoise:DOUBLE), " +
            "(minNoise:DOUBLE), " +
            "(noiseDecay:DOUBLE)";

    /**
     * Random function for weight noising.
     *
     */
    private final Random random = new Random();

    /**
     * Current noise.
     *
     */
    private double currentNoise;

    /**
     * Initial noise.
     *
     */
    private double initialNoise;

    /**
     * Initial noise.
     *
     */
    private double minNoise;

    /**
     * Initial noise.
     *
     */
    private double noiseDecay;

    /**
     * Constructor for WeightNoising layer.
     *
     * @param layerIndex layer Index.
     * @param initialization initialization function for weight.
     * @param params parameters for feedforward layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public WeightNoising(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException {
        super (layerIndex, initialization, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        initialNoise = 0.02;
        minNoise = 0;
        noiseDecay = 0.999;
        currentNoise = initialNoise;
    }

    /**
     * Returns parameters used for WeightNoising layer.
     *
     * @return parameters used for WeightNoising layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + WeightNoising.paramNameTypes;
    }

    /**
     * Sets parameters used for WeightNoising.<br>
     * <br>
     * Supported parameters are:<br>
     *     - initialNoise: initial noise level. Default value 0.02.<br>
     *     - minNoise: minimum noise level. Default value 0.<br>
     *     - noiseDecay: noise decay factor. Default value 0.999.<br>
     *
     * @param params parameters used for weight noising.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("initialNoise")) initialNoise = params.getValueAsDouble("initialNoise");
        if (params.hasParam("minNoise")) minNoise = params.getValueAsDouble("minNoise");
        if (params.hasParam("noiseDecay")) noiseDecay = params.getValueAsDouble("noiseDecay");
        currentNoise = initialNoise;
    }

    /**
     * Defines layer procedure for forward and backward calculation (automatic gradient) by applying procedure factory.<br>
     *
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     */
    protected void defineProcedure() throws NeuralNetworkException {
        if (getNextLayer().getWeightsMap().isEmpty()) throw new NeuralNetworkException("Unable initialize weight noising. Next layer does not contain any weights.");
    }

    /**
     * Takes single forward processing step to process layer input(s).<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forwardProcess() throws MatrixException {
        resetLayerOutputs();
        getLayerOutputs().putAll(getPreviousLayerOutputs());

        if (isTraining()) {
            HashSet<Matrix> nextLayerNormalizedWeights = getNextLayer().getNormalizedWeights();
            for (Matrix weight : nextLayerNormalizedWeights) {
                weight.apply(value -> value + currentNoise * (1 - 2 * random.nextDouble()), true);
                if (currentNoise > minNoise) currentNoise *= noiseDecay;
            }
        }
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
    }

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        return "Initial noise: " + initialNoise + ", Noise decay: " + noiseDecay + ", Min noise: " + minNoise;
    }

}
