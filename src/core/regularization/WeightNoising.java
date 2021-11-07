/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.regularization;

import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.sampling.Sequence;

import java.util.Random;

/**
 * Class that adds noise to the weights during training (backward) phase.<br>
 *
 */
public class WeightNoising extends AbstractRegularization {

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
     * Constructor for weight noising class.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public WeightNoising() throws DynamicParamException {
        super(RegularizationType.WEIGHT_NOISING, WeightNoising.paramNameTypes);
    }

    /**
     * Constructor for weight noising class.
     *
     * @param params parameters for gradient clipping.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public WeightNoising(String params) throws DynamicParamException {
        super(RegularizationType.WEIGHT_NOISING, WeightNoising.paramNameTypes, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        initialNoise = 0.02;
        minNoise = 0;
        noiseDecay = 0.999;
        currentNoise = initialNoise;
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
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("initialNoise")) initialNoise = params.getValueAsDouble("initialNoise");
        if (params.hasParam("minNoise")) minNoise = params.getValueAsDouble("minNoise");
        if (params.hasParam("noiseDecay")) noiseDecay = params.getValueAsDouble("noiseDecay");
        currentNoise = initialNoise;
    }

    /**
     * Not used.
     *
     * @param sequence input sequence.
     */
    public void forward(Sequence sequence) {}

    /**
     * Not used.
     *
     * @param inputs inputs.
     */
    public void forward(MMatrix inputs) {}

    /**
     * Not used.
     *
     * @param weight weight matrix.
     * @return not used.
     */
    public double error(Matrix weight) {
        return 0;
    }

    /**
     * Executes weight noising prior weight update step for neural network.<br>
     *
     * @param weight weight matrix.
     * @param weightGradientSum gradient sum of weight.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backward(Matrix weight, Matrix weightGradientSum) throws MatrixException {
        weight.apply(value -> value + currentNoise * (1 - 2 * random.nextDouble()), true);
        if (currentNoise > minNoise) currentNoise *= noiseDecay;
    }

}
