/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.regularization;

import utils.Configurable;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.Sequence;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;
import java.util.Random;

/**
 * Class that adds noise to the weights during training (backward) phase.<br>
 *
 */
public class WeightNoising implements Configurable, Regularization, Serializable {

    @Serial
    private static final long serialVersionUID = -6830727265041914868L;

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
     * Type of regularization.
     *
     */
    private final RegularizationType regularizationType = RegularizationType.WEIGHT_NOISING;

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
     */
    public WeightNoising() {
        initializeDefaultParams();
    }

    /**
     * Constructor for weight noising class.
     *
     * @param params parameters for gradient clipping.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public WeightNoising(String params) throws DynamicParamException {
        this();
        setParams(new DynamicParam(params, getParamDefs()));
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
     * Returns parameters used for weight noising.
     *
     * @return parameters used for weight noising.
     */
    public String getParamDefs() {
        return WeightNoising.paramNameTypes;
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
     * @param isTraining if true neural network is in state otherwise false.
     */
    public void setTraining(boolean isTraining) {
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

    /**
     * Returns name of regularization.
     *
     * @return name of regularization.
     */
    public String getName() {
        return regularizationType.toString();
    }

}
