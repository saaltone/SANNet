/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.regularization;

import utils.DynamicParam;
import utils.DynamicParamException;
import utils.Sequence;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Random;

/**
 * Class that adds noise to the weights during training (backward) phase.
 *
 */
public class WeightNoising implements Regularization, Serializable {

    private static final long serialVersionUID = -6830727265041914868L;

    /**
     * Random function for NoisyNextBestPolicy.
     *
     */
    private final Random random = new Random();

    /**
     * Type of regularization.
     *
     */
    private final RegularizationType regularizationType;

    /**
     * Current noise.
     *
     */
    private double currentNoise;

    /**
     * Initial noise.
     *
     */
    private double initialNoise = 0.02;

    /**
     * Initial noise.
     *
     */
    private double minNoise = 0;

    /**
     * Initial noise.
     *
     */
    private double noiseDecay = 0.999;

    /**
     * Constructor for WeightNoising class.
     *
     * @param regularizationType regularizationType.
     */
    public WeightNoising(RegularizationType regularizationType) {
        this.regularizationType = regularizationType;
        currentNoise = initialNoise;
    }

    /**
     * Constructor for WeightNoising class.
     *
     * @param regularizationType regularizationType.
     * @param params parameters for gradient clipping.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public WeightNoising(RegularizationType regularizationType, String params) throws DynamicParamException {
        this(regularizationType);
        this.setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for WeightNoising.
     *
     * @return parameters used for WeightNoising.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("initialNoise", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("minNoise", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("noiseDecay", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for WeightNoising.<br>
     * <br>
     * Supported parameters are:<br>
     *     - initialNoise: initial noise level. Default value 0.1.<br>
     *     - minNoise: minimum noise level. Default value 0.<br>
     *     - noiseDecay: noise decay factor. Default value 0.999.<br>
     *
     * @param params parameters used for WeightNoising.
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
     * Executes WeightNoising prior weight update step for neural network.<br>
     *
     */
    public void backward(Matrix weight, Matrix weightGradientSum) {
        for (int row = 0; row < weight.getRows(); row++) {
            for (int column = 0; column < weight.getColumns(); column++) {
                weight.setValue(row, column, weight.getValue(row, column) + currentNoise * (1 - 2 * random.nextDouble()));
            }
        }
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
