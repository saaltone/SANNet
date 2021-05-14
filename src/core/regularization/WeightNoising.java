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

import java.io.Serial;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Random;

/**
 * Class that adds noise to the weights during training (backward) phase.<br>
 *
 */
public class WeightNoising implements Regularization, Serializable {

    @Serial
    private static final long serialVersionUID = -6830727265041914868L;

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
     * Constructor for weight noising class.
     *
     */
    public WeightNoising() {
        currentNoise = initialNoise;
    }

    /**
     * Constructor for weight noising class.
     *
     * @param params parameters for gradient clipping.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public WeightNoising(String params) throws DynamicParamException {
        this();
        this.setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for weight noising.
     *
     * @return parameters used for weight noising.
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
     */
    public void backward(Matrix weight, Matrix weightGradientSum) {
        int rows = weight.getRows();
        int columns = weight.getColumns();
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
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
