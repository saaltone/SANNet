/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.regularization;

import core.metrics.Metrics;
import utils.DynamicParam;
import utils.DynamicParamException;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Implements early stopping method for neural network.<br>
 * This class seeks ideas for underlying implementation from below referenced paper.<br>
 * <br>
 * Reference: https://www.researchgate.net/publication/2874749_Early_Stopping_-_But_When<br>
 *
 */
public class EarlyStopping implements Serializable {

    private static final long serialVersionUID = -8362385201353383426L;

    /**
     * Size for training error rolling average.
     *
     */
    private int trainingAverageSize = 100;

    /**
     * Sets stop threshold for training error condition.
     *
     */
    private int trainingStopThreshold = 20;

    /**
     * Stores previous training error average.
     *
     */
    private double previousTrainingAverage = Double.MIN_VALUE;

    /**
     * Size for validation error rolling average.
     *
     */
    private int validationAverageSize = 100;

    /**
     * Sets stop threshold for validation error condition.
     *
     */
    private int validationStopThreshold = 20;

    /**
     * Stores previous validation error average.
     *
     */
    private double previousValidationAverage = Double.MIN_VALUE;

    /**
     * Reference to training error instance.
     *
     */
    private transient Metrics trainingError;

    /**
     * Reference to validation error instance.
     *
     */
    private transient Metrics validationError;

    /**
     * Stores current neural network training iteration.
     *
     */
    private transient int iteration = 0;

    /**
     * Flag if training stop condition has been achieved.
     *
     */
    private transient boolean trainingStopCondition = false;

    /**
     * Count for training stop condition.
     *
     */
    private transient int trainingStopCount = 0;

    /**
     * Flag if validation stop condition has been achieved.
     *
     */
    private transient boolean validationStopCondition = false;

    /**
     * Count for validation stop condition.
     *
     */
    private transient int validationStopCount = 0;

    /**
     * Default constructor for early stopping class.
     *
     */
    public EarlyStopping() {
    }

    /**
     * Constructor for early dropping class.
     *
     * @param params parameters for early stopping.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public EarlyStopping(String params) throws DynamicParamException {
        this.setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Gets parameters used for early stopping.
     *
     * @return parameters used for early stopping.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("trainingAverageSize", DynamicParam.ParamType.INT);
        paramDefs.put("trainingStopThreshold", DynamicParam.ParamType.INT);
        paramDefs.put("validationAverageSize", DynamicParam.ParamType.INT);
        paramDefs.put("validationStopThreshold", DynamicParam.ParamType.INT);
        return paramDefs;
    }

    /**
     * Sets parameters used for early stopping.<br>
     * <br>
     * Supported parameters are:<br>
     *     - trainingAverageSize: size for training error rolling average. Default value 100 (iterations).<br>
     *     - trainingStopThreshold: stop threshold for training error condition. Default 20 (consequent iterations where condition is met).<br>
     *     - validationAverageSize: size for validation error rolling average. Default value 100 (iterations).<br>
     *     - validationStopThreshold: stop threshold for validation error condition. Default 20 (consequent iterations where condition is met).<br>
     *
     * @param params parameters used for stopping.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("trainingAverageSize")) trainingAverageSize = params.getValueAsInteger("trainingAverageSize");
        if (params.hasParam("trainingStopThreshold")) trainingStopThreshold = params.getValueAsInteger("trainingStopThreshold");
        if (params.hasParam("validationAverageSize")) validationAverageSize = params.getValueAsInteger("validationAverageSize");
        if (params.hasParam("validationStopThreshold")) validationStopThreshold = params.getValueAsInteger("validationStopThreshold");
    }

    /**
     * Sets training error reference.
     *
     * @param trainingError training error reference.
     */
    public void setTrainingError(Metrics trainingError) {
        this.trainingError = trainingError;
    }

    /**
     * Sets validation error reference.
     *
     * @param validationError validation error reference.
     */
    public void setValidationError(Metrics validationError) {
        this.validationError = validationError;
    }

    /**
     * Function that evaluates training condition.<br>
     * It first checks that iteration count is in minimum bigger than training average size and training condition threshold is not reached yet.<br>
     * It then compares if current rolling training average is less than previous rolling training average.<br>
     * If yes training condition count is increased otherwise it is reset to zero.<br>
     *
     * @param iteration current neural network training iteration.
     */
    public void evaluateTrainingCondition(int iteration) {
        this.iteration = iteration;

        if (!trainingStopCondition && iteration >= trainingAverageSize) {
            double lastAverage = trainingError.getAverage(trainingAverageSize);
            if (previousTrainingAverage <= lastAverage && previousTrainingAverage != Double.MIN_VALUE) trainingStopCount++;
            else {
                previousTrainingAverage = lastAverage;
                trainingStopCount = 0;
            }
            if (trainingStopCount >= trainingStopThreshold) trainingStopCondition = true;
        }

    }

    /**
     * Function that evaluates validation condition.<br>
     * It first checks that iteration count is in minimum bigger than validation average size and validation condition threshold is not reached yet.<br>
     * It then compares if current rolling validation average is less than previous rolling validation average.<br>
     * If yes validation condition count is increased otherwise it is reset to zero.<br>
     *
     * @param iteration current neural network training iteration.
     */
    public void evaluateValidationCondition(int iteration) {
        this.iteration = iteration;

        if (!validationStopCondition && iteration >= validationAverageSize) {
            double lastAverage = validationError.getAverage(validationAverageSize);
            if (previousValidationAverage <= lastAverage && previousValidationAverage != Double.MIN_VALUE) validationStopCount++;
            else {
                previousValidationAverage = lastAverage;
                validationStopCount = 0;
            }
            if (validationStopCount >= validationStopThreshold) validationStopCondition = true;
        }

    }

    /**
     * Returns true if early stopping condition is reached otherwise false.<br>
     * Early stopping condition is reached if both training and validation stopping conditions have been reached.<br>
     *
     * @return true if early stopping condition is reached otherwise false.
     */
    public boolean stopTraining() {
        return trainingStopCondition && validationStopCondition;
    }

}
