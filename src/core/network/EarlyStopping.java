/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.network;

import core.metrics.Metric;
import core.metrics.SingleRegressionMetric;
import utils.configurable.Configurable;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;
import java.util.TreeMap;

/**
 * Implements early stopping method for neural network.<br>
 * This class seeks ideas for underlying implementation from below referenced paper.<br>
 * <br>
 * Reference: <a href="https://www.researchgate.net/publication/2874749_Early_Stopping_-_But_When">...</a><br>
 *
 */
public class EarlyStopping implements Configurable, Serializable {

    @Serial
    private static final long serialVersionUID = -8362385201353383426L;

    /**
     * Parameter name types for early stopping.
     *     - trainingPatience: training patience in terms of iterations. Default value 1 (iterations).<br>
     *     - trainingAverageSize: size for training error rolling average. Default value 500 (iterations).<br>
     *     - trainingStopThreshold: stop threshold for training error condition. Default 100 (consequent iterations where condition is met).<br>
     *     - validationPatience: validation patience in terms of iterations. Default value 1 (iterations).<br>
     *     - validationAverageSize: size for validation error rolling average. Default value 500 (iterations).<br>
     *     - validationStopThreshold: stop threshold for validation error condition. Default 100 (consequent iterations where condition is met).<br>
     *
     */
    private final static String paramNameTypes = "(trainingPatience:INT), " +
            "(trainingAverageSize:INT), " +
            "(trainingStopThreshold:INT), " +
            "(validationPatience:INT), " +
            "(validationAverageSize:INT), " +
            "(validationStopThreshold:INT)";

    /**
     * Params for early stopping.
     *
     */
    private final String params;

    /**
     * Training patience in terms of iterations.
     *
     */
    private int trainingPatience;

    /**
     * Size for training error rolling average.
     *
     */
    private int trainingAverageSize;

    /**
     * Training averages.
     *
     */
    private final TreeMap<Integer, Double> trainingAverages = new TreeMap<>();

    /**
     * Sets stop threshold for training error condition.
     *
     */
    private int trainingStopThreshold;

    /**
     * Stores previous training error average.
     *
     */
    private double previousTrainingAverage = Double.MAX_VALUE;

    /**
     * Validation patience in terms of iterations.
     *
     */
    private int validationPatience;

    /**
     * Size for validation error rolling average.
     *
     */
    private int validationAverageSize;

    /**
     * Validation averages.
     *
     */
    private final TreeMap<Integer, Double> validationAverages = new TreeMap<>();

    /**
     * Sets stop threshold for validation error condition.
     *
     */
    private int validationStopThreshold;

    /**
     * Stores previous validation error average.
     *
     */
    private double previousValidationAverage = Double.MAX_VALUE;

    /**
     * Reference to training error instance.
     *
     */
    private transient SingleRegressionMetric trainingMetric;

    /**
     * Reference to validation error instance.
     *
     */
    private transient Metric validationMetric;

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
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public EarlyStopping() throws DynamicParamException {
        this(null);
    }

    /**
     * Constructor for early stopping class.
     *
     * @param params parameters for early stopping.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public EarlyStopping(String params) throws DynamicParamException {
        initializeDefaultParams();
        this.params = params;
        if (params != null) setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        trainingPatience = 1;
        trainingAverageSize = 500;
        trainingStopThreshold = 100;
        validationPatience = 1;
        validationAverageSize = 500;
        validationStopThreshold = 100;
    }

    /**
     * Returns parameters used for early stopping.
     *
     * @return parameters used for early stopping.
     */
    public String getParamDefs() {
        return EarlyStopping.paramNameTypes;
    }

    /**
     * Sets parameters used for early stopping.<br>
     * <br>
     * Supported parameters are:<br>
     *     - trainingPatience: training patience in terms of iterations. Default value 1 (iterations).<br>
     *     - trainingAverageSize: size for training error rolling average. Default value 500 (iterations).<br>
     *     - trainingStopThreshold: stop threshold for training error condition. Default 100 (consequent iterations where condition is met).<br>
     *     - validationPatience: validation patience in terms of iterations. Default value 1 (iterations).<br>
     *     - validationAverageSize: size for validation error rolling average. Default value 500 (iterations).<br>
     *     - validationStopThreshold: stop threshold for validation error condition. Default 100 (consequent iterations where condition is met).<br>
     *
     * @param params parameters used for stopping.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("trainingPatience")) trainingPatience = params.getValueAsInteger("trainingPatience");
        if (params.hasParam("trainingAverageSize")) trainingAverageSize = params.getValueAsInteger("trainingAverageSize");
        if (params.hasParam("trainingStopThreshold")) trainingStopThreshold = params.getValueAsInteger("trainingStopThreshold");
        if (params.hasParam("validationPatience")) validationPatience = params.getValueAsInteger("validationPatience");
        if (params.hasParam("validationAverageSize")) validationAverageSize = params.getValueAsInteger("validationAverageSize");
        if (params.hasParam("validationStopThreshold")) validationStopThreshold = params.getValueAsInteger("validationStopThreshold");
    }

    /**
     * Returns reference to early stopping.
     *
     * @return reference to early stopping.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public EarlyStopping reference() throws DynamicParamException {
        return new EarlyStopping(params);
    }

    /**
     * Sets training error reference.
     *
     * @param trainingMetric training error reference.
     */
    public void setTrainingMetric(SingleRegressionMetric trainingMetric) {
        this.trainingMetric = trainingMetric;
    }

    /**
     * Sets validation error reference.
     *
     * @param validationMetric validation error reference.
     */
    public void setValidationMetric(Metric validationMetric) {
        this.validationMetric = validationMetric;
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
        if (!trainingStopCondition && iteration > trainingPatience) {
            double lastTrainingAverage = getAverageError(trainingAverages, iteration -trainingPatience, trainingAverageSize, trainingMetric.getLastError());
            if (iteration >= trainingAverageSize) {
                if (previousTrainingAverage == Double.MAX_VALUE) previousTrainingAverage = lastTrainingAverage;
                else {
                    if (previousTrainingAverage >= lastTrainingAverage) trainingStopCount++;
                    else {
                        previousTrainingAverage = lastTrainingAverage;
                        trainingStopCount = 0;
                    }
                }
                if (trainingStopCount >= trainingStopThreshold) trainingStopCondition = true;
            }
        }
    }

    /**
     * Function that evaluates validation condition.<br>
     * It first checks that iteration count is in minimum bigger than validation average size and validation condition threshold is not reached yet.<br>
     * It then compares if current rolling validation average is less than previous rolling validation average.<br>
     * If yes validation condition count is increased otherwise it is reset to zero.<br>
     *
     * @param iteration current neural network training iteration.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void evaluateValidationCondition(int iteration) throws MatrixException, DynamicParamException {
        if (!validationStopCondition && iteration > validationPatience) {
            double lastValidationAverage = getAverageError(validationAverages, iteration - validationPatience, validationAverageSize, validationMetric.getLastError());
            if (iteration >= validationAverageSize) {
                if (previousValidationAverage == Double.MAX_VALUE) previousValidationAverage = lastValidationAverage;
                else {
                    if (previousValidationAverage >= lastValidationAverage) validationStopCount++;
                    else {
                        previousValidationAverage = lastValidationAverage;
                        validationStopCount = 0;
                    }
                }
                if (validationStopCount >= validationStopThreshold) validationStopCondition = true;
            }
        }
    }

    /**
     * Returns cumulative error.
     *
     * @param errors errors.
     * @param iteration iteration.
     * @param maxSize max size of error queue.
     * @param lastError latest error.
     * @return average error.
     */
    private double getAverageError(TreeMap<Integer, Double> errors, int iteration, int maxSize, double lastError) {
        if (!errors.isEmpty()) {
            int lastKey = errors.lastKey();
            double lastValue = errors.get(lastKey);
            for (int index = lastKey + 1; index < iteration - 1; index++) errors.put(index, lastValue);
        }

        errors.put(iteration, lastError);

        while (errors.size() > maxSize) errors.remove(errors.firstKey());

        double movingAverageError = Double.MIN_VALUE;
        double smoothingFactor = 0.99;
        for (Double error : errors.values()) movingAverageError = movingAverageError == Double.MIN_VALUE ? error : smoothingFactor * movingAverageError + (1 - smoothingFactor) * error;
        return movingAverageError;
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
