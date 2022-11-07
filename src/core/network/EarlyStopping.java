/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
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
import java.util.ArrayDeque;
import java.util.Deque;

/**
 * Implements early stopping method for neural network.<br>
 * This class seeks ideas for underlying implementation from below referenced paper.<br>
 * <br>
 * Reference: https://www.researchgate.net/publication/2874749_Early_Stopping_-_But_When<br>
 *
 */
public class EarlyStopping implements Configurable, Serializable {

    @Serial
    private static final long serialVersionUID = -8362385201353383426L;

    /**
     * Parameter name types for early stopping.
     *     - trainingAverageSize: size for training error rolling average. Default value 100 (iterations).<br>
     *     - trainingStopThreshold: stop threshold for training error condition. Default 20 (consequent iterations where condition is met).<br>
     *     - validationAverageSize: size for validation error rolling average. Default value 100 (iterations).<br>
     *     - validationStopThreshold: stop threshold for validation error condition. Default 20 (consequent iterations where condition is met).<br>
     *
     */
    private final static String paramNameTypes = "(trainingAverageSize:INT), " +
            "(trainingStopThreshold:INT), " +
            "(validationAverageSize:INT), " +
            "(validationStopThreshold:INT)";

    /**
     * Params for early stopping.
     *
     */
    private final String params;

    /**
     * Size for training error rolling average.
     *
     */
    private int trainingAverageSize;

    /**
     * Training averages.
     *
     */
    private final Deque<Double> trainingAverages = new ArrayDeque<>();

    /**
     * Sets stop threshold for training error condition.
     *
     */
    private int trainingStopThreshold;

    /**
     * Stores previous training error average.
     *
     */
    private double previousTrainingAverage = Double.NEGATIVE_INFINITY;

    /**
     * Size for validation error rolling average.
     *
     */
    private int validationAverageSize;

    /**
     * Validation averages.
     *
     */
    private final Deque<Double> validationAverages = new ArrayDeque<>();

    /**
     * Sets stop threshold for validation error condition.
     *
     */
    private int validationStopThreshold;

    /**
     * Stores previous validation error average.
     *
     */
    private double previousValidationAverage = Double.NEGATIVE_INFINITY;

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
        trainingAverageSize = 100;
        trainingStopThreshold = 20;
        validationAverageSize = 100;
        validationStopThreshold = 20;
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
     *     - trainingAverageSize: size for training error rolling average. Default value 100 (iterations).<br>
     *     - trainingStopThreshold: stop threshold for training error condition. Default 20 (consequent iterations where condition is met).<br>
     *     - validationAverageSize: size for validation error rolling average. Default value 100 (iterations).<br>
     *     - validationStopThreshold: stop threshold for validation error condition. Default 20 (consequent iterations where condition is met).<br>
     *
     * @param params parameters used for stopping.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("trainingAverageSize")) trainingAverageSize = params.getValueAsInteger("trainingAverageSize");
        if (params.hasParam("trainingStopThreshold")) trainingStopThreshold = params.getValueAsInteger("trainingStopThreshold");
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
        if (!trainingStopCondition && iteration >= trainingAverageSize) {
            double lastAverage = getAverageError(trainingAverages, trainingAverageSize, trainingMetric.getLastError());
            if (previousTrainingAverage <= lastAverage && previousTrainingAverage != Double.NEGATIVE_INFINITY) trainingStopCount++;
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
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void evaluateValidationCondition(int iteration) throws MatrixException, DynamicParamException {
        if (!validationStopCondition && iteration >= validationAverageSize) {
            double lastAverage = getAverageError(validationAverages, validationAverageSize, validationMetric.getLastError());
            if (previousValidationAverage <= lastAverage && previousValidationAverage != Double.NEGATIVE_INFINITY) validationStopCount++;
            else {
                previousValidationAverage = lastAverage;
                validationStopCount = 0;
            }
            if (validationStopCount >= validationStopThreshold) validationStopCondition = true;
        }
    }

    /**
     * Returns cumulative error.
     *
     * @param errors errors.
     * @param maxSize max size of error queue.
     * @param lastError latest error.
     * @return average error.
     */
    private double getAverageError(Deque<Double> errors, int maxSize, double lastError) {
        double cumulativeError = 0;
        if (errors.size() == maxSize) errors.pollLast();
        errors.addFirst(lastError);
        for (Double error : errors) cumulativeError += error;
        return cumulativeError / (double)errors.size();
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
