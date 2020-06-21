/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.value;

import core.reinforcement.RLSample;
import core.reinforcement.State;
import core.reinforcement.function.FunctionEstimator;
import utils.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.Matrix;

/**
 * Class that defines PlainValueFunction.
 *
 */
public class PlainValueFunction extends AbstractValueFunction {

    /**
     * Constructor for PlainValueFunction.
     *
     */
    public PlainValueFunction() {
        super(1);
    }

    /**
     * Constructor for PlainValueFunction.
     *
     * @param numberOfActions number of actions for PlainValueFunction.
     */
    public PlainValueFunction(int numberOfActions) {
        super(numberOfActions);
    }

    /**
     * Constructor for PlainValueFunction.
     *
     * @param params parameters for PlainValueFunction.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public PlainValueFunction(String params) throws DynamicParamException {
        super(1, params);
    }

    /**
     * Constructor for PlainValueFunction.
     *
     * @param numberOfActions number of actions for PlainValueFunction.
     * @param params parameters for PlainValueFunction.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public PlainValueFunction(int numberOfActions, String params) throws DynamicParamException {
        super(numberOfActions, params);
    }

    /**
     * Not used.
     *
     */
    public void start() {}

    /**
     * Not used.
     *
     */
    public void stop() {}

    /**
     * Returns new values as empty matrix per number of actions defined.
     *
     * @param state state.
     * @return new values for state.
     */
    protected Matrix getValues(State state) {
        return new DMatrix(getNumberOfActions(), 1);
    }

    /**
     * Return target value for state based on it's next state.
     *
     * @param nextState next state.
     * @return target value of sample.
     */
    public double getTargetValue(State nextState) {
        return nextState.sample.stateValues.getValue(getAction(nextState), 0);
    }

    /**
     * Updates baseline value with reward value of sample.
     *
     * @param sample sample.
     */
    protected void updateBaseline(RLSample sample) {
    }

    /**
     * Returns FunctionEstimator.
     *
     * @return FunctionEstimator.
     */
    public FunctionEstimator getFunctionEstimator() {
        return null;
    }

    /**
     * Updates target value FunctionEstimator.
     *
     */
    public void updateTargetFunctionEstimator() {}

    /**
     * Returns current value error.
     *
     * @return current value error.
     */
    public double getValueError() {
        return 0;
    }

}
