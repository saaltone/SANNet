/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement;

import utils.matrix.Matrix;

import java.io.Serializable;

/**
 * Defines RLSample (reinforcement learning sample) that contains information per current state and value of state.
 *
 */
public class RLSample implements Serializable {

    private static final long serialVersionUID = -3307739679812329676L;

    /**
     * State reference for sample.
     *
     */
    public final State state;

    /**
     * Current policy value estimated by policy function.
     *
     */
    public double policyValue;

    /**
     * State values estimated by value function.
     *
     */
    public Matrix stateValues;

    /**
     * Target TD value.
     *
     */
    public double tdTarget;

    /**
     * TD (temporal difference) error.
     *
     */
    public double tdError;

    /**
     * Baseline for sample.
     *
     */
    public double baseline;

    /**
     * Current time step.
     *
     */
    public int timeStep;

    /**
     * Priority value for sample.
     *
     */
    public double priority;

    /**
     * Entropy value for sample.
     *
     */
    public double entropy;

    /**
     * Importance sampling weight.
     *
     */
    public double importanceSamplingWeight;

    /**
     * Constructor for sample.
     *
     * @param state state.
     */
    public RLSample(State state) {
        this.state = state;
        state.sample = this;
    }

    /**
     * Returns current values of sample.
     *
     * @return current values of sample.
     */
    public Matrix getValues(){
        return stateValues;
    }

    /**
     * Sets new values for sample.
     *
     * @param newValues new sample values.
     */
    public void setValues(Matrix newValues) {
        stateValues = newValues;
    }

    /**
     * Get value of sample matching current action.
     *
     * @param action current action.
     * @return value of sample matching current action.
     */
    public double getValue(int action) {
        return getValues().getValue(action, 0);
    }

    /**
     * Sets value of sample matching current action.
     *
     * @param action current action.
     * @param newValue new value of sample matching current action.
     */
    public void setValue(int action, double newValue) {
        getValues().setValue(action, 0, newValue);
    }

}
