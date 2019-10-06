/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement;

import utils.Matrix;
import utils.MatrixException;

/**
 * Implements sample that contains information of current state, action taken, reward received, next state and value / error of target state.
 *
 */
public class Sample {

    /**
     * Environment state.
     *
     */
    public Matrix state;

    /**
     * Action taken to move from state to target state.
     *
     */
    public int action;

    /**
     * True if taken action is valid per assessment of environment.
     *
     */
    public boolean validAction;

    /**
     * Immediate reward after taking specific action in state.
     *
     */
    public double reward;

    /**
     * Target state after action taken.
     *
     */
    public Matrix targetState;

    /**
     * Predicted values of current environment state.
     *
     */
    public Matrix values;

    /**
     * If state is terminal
     *
     */
    public boolean terminalState;

    /**
     * Delta value for sample (target value minus current value for specific action).
     *
     */
    public double delta;

    /**
     * Priority value for sample.
     *
     */
    public double priority = 0;

    /**
     * Default constructor for sample.
     *
     */
    public Sample() {
    }

    /**
     * Constructor for sample.
     *
     * @param state environment state.
     * @param action action taken to move from state to target state.
     * @param validAction true if taken action is valid per assessment of environment.
     * @param reward immediate reward after taking specific action in state.
     * @param targetState target state after action taken.
     * @param values predicted values of state.
     * @param terminalState true if sample state is terminal (final).
     * @param delta delta value for sample.
     */
    public Sample(Matrix state, int action, boolean validAction, double reward, Matrix targetState, Matrix values, boolean terminalState, double delta) {
        this(state, action, validAction, reward, targetState, values, terminalState, delta, false);
    }

    /**
     * Constructor for sample
     *
     * @param state environment state.
     * @param action action taken to move from state to target state.
     * @param validAction true if taken action is valid per assessment of environment.
     * @param reward immediate reward after taking specific action in state.
     * @param targetState target state after action taken.
     * @param values predicted values of state.
     * @param terminalState true if sample state is terminal (final).
     * @param delta delta value for sample.
     * @param asCopy if true values are added as deep copy.
     */
    public Sample(Matrix state, int action, boolean validAction, double reward, Matrix targetState, Matrix values, boolean terminalState, double delta, boolean asCopy) {
        this.state = !asCopy || state == null ? state : state.copy();
        this.action = action;
        this.validAction = validAction;
        this.reward = reward;
        this.targetState = !asCopy || targetState == null ? targetState : targetState.copy();
        this.values = !asCopy || values == null ? values : values.copy();
        this.terminalState = terminalState;
        this.delta = delta;
    }

    /**
     * Returns copy of sample.
     *
     * @return copy of sample.
     */
    public Sample copy() {
        return new Sample(state, action, validAction, reward, targetState, values, terminalState, delta,true);
    }

    /**
     * Compares samples by state.
     *
     * @param otherSample other sample to be compared.
     * @return true if samples are equal otherwise false.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public boolean equals(Sample otherSample) throws MatrixException {
        return state.equals(otherSample.state);
    }

    /**
     * Makes current sample equal to other sample.
     *
     * @param otherSample other sample
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void setEqualTo(Sample otherSample) throws MatrixException {
        state.setEqualTo(otherSample.state);
        action = otherSample.action;
        validAction = otherSample.validAction;
        reward = otherSample.reward;
        targetState.setEqualTo(otherSample.targetState);
        values.setEqualTo(otherSample.values);
        terminalState = otherSample.terminalState;
        delta = otherSample.delta;
        priority = otherSample.priority;
    }

}
