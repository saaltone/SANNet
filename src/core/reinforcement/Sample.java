/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements sample that contains information of current state, action taken, reward received, next state and error of target value.
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
     * Immediate reward after taking specific action in state.
     *
     */
    public double reward;

    /**
     * Next state after action taken.
     *
     */
    public Matrix nextState;

    /**
     * Priority value for sample.
     *
     */
    public double priority;

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
     */
    public Sample(Matrix state) {
        this.state = state;
    }

    /**
     * Constructor for sample.
     *
     * @param state environment state.
     * @param action action taken to move from state to target state.
     * @param reward immediate reward after taking specific action in state.
     * @param nextState next state after action taken.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Sample(Matrix state, int action, double reward, Matrix nextState) throws MatrixException {
        this(state, action, reward, nextState, false);
    }

    /**
     * Constructor for sample
     *
     * @param state environment state.
     * @param action action taken to move from state to target state.
     * @param reward immediate reward after taking specific action in state.
     * @param nextState next state after action taken.
     * @param asCopy if true values are added as deep copy.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Sample(Matrix state, int action, double reward, Matrix nextState, boolean asCopy) throws MatrixException {
        this.state = !asCopy || state == null ? state : state.copy();
        this.action = action;
        this.reward = reward;
        this.nextState = !asCopy || nextState == null ? nextState : nextState.copy();
    }

    /**
     * Returns copy of sample.
     *
     * @return copy of sample.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Sample copy() throws MatrixException {
        return new Sample(state, action, reward, nextState, true);
    }

    /**
     * Compares samples by state, action, reward and next state.
     *
     * @param otherSample other sample to be compared.
     * @return true if samples are equal otherwise false.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public boolean equals(Sample otherSample) throws MatrixException {
        return compare(state, otherSample.state) && action == otherSample.action && reward == otherSample.reward && compare(nextState, otherSample.nextState);
    }

    /**
     * Compares two matrices with each other.
     *
     * @param matrix1 first matrix to be compared.
     * @param matrix2 second matrix to be compared.
     * @return returns true if matrices are equal otherwise returns false.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private boolean compare(Matrix matrix1, Matrix matrix2) throws MatrixException {
        if ((matrix1 == null && matrix2 == null)) return true;
        else return (matrix1 != null && matrix2 != null) && matrix1.equals(matrix2);
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
        reward = otherSample.reward;
        nextState.setEqualTo(otherSample.nextState);
        priority = otherSample.priority;
    }

}
