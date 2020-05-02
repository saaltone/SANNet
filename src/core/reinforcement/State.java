/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serializable;
import java.util.HashSet;

/**
 * Class that implements state containing information of state matrix, action and available actions, reward and reference to previous and next states.
 *
 */
public class State implements Serializable {

    private static final long serialVersionUID = 3018272924414901045L;

    /**
     * Sample reference for state.
     *
     */
    public RLSample sample;

    /**
     * Current state matrix.
     *
     */
    public final Matrix stateMatrix;

    /**
     * Available actions in current state.
     *
     */
    public HashSet<Integer> availableActions;

    /**
     * Action taken to move from state to next state.
     *
     */
    public int action;

    /**
     * Immediate reward after taking specific action in current state.
     *
     */
    public double reward;

    /**
     * If true state is final episode state.
     *
     */
    public boolean finalState = false;

    /**
     * Previous state.
     *
     */
    public State previousState;

    /**
     * Next state.
     *
     */
    public State nextState;

    /**
     * Constructor for State.
     * @param stateMatrix current state matrix
     */
    public State(Matrix stateMatrix) {
        this.stateMatrix = stateMatrix;
    }

    /**
     * Checks if state is final
     *
     * @return returns true if state is final (for episode) otherwise returns false.
     */
    public boolean isFinalState() {
        return finalState;
    }

    /**
     * Returns next state from current state.
     *
     * @param nextState next state matrix.
     * @return next state.
     */
    public State getNextState(Matrix nextState) {
        State newState = new State(nextState);
        this.nextState = newState;
        newState.previousState = this;
        return newState;
    }

    /**
     * Compares state by state, action, reward and next state.
     *
     * @param otherState state to be compared.
     * @return true if states are equal otherwise false.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public boolean equals(State otherState) throws MatrixException {
        if (!compare(stateMatrix, otherState.stateMatrix)) return false;
        if (action != otherState.action) return false;
        if (reward != otherState.reward) return false;
        if (nextState == null && otherState.nextState == null) return true;
        else {
            if (nextState == null || otherState.nextState == null) return false;
            else return compare(nextState.stateMatrix, otherState.nextState.stateMatrix);
        }
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
        if (matrix1 == null && matrix2 == null) return true;
        else {
            if (matrix1 == null || matrix2 == null) return false;
            else return matrix1.equals(matrix2);
        }
    }

}
