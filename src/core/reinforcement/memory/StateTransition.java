/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.reinforcement.memory;

import core.reinforcement.agent.EnvironmentState;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;

/**
 * Class that implements StateTransition containing information of state matrix, action and available actions, reward and reference to previous and next states transitions.<br>
 *
 */
public class StateTransition implements Serializable, Comparable<StateTransition> {

    @Serial
    private static final long serialVersionUID = 3018272924414901045L;

    /**
     * Current environment state.
     *
     */
    public final EnvironmentState environmentState;

    /**
     * Action taken to move from current environment state to next state.
     *
     */
    public int action;

    /**
     * Immediate reward after taking specific action in current environment state.
     *
     */
    public double reward;

    /**
     * Previous StateTransition.
     *
     */
    public StateTransition previousStateTransition;

    /**
     * Next StateTransition.
     *
     */
    public StateTransition nextStateTransition;

    /**
     * Priority based on TD error.
     *
     */
    public double priority;

    /**
     * Importance sampling weight.
     *
     */
    public double importanceSamplingWeight;

    /**
     * State value.
     *
     */
    public double value;

    /**
     * TD target value.
     *
     */
    public double tdTarget;

    /**
     * TD error.
     *
     */
    public double tdError;

    /**
     * Advantage.
     *
     */
    public double advantage;

    /**
     * Constructor for StateTransition.
     *
     * @param environmentState current environment state.
     */
    public StateTransition(EnvironmentState environmentState) {
        this.environmentState = environmentState;
    }

    /**
     * Checks if environment state is final in episodic learning.
     *
     * @return returns true if state is final otherwise returns false.
     */
    public boolean isFinalState() {
        return nextStateTransition == null;
    }

    /**
     * Returns next StateTransition based on current StateTransition.
     *
     * @param environmentState current StateTransition.
     * @return next StateTransition.
     */
    public StateTransition getNextStateTransition(EnvironmentState environmentState) {
        StateTransition newStateTransition = new StateTransition(environmentState);
        this.nextStateTransition = newStateTransition;
        newStateTransition.previousStateTransition = this;
        return newStateTransition;
    }

    /**
     * Compares this StateTransition to other StateTransition.
     *
     * @param otherStateTransition StateTransition to be compared.
     * @return true if StateTransitions are equal otherwise false.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public boolean equals(StateTransition otherStateTransition) throws MatrixException {
        if (!compare(environmentState.state(), otherStateTransition.environmentState.state())) return false;
        if (action != otherStateTransition.action) return false;
        if (reward != otherStateTransition.reward) return false;
        if (nextStateTransition == null && otherStateTransition.nextStateTransition == null) return true;
        else {
            if (nextStateTransition == null || otherStateTransition.nextStateTransition == null) return false;
            else return compare(nextStateTransition.environmentState.state(), otherStateTransition.nextStateTransition.environmentState.state());
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

    /**
     * Compares this StateTransition to other StateTransition.<br>
     * If other StateTransition is precedent to this StateTransition returns 1.<br>
     * If other StateTransition succeeds this StateTransition returns -1.<br>
     * Otherwise returns 0.<br>
     *
     * @param otherStateTransition other StateTransition.
     * @return return value of comparison.
     */
    public int compareTo(StateTransition otherStateTransition) {
        return environmentState.compareTo(otherStateTransition.environmentState);
    }

    /**
     * Prints state transition.
     *
     */
    public void print() {
        environmentState.print();
        System.out.println("Action: " + action + " Reward: " + reward + " Value: " + value + " TD target: " + tdTarget + " TD error: " + tdError + " Advantage: " + advantage);
    }

    /**
     * Print state transition chain.
     *
     * @param forward if true prints forward direction otherwise prints backward direction.
     */
    public void print(boolean forward) {
        print();
        if (forward) {
            if (nextStateTransition != null) nextStateTransition.print(true);
        }
        else {
            if (previousStateTransition != null) previousStateTransition.print(false);
        }
    }
}
