/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.reinforcement.agent;

import utils.matrix.Matrix;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashSet;

/**
 * Record that defines state of environment.
 *
 * @param state state of environment.
 * @param availableActions actions available at state.
 *
 */
public record EnvironmentState(Matrix state, HashSet<Integer> availableActions) implements Serializable {

    @Serial
    private static final long serialVersionUID = -3840329155579749639L;

    /**
     * Prints environment state.
     *
     */
    public void print() {
        state.print();
        System.out.println(availableActions);
    }

}
