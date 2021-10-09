/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.reinforcement.agent;

/**
 * Class defining agent exception thrown in agent error situations.<br>
 *
 */
public class AgentException extends Exception {

    /**
     * Verbal text of exception cause.
     *
     */
    private final String cause;

    /**
     * Default constructor for exception.
     *
     * @param cause verbal text of exception cause.
     */
    public AgentException(String cause) {
        this.cause = cause;
    }

    /**
     * Returns exception as string.
     *
     * @return exception as string.
     */
    public String toString() {
        return ("AgentException occurred with cause: " + cause);
    }

}
