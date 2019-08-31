/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement;

/**
 * Class defining agent exception thrown in agent error situations.
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
