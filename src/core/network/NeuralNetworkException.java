/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.network;

/**
 * Class defining neural network exception thrown in neural network error situations.<br>
 *
 */
public class NeuralNetworkException extends Exception {

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
    public NeuralNetworkException(String cause) {
        this.cause = cause;
    }

    /**
     * Returns exception as string.
     *
     * @return exception as string.
     */
    public String toString() {
        return ("NeuralNetworkException occurred with cause: " + cause);
    }

}

