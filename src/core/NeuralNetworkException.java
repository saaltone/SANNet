/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core;

/**
 * Class defining neural network exception thrown in neural network error situations.
 *
 */
public class NeuralNetworkException extends Exception {

    /**
     * Verbal text of exception cause.
     *
     */
    String cause;

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

