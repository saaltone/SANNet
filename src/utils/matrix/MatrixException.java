/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.matrix;

/**
 * Class defining matrix exception thrown in matrix operation error situations.<br>
 *
 */
public class MatrixException extends Exception {

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
    public MatrixException(String cause) {
        this.cause = cause;
    }

    /**
     * Returns exception as string.
     *
     * @return exception as string.
     */
    public String toString() {
        return ("MatrixException occurred with cause: " + cause);
    }

}

