/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils;

/**
 * Class defining dynamic parameter exception thrown in dynamic parameter handling error situations.
 *
 */
public class DynamicParamException extends Exception {

    /**
     * Verbal text of exception cause.
     *
     */
    final String cause;

    /**
     * Default constructor for exception.
     *
     * @param cause verbal text of exception cause.
     */
    public DynamicParamException(String cause) {
        this.cause = cause;
    }

    /**
     * Returns exception as string.
     *
     * @return exception as string.
     */
    public String toString() {
        return ("DynamicParamException occurred with cause: " + cause);
    }

}

