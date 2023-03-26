/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.matrix;

/**
 * Defines initialization for matrix.<br>
 * <br>
 * Reference: <a href="https://stats.stackexchange.com/questions/373136/softmax-weights-initialization">...</a><br>
 *
 */
public enum Initialization {

    /**
     * Initialized with zero values.
     *
     */
    ZERO,

    /**
     * Initialized with one values.
     *
     */
    ONE,

    /**
     * Initialized with uniform random values between zero and one.
     *
     */
    RANDOM,

    /**
     * Initialized as identity matrix i.e. matrix with one in diagonal and zeros elsewhere.
     *
     */
    IDENTITY,

    /**
     * Initialized with normal (Gaussian) Xavier initialization.
     *
     */
    NORMAL_XAVIER,

    /**
     * Initialized with uniform Xavier initialization.
     *
     */
    UNIFORM_XAVIER,

    /**
     * Initialized with normal (Gaussian) He-initialization.
     *
     */
    NORMAL_HE,

    /**
     * Initialized with uniform He-initialization.
     *
     */
    UNIFORM_HE,

    /**
     * Initialized with normal (Gaussian) Lecun initialization.
     *
     */
    NORMAL_LECUN,

    /**
     * Initialized with uniform Lecun initialization.
     *
     */
    UNIFORM_LECUN,

    /**
     * Initialized with normal (Gaussian) Xavier initialization for convolutional layer.
     *
     */
    NORMAL_XAVIER_CONV,

    /**
     * Initialized with uniform Xavier initialization for convolutional layer.
     *
     */
    UNIFORM_XAVIER_CONV,

    /**
     * Initialized with normal (Gaussian) He-initialization for convolutional layer.
     *
     */
    NORMAL_HE_CONV,

    /**
     * Initialized with uniform He-initialization for convolutional layer.
     *
     */
    UNIFORM_HE_CONV,

    /**
     * Initialized with normal (Gaussian) Lecun initialization for convolutional layer.
     *
     */
    NORMAL_LECUN_CONV,

    /**
     * Initialized with uniform Lecun initialization for convolutional layer.
     *
     */
    UNIFORM_LECUN_CONV

}
