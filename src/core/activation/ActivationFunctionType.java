/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.activation;

/**
 * Defines supported activation functions.
 *
 */
public enum ActivationFunctionType {

    /**
     * Linear
     *
     */
    LINEAR,

    /**
     * Sigmoid
     *
     */
    SIGMOID,

    /**
     * Swish
     *
     */
    SWISH,

    /**
     * Hard sigmoid
     *
     */
    HARDSIGMOID,

    /**
     * Bi-polar sigmoid
     *
     */
    BIPOLARSIGMOID,
    /**
     * Scaled Hyperbolic tangent
     *
     */
    STANH,

    /**
     * Hyperbolic tangent
     *
     */
    TANH,
    /**
     * Hyperbolic tangent sigmoid
     *
     */
    TANHSIG,

    /**
     * Approximated tangent.
     *
     */
    TANHAPPR,

    /**
     * Hard tangent
     *
     */
    HARDTANH,

    /**
     * Exponential
     *
     */
    EXP,

    /**
     * Softplus
     *
     */
    SOFTPLUS,

    /**
     * Softsign
     *
     */
    SOFTSIGN,

    /**
     * Rectified Linear Unit (ReLU)
     *
     */
    RELU,

    /**
     * Cosine ReLU
     *
     */
    RELU_COS,

    /**
     * Sine ReLU
     *
     */
    RELU_SIN,

    /**
     * Exponential Linear Unit (ELU)
     *
     */
    ELU,

    /**
     * Scaled Exponential Linear Unit (SELU)
     *
     */
    SELU,

    /**
     * Gaussian Error Linear Units (GELU)
     *
     */
    GELU,

    /**
     * Softmax
     *
     */
    SOFTMAX,

    /**
     * Gaussian
     *
     */
    GAUSSIAN,

    /**
     * Sine activation
     *
     */
    SINACT,

    /**
     * Logit
     *
     */
    LOGIT,

    /**
     * Custom (user definable) function
     *
     */
    CUSTOM
    
}
