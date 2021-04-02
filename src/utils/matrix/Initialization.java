/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.matrix;

/**
 * Defines initialization for matrix:<br>
 * - ZERO: initialized with zero values.<br>
 * - ONE: initialized with one values.<br>
 * - RANDOM: initialized with random values between zero and one.<br>
 * - IDENTITY: initialized as identity matrix i.e. ones in diagonal and zero elsewhere.<br>
 * - NORMAL_XAVIER: initialized with normal (Gaussian) XAVIER initialization.<br>
 * - UNIFORM_XAVIER: initialized with uniform XAVIER initialization.<br>
 * - NORMAL_HE: initialized with normal (Gaussian) HE initialization.<br>
 * - UNIFORM_HE: initialized with uniform HE initialization.<br>
 * - NORMAL_LECUN: initialized with normal (Gaussian) LECUN initialization.<br>
 * - UNIFORM_LECUN: initialized with uniform LECUN initialization.<br>
 * - NORMAL_XAVIER_CONV: initialized with normal (Gaussian) XAVIER initialization for convolutional layer.<br>
 * - UNIFORM_XAVIER_CONV: initialized with uniform XAVIER initialization for convolutional layer.<br>
 * - NORMAL_HE_CONV: initialized with normal (Gaussian) HE initialization for convolutional layer.<br>
 * - UNIFORM_HE_CONV: initialized with uniform HE initialization for convolutional layer.<br>
 * - NORMAL_LECUN_CONV: initialized with normal (Gaussian) LECUN initialization for convolutional layer.<br>
 * - UNIFORM_LECUN_CONV: initialized with uniform LECUN initialization for convolutional layer.<br>
 * <br>
 * Reference: https://stats.stackexchange.com/questions/373136/softmax-weights-initialization<br>
 *
 */
public enum Initialization {

    ZERO,
    ONE,
    RANDOM,
    IDENTITY,
    NORMAL_XAVIER,
    UNIFORM_XAVIER,
    NORMAL_HE,
    UNIFORM_HE,
    NORMAL_LECUN,
    UNIFORM_LECUN,
    NORMAL_XAVIER_CONV,
    UNIFORM_XAVIER_CONV,
    NORMAL_HE_CONV,
    UNIFORM_HE_CONV,
    NORMAL_LECUN_CONV,
    UNIFORM_LECUN_CONV

}
