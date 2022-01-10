/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.layer;

/**
 * Defines supported neural network layer types.
 *
 */
public enum LayerType {

    /**
     * Feedforward layer
     *
     */
    FEEDFORWARD,

    /**
     * Dense layer
     *
     */
    DENSE,

    /**
     * Activation layer
     *
     */
    ACTIVATION,

    /**
     * Flatten layer.
     *
     */
    FLATTEN,

    /**
     * Vanilla recurrent layer
     *
     */
    RECURRENT,

    /**
     * Long Short Term Memory (LSTM) layer
     *
     */
    LSTM,

    /**
     * Peephole Long Short Term Memory (LSTM) layer
     *
     */
    PEEPHOLELSTM,

    /**
     * Graves style Long Short Term Memory (LSTM) layer
     *
     */
    GRAVESLSTM,

    /**
     * Gated Recurrent Unit (GRU) layer
     *
     */
    GRU,

    /**
     * Minimal Gated Recurrent Unit (GRU) layer
     *
     */
    MINGRU,

    /**
     * Bidirectional Vanilla recurrent layer
     *
     */
    BIRECURRENT,

    /**
     * Bidirectional Long Short Term Memory (LSTM) layer
     *
     */
    BILSTM,

    /**
     * Bidirectional Peephole Long Short Term Memory (LSTM) layer
     *
     */
    BIPEEPHOLELSTM,

    /**
     * Bidirectional Graves style Long Short Term Memory (LSTM) layer
     *
     */
    BIGRAVESLSTM,

    /**
     * Bidirectional Gated Recurrent Unit (GRU) layer
     *
     */
    BIGRU,

    /**
     * Bidirectional Minimal Gated Recurrent Unit (GRU) layer
     *
     */
    BIMINGRU,

    /**
     * Convolution layer.
     *
     */
    CONVOLUTION,

    /**
     * Cross-correlation layer.
     *
     */
    CROSSCORRELATION,

    /**
     * Winograd convolution layer.
     *
     */
    WINOGRAD_CONVOLUTION,

    /**
     * Depth-wise separable Convolution layer.
     *
     */
    DSCONVOLUTION,

    /**
     * Depth-wise separable Cross-correlation layer.
     *
     */
    DSCROSSCORRELATION,

    /**
     * Max pooling layer
     *
     */
    MAX_POOLING,

    /**
     * Random pooling layer
     *
     */
    RANDOM_POOLING,

    /**
     * Cyclic pooling layer
     *
     */
    CYCLIC_POOLING,

    /**
     * Average pooling layer
     *
     */
    AVERAGE_POOLING,

    /**
     * Batch normalization layer
     *
     */
    BATCH_NORMALIZATION,

    /**
     * Layer normalization layer
     *
     */
    LAYER_NORMALIZATION,

    /**
     * Weight normalization layer
     *
     */
    WEIGHT_NORMALIZATION,

    /**
     * Drop out layer
     *
     */
    DROPOUT,

    /**
     * Gradient clipping layer.
     *
     */
    GRADIENT_CLIPPING,

    /**
     * L1 regularization layer.
     *
     */
    L1_REGULARIZATION,

    /**
     * L1 regularization layer.
     *
     */
    L2_REGULARIZATION,

    /**
     * L1 regularization layer.
     *
     */
    Lp_REGULARIZATION,

    /**
     * Weight noising layer.
     *
     */
    WEIGHT_NOISING

}
