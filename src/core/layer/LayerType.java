/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
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
     * Dueling layer
     *
     */
    DUELING,

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
     * Single Convolution layer.
     *
     */
    SINGLE_CONVOLUTION,

    /**
     * Single Cross-correlation layer.
     *
     */
    SINGLE_CROSSCORRELATION,

    /**
     * Single Depth-wise Convolution layer.
     *
     */
    SINGLE_DW_CONVOLUTION,

    /**
     * Single Depth-wise Cross-correlation layer.
     *
     */
    SINGLE_DW_CROSSCORRELATION,

    /**
     * Single Point-wise Convolution layer.
     *
     */
    SINGLE_PW_CONVOLUTION,

    /**
     * Single Point-wise Cross-correlation layer.
     *
     */
    SINGLE_PW_CROSSCORRELATION,

    /**
     * Single Max pooling layer
     *
     */
    SINGLE_MAX_POOLING,

    /**
     * Single Random pooling layer
     *
     */
    SINGLE_RANDOM_POOLING,

    /**
     * Single Cyclic pooling layer
     *
     */
    SINGLE_CYCLIC_POOLING,

    /**
     * Single Average pooling layer
     *
     */
    SINGLE_AVERAGE_POOLING,

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
     * Layer normalization layer
     *
     */
    INSTANCE_NORMALIZATION,

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
    WEIGHT_NOISING,

    /**
     * Connect layer.
     *
     */
    CONNECT,

    /**
     * Join layer.
     *
     */
    JOIN,

    /**
     * Add layer.
     *
     */
    ADD,

    /**
     * Subtract layer.
     *
     */
    SUBTRACT,

    /**
     * Element-wise multiply layer.
     *
     */
    MULTIPLY,

    /**
     * Dot layer.
     *
     */
    DOT,

    /**
     * Divide layer.
     *
     */
    DIVIDE,

    /**
     * Transform layer.
     *
     */
    TRANSFORM,

    /**
     * Additive Attention layer.
     *
     */
    ADDITIVE_ATTENTION,

    /**
     * Dot Attention layer.
     *
     */
    DOT_ATTENTION,

    /**
     * Location-Based Attention layer.
     *
     */
    LOCATION_BASED_ATTENTION,

    /**
     * Positional encoding layer.
     *
     */
    POSITIONAL_ENCODING

}
