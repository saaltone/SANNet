/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.layer;

/**
 * Enum for supported neural network layer types.<br>
 * Currently supportes types are:
 *     FEEDFORWARD,
 *     RECURRENT,
 *     LSTM,
 *     GRAVESLSTM,
 *     GRU,
 *     CONVOLUTIONAL,
 *     POOLING
 *
 */
public enum LayerType {

    FEEDFORWARD,
    RECURRENT,
    LSTM,
    GRAVESLSTM,
    GRU,
    CONVOLUTIONAL,
    POOLING

}
