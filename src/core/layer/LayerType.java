/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.layer;

/**
 * Defines supported neural network layer types.<br>
 * <br>
 * Currently supports types are:
 *     FEEDFORWARD,
 *     RECURRENT,
 *     LSTM,
 *     PEEPHOLELSTM,
 *     GRAVESLSTM,
 *     GRU,
 *     MINGRU,
 *     CONVOLUTIONAL,
 *     POOLING
 *
 */
public enum LayerType {

    FEEDFORWARD,
    RECURRENT,
    LSTM,
    PEEPHOLELSTM,
    GRAVESLSTM,
    GRU,
    MINGRU,
    CONVOLUTIONAL,
    POOLING

}
