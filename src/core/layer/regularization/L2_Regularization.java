/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.layer.regularization;

import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.*;

/**
 * Implements L2 (ridge) regularization.<br>
 * <br>
 * Reference: https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c<br>
 *
 */
public class L2_Regularization extends AbstractLx_Regularization {

    /**
     * Constructor for L2 regularization layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for feedforward layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public L2_Regularization(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException {
        super (layerIndex, initialization, params);
    }

    /**
     * Applies regularization.
     *
     * @param weight weight
     * @param lambda lambda value
     * @return regularization result
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix applyRegularization(Matrix weight, double lambda) throws MatrixException {
        return weight.multiply(2 * lambda);
    }

}
