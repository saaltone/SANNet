/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.layer.recurrent;

import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.Initialization;
import utils.matrix.MatrixException;
import utils.procedure.Procedure;
import utils.procedure.ProcedureFactory;

/**
 * Implements bidirectional gated recurrent unit (GRU).<br>
 * <br>
 * Reference: https://en.wikipedia.org/wiki/Gated_recurrent_unit and https://github.com/erikvdplas/gru-rnn<br>
 * <br>
 * Equations applied for forward operation:<br>
 *     z = sigmoid(Wz * x + Uz * out(t-1) + bz) → Update gate<br>
 *     r = sigmoid(Wr * x + Ur * out(t-1) + br) → Reset gate<br>
 *     h = tanh(Wh * x + Uh * out(t-1) * r + bh) → Input activation<br>
 *     s = (1 - z) x h + z x out(t-1) → Internal state<br>
 *
 */
public class BiGRULayer extends GRULayer {

    /**
     * Reverse weight set.
     *
     */
    private GRUWeightSet reverseWeightSet = null;

    /**
     * Constructor for bidirectional GRU layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for bidirectional GRU layer.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public BiGRULayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, true, params);
    }

    /**
     * Returns reversed procedure.
     *
     * @return reversed procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected Procedure getReverseProcedure() throws MatrixException, DynamicParamException {
        currentWeightSet = reverseWeightSet;
        Procedure reverseProcedure = new ProcedureFactory().getProcedure(this, reverseWeightSet.getWeights(), getConstantMatrices(), getStopGradients(), null, isJoinedInput());
        currentWeightSet = weightSet;
        return reverseProcedure;
    }

    /**
     * Initializes neural network layer weights.
     *
     */
    public void initializeWeights() {
        super.initializeWeights();
        reverseWeightSet = new GRUWeightSet(initialization, getPreviousLayerWidth(), getInternalLayerWidth(), getRegulateDirectWeights(), getRegulateRecurrentWeights());
    }

}
