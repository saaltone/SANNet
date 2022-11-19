/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.layer.recurrent;

import core.activation.ActivationFunction;
import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.Initialization;
import utils.matrix.MatrixException;
import utils.procedure.Procedure;
import utils.procedure.ProcedureFactory;

/**
 * Implements bidirectional peephole Long Short Term Memory (LSTM)<br>
 * <br>
 * Reference: https://en.wikipedia.org/wiki/Long_short-term_memory<br>
 * <br>
 * Equations applied for forward operation:<br>
 *   i = sigmoid(Wi * x + Ui * c(t-1) + bi) → Input gate<br>
 *   f = sigmoid(Wf * x + Uf * c(t-1) + bf) → Forget gate<br>
 *   o = sigmoid(Wo * x + Uo * c(t-1) + bo) → Output gate<br>
 *   s = tanh(Ws * x + bs) → State update<br>
 *   c = i x s + f x c-1 → Internal cell state<br>
 *   h = tanh(c) x o or h = c x o → Output<br>
 *
 */
public class BiPeepholeLSTMLayer extends PeepholeLSTMLayer {

    /**
     * Reverse weight set.
     *
     */
    private PeepholeLSTMWeightSet reverseWeightSet = null;

    /**
     * Constructor for bidirectional peephole LSTM layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for bidirectional peephole LSTM layer.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public BiPeepholeLSTMLayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, true, params);
    }

    /**
     * Constructor for bidirectional peephole  LSTM layer.
     *
     * @param layerIndex layer index
     * @param activationFunction activation function used.
     * @param initialization initialization function for weight.
     * @param params parameters for bidirectional peephole LSTM layer.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public BiPeepholeLSTMLayer(int layerIndex, ActivationFunction activationFunction, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, activationFunction, initialization, true, params);
    }

    /**
     * Returns current weight set.
     *
     * @return current weight set.
     */
    protected WeightSet getCurrentWeightSet() {
        return reverseWeightSet;
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
        reverseWeightSet = new PeepholeLSTMWeightSet(initialization, getPreviousLayerWidth(), getInternalLayerWidth(), getRegulateDirectWeights(), getRegulateRecurrentWeights());
    }

}
