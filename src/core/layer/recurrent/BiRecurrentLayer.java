/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.layer.recurrent;

import core.activation.ActivationFunction;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.Initialization;
import utils.matrix.MatrixException;
import utils.procedure.ProcedureFactory;

/**
 * Implements bidirectional basic simple recurrent layer.<br>
 * This layer is by nature prone to numerical instability with limited temporal memory.<br>
 *
 */
public class BiRecurrentLayer extends RecurrentLayer {

    /**
     * Reverse weight set.
     *
     */
    private RecurrentWeightSet reverseWeightSet = null;

    /**
     * Constructor for bidirectional recurrent layer.
     *
     * @param layerIndex layer index
     * @param activationFunction activation function used.
     * @param initialization initialization function for weight.
     * @param params parameters for bidirectional recurrent layer.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public BiRecurrentLayer(int layerIndex, ActivationFunction activationFunction, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, activationFunction, initialization, true, params);
    }

    /**
     * Initializes neural network layer weights.
     *
     */
    public void initializeWeights() {
        super.initializeWeights();
        reverseWeightSet = new RecurrentWeightSet(initialization, getPreviousLayerWidth(), getInternalLayerWidth(), getRegulateDirectWeights(), getRegulateRecurrentWeights());
    }

    /**
     * Defines layer procedure for forward and backward calculation (automatic gradient) by applying procedure factory.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected void defineProcedure() throws MatrixException, DynamicParamException, NeuralNetworkException {
        super.defineProcedure();
        currentWeightSet = reverseWeightSet;
        reverseProcedure = new ProcedureFactory().getProcedure(this, reverseWeightSet.getWeights(), getConstantMatrices(), getStopGradients(), true);
    }

}
