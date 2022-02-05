/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.layer.recurrent;

import core.activation.ActivationFunction;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParamException;
import utils.matrix.Initialization;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.procedure.Procedure;
import utils.procedure.ProcedureFactory;
import utils.sampling.Sequence;

import java.util.HashMap;

/**
 * Implements bidirectional Graves type of Long Short Term Memory (LSTM)<br>
 * <br>
 * Reference: https://www.cs.toronto.edu/~graves/phd.pdf<br>
 * <br>
 * Equations applied for forward operation:<br>
 *   i = sigmoid(Wi * x + Ui * out(t-1) + Ci * c(t-1) + bi) → Input gate<br>
 *   f = sigmoid(Wf * x + Uf * out(t-1) + Cf * c(t-1) + bf) → Forget gate<br>
 *   s = tanh(Ws * x + Us * out(t-1) + bs) → State update<br>
 *   c = i x s + f x c(t-1) → Internal cell state<br>
 *   o = sigmoid(Wo * x + Uo * out(t-1) + Co * ct + bo) → Output gate<br>
 *   h = tanh(c) x o or h = c x o → Output<br>
 *
 */
public class BiGravesLSTMLayer extends GravesLSTMLayer {

    /**
     * Reverse weight set.
     *
     */
    private GravesLSTMWeightSet reverseWeightSet = null;

    /**
     * Reverse procedure for layer. Procedure contains chain of forward and backward expressions.
     *
     */
    protected Procedure reverseProcedure = null;

    /**
     * Layer gradients for reverse sequence.
     *
     */
    private transient Sequence reverseLayerGradients;

    /**
     * Constructor for bidirectional Graves type LSTM layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for bidirectional Graves type LSTM layer.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public BiGravesLSTMLayer(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, initialization, true, params);
    }

    /**
     * Constructor for bidirectional Graves type LSTM layer.
     *
     * @param layerIndex layer index
     * @param activationFunction activation function used.
     * @param initialization initialization function for weight.
     * @param params parameters for bidirectional Graves type LSTM layer.
     * @throws NeuralNetworkException throws exception setting of activation function fails or layer dimension requirements are not met.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public BiGravesLSTMLayer(int layerIndex, ActivationFunction activationFunction, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        super (layerIndex, activationFunction, initialization, true, params);
    }

    /**
     * Initializes neural network layer weights.
     *
     */
    public void initializeWeights() {
        super.initializeWeights();
        reverseWeightSet = new GravesLSTMWeightSet(initialization, getPreviousLayerWidth(), getInternalLayerWidth(), getRegulateDirectWeights(), getRegulateRecurrentWeights());
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

    /**
     * Resets layer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void resetLayer() throws MatrixException {
        super.resetLayer();
        if (reverseProcedure != null) reverseProcedure.reset((isTraining() && resetStateTraining) || (!isTraining() && resetStateTesting));
    }

    /**
     * Takes single forward processing step to process layer input(s).<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void forwardProcess() throws MatrixException, DynamicParamException {
        resetLayer();
        setLayerOutputs(procedure.calculateExpression(getPreviousLayerOutputs()));
        setLayerOutputs(getLayerOutputs().join(reverseProcedure.calculateExpression(getPreviousLayerOutputs()), true));
    }

    /**
     * Takes single backward processing step to process layer output gradient(s) towards input.<br>
     * Applies automated backward (automatic gradient) procedure when relevant to layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void backwardProcess() throws MatrixException, DynamicParamException {
        Sequence nextLayerGradients = getNextLayerGradients();
        Sequence directNextLayerGradients = nextLayerGradients.unjoin(0);
        Sequence reverseNextLayerGradients = nextLayerGradients.unjoin(1);
        if (procedure != null) setLayerGradients(procedure.calculateGradient(directNextLayerGradients, getTruncateSteps()));
        if (reverseProcedure != null) reverseLayerGradients = reverseProcedure.calculateGradient(reverseNextLayerGradients, getTruncateSteps());
        Sequence layerGradients = getLayerGradients();
        Sequence updatedLayerGradients = new Sequence();
        for (Integer sampleIndex : layerGradients.keySet()) {
            updatedLayerGradients.put(sampleIndex, layerGradients.get(sampleIndex).add(reverseLayerGradients.get(sampleIndex)));
        }
        setLayerGradients(updatedLayerGradients);
    }

    /**
     * Returns neural network weight gradients.
     *
     * @return neural network weight gradients.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public HashMap<Matrix, Matrix> getLayerWeightGradients() throws MatrixException {
        HashMap<Matrix, Matrix> layerWeightGradients = new HashMap<>(procedure.getGradients());
        layerWeightGradients.putAll(reverseProcedure.getGradients());
        return layerWeightGradients;
    }

    /**
     * Returns number of layer parameters.
     *
     * @return number of layer parameters.
     */
    protected int getNumberOfParameters() {
        return super.getWeightSet().getNumberOfParameters() + reverseWeightSet.getNumberOfParameters();
    }

}
