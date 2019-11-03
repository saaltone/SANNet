/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.layer;

import core.NeuralNetworkException;
import core.activation.ActivationFunction;
import core.loss.LossFunction;
import utils.*;
import utils.matrix.*;

/**
 * Defines class for output layer of neural network.
 *
 */
public class OutputLayer extends AbstractLayer {

    /**
     * Neural network loss function for output layer.
     *
     */
    private LossFunction lossFunction = new LossFunction(BinaryFunctionType.MEAN_SQUARED_ERROR);

    /**
     * Neural network output error.
     *
     */
    private transient Matrix error;

    /**
     * Target (actual true) output values for error calculation in training phase.
     *
     */
    private transient Sequence targets;

    /**
     * Output error gradient.
     *
     */
    private transient Sequence douts;

    /**
     * Constructor for output layer.
     *
     * @param layerIndex index of layer.
     * @param layerType type of output layer.
     * @param activationFunction activation function for output layer.
     * @param initialization initialization function for output layer.
     * @param params parameters for output layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public OutputLayer(int layerIndex, LayerType layerType, ActivationFunction activationFunction, Init initialization, String params) throws NeuralNetworkException, DynamicParamException {
        super(layerIndex);
        super.setExecutionLayer(LayerFactory.create(layerType, this, activationFunction, initialization, params));
    }

    /**
     * Sets loss function for neural network (output layer).
     *
     * @param lossFunction loss function of neural network.
     */
    public void setLossFunction(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }

    /**
     * Clears targets (actual true output values) of output layer.<br>
     * In error calculation predicted output is compared to actual true output values.<br>
     *
     */
    public void clearTargets() {
        targets = null;
    }

    /**
     * Sets targets (actual true output values) of neural network (output layer).<br>
     * In error calculation predicted output is compared to actual true output values.<br>
     *
     * @param target targets of output layer.
     */
    public void setTargets(Sequence target) {
        clearTargets();
        targets = new Sequence(target.getDepth());
        targets.putAll(target);
    }

    /**
     * Resets error of neural network (output layer).
     *
     */
    public void resetError() {
        error = new DMatrix(getWidth(), 1);
    }

    /**
     * Returns total error of neural network including impact of regularization.
     *
     * @return total error of neural network.
     */
    public double getTotalError() {
        return error.mean() + getBackward().error() / (double)targets.totalSize();
    }

    /**
     * Executes backward step of neural network.
     *
     * @throws NeuralNetworkException throws exception if targets are not set or output and target dimensions are not matching.
     */
    public void backward() throws NeuralNetworkException  {
        if (targets.isEmpty()) throw new NeuralNetworkException("No targets defined");
        if (targets.totalSize() != getOuts().totalSize()) throw new NeuralNetworkException("Target size: "+ targets.totalSize() + " is not matching with output size: " + getOuts().totalSize());
        super.backward();
    }

    /**
     * Executes backward step of neural network.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void backwardProcess() throws MatrixException {
        calculateOutputDeltas();
        super.backwardProcess();
    }

    /**
     * Updates output error of neural network (output layer).
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if output and target dimensions are not matching.
     */
    public void updateOutputError() throws MatrixException, NeuralNetworkException {
        if (targets == null) return;
        if (targets.totalSize() != getOuts().totalSize()) throw new NeuralNetworkException("Target size: "+ targets.totalSize() + " is not matching with output size: " + getOuts().totalSize());
        for (Integer sampleIndex : targets.keySet()) {
            for (Integer matrixIndex : targets.sampleKeySet()) {
                Matrix loss = getOuts().get(sampleIndex, matrixIndex).applyBi(targets.get(sampleIndex, matrixIndex), lossFunction.getFunction());
                error.add(loss, error);
            }
        }
        error.divide(targets.totalSize(), error);
    }

    /**
     * Calculates output error gradient.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private void calculateOutputDeltas() throws MatrixException {
        douts = new Sequence(targets.getDepth());
        for (Integer sampleIndex : targets.keySet()) {
            for (Integer matrixIndex : targets.sampleKeySet()) {
                douts.put(sampleIndex, matrixIndex, getOuts().get(sampleIndex, matrixIndex).applyBi(targets.get(sampleIndex, matrixIndex), lossFunction.getDerivative()));
            }
        }
    }

    /**
     * Returns gradients of output error.
     *
     * @return gradients of output error.
     */
    public Sequence getdEosN() {
        return douts;
    }

}
