/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.layer;

import core.NeuralNetworkException;
import core.activation.ActivationFunction;
import core.loss.LossFunction;
import core.loss.LossFunctionType;
import utils.*;

import java.util.TreeMap;

/**
 * Defines class for output layer of neural network.
 *
 */
public class OutputLayer extends AbstractLayer {

    /**
     * Neural network loss function for output layer.
     *
     */
    private LossFunction lossFunction = new LossFunction(LossFunctionType.MEAN_SQUARED_ERROR);

    /**
     * Neural network output error.
     *
     */
    private transient Matrix error;

    /**
     * Target (actual true) output values for error calculation in training phase.
     *
     */
    private transient TreeMap<Integer, Matrix> targets;

    /**
     * Output error gradient.
     *
     */
    private transient TreeMap<Integer, Matrix> douts;

    /**
     * Constructor for output layer.
     *
     * @param layerIndex index of layer.
     * @param layerType type of output layer.
     * @param activationFunction activation function for output layer.
     * @param initialization initialization function for output layer.
     * @param params parameters for output layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public OutputLayer(int layerIndex, LayerType layerType, ActivationFunction activationFunction, Init initialization, String params) throws DynamicParamException {
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
        targets = new TreeMap<>();
    }

    /**
     * Sets targets (actual true output values) of neural network (output layer).<br>
     * In error calculation predicted output is compared to actual true output values.<br>
     *
     * @param target targets of output layer.
     */
    public void setTargets(TreeMap<Integer, Matrix> target) {
        clearTargets();
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
     * Gets total error of neural network including impact of regularization.
     *
     * @return total error of neural network.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public double getTotalError() throws MatrixException {
        return error.mean() + getBackward().error() / (double)targets.size();
    }

    /**
     * Executes backward step of neural network.
     *
     * @throws NeuralNetworkException throws exception if targets are not set or output and target dimensions are not matching.
     */
    public void backward() throws NeuralNetworkException  {
        if (targets.isEmpty()) throw new NeuralNetworkException("No targets defined");
        if (targets.size() != getOuts().size()) throw new NeuralNetworkException("Target size: "+ targets.size() + " is not matching with output size: " + getOuts().size());
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
        if (targets.isEmpty()) return;
        if (targets.size() != getOuts().size()) throw new NeuralNetworkException("Target size: "+ targets.size() + " is not matching with output size: " + getOuts().size());
        for (Integer index : targets.keySet()) {
            Matrix loss = targets.get(index).applyBi(getOuts().get(index), lossFunction.getFunction());
            error.add(loss, error);
        }
        error.divide(targets.size(), error);
    }

    /**
     * Calculates output error gradient.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private void calculateOutputDeltas() throws MatrixException {
        douts = new TreeMap<>();
        for (Integer index : targets.keySet()) {
            douts.put(index, targets.get(index).applyBi(getOuts().get(index), lossFunction.getDerivative()));
        }
    }

    /**
     * Returns gradients of output error.
     *
     * @return gradients of output error.
     */
    public TreeMap<Integer, Matrix> getdEosN() {
        return douts;
    }

}
