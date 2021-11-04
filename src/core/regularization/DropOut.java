/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.regularization;

import utils.configurable.Configurable;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.sampling.Sequence;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;

/**
 * Implements drop out regularization method for layer weights (parameters).<br>
 * Drop out is based on stochastic selection of layer nodes that are removed from training process at each training step.<br>
 * This forces other nodes to take over learning process reducing neural network's tendency to overfit.<br>
 * <br>
 * Reference: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf<br>
 *
 */
public class DropOut implements Configurable, Regularization, Serializable {

    @Serial
    private static final long serialVersionUID = 1335548498128292515L;

    /**
     * Parameter name types for DropOut.
     *     - probability: probability of masking out a layer node. Default value 0.5.<br>
     *
     */
    private final static String paramNameTypes = "(probability:DOUBLE)";

    /**
     * Type of regularization.
     *
     */
    private final RegularizationType regularizationType = RegularizationType.DROPOUT;

    /**
     * If true neural network is in state otherwise false.
     *
     */
    private transient boolean isTraining;

    /**
     * Drop out probability of node.
     *
     */
    private double probability;

    /**
     * Constructor for drop out class.
     *
     */
    public DropOut() {
        initializeDefaultParams();
    }

    /**
     * Constructor for drop out class.
     *
     * @param params parameters for drop out.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public DropOut(String params) throws DynamicParamException {
        this();
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        probability = 0.5;
    }

    /**
     * Returns parameters used for drop out.
     *
     * @return parameters used for drop out.
     */
    public String getParamDefs() {
        return DropOut.paramNameTypes;
    }

    /**
     * Sets parameters used for drop out.<br>
     * <br>
     * Supported parameters are:<br>
     *     - probability: probability of masking out a layer node. Default value 0.5.<br>
     *
     * @param params parameters used for drop out.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("probability")) probability = 1 - params.getValueAsDouble("probability");
    }

    /**
     * Sets flag for drop out if neural network is in training state.
     *
     * @param isTraining if true neural network is in state otherwise false.
     */
    public void setTraining(boolean isTraining) {
        this.isTraining = isTraining;
    }

    /**
     * Implements forward step for inverted drop out.<br>
     * Function selectively masks out certain percentage of node governed by parameter probability during training phase.<br>
     * During training phase it also compensates all remaining inputs by dividing by probability.<br>
     *
     * @param sequence input sequence.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forward(Sequence sequence) throws MatrixException {
        for (Integer sampleIndex : sequence.keySet()) {
            for (Integer entryIndex : sequence.sampleKeySet()) {
                forward (sequence.get(sampleIndex).get(entryIndex));
            }
        }
    }

    /**
     * Implements forward step for inverted drop out.<br>
     * Function selectively masks out certain percentage of node governed by parameter probability during training phase.<br>
     * During training phase it also compensates all remaining inputs by dividing by probability.<br>
     *
     * @param inputs inputs.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forward(MMatrix inputs) throws MatrixException {
        for (Integer index : inputs.keySet()) forward (inputs.get(index));
    }

    /**
     * Implements forward step for inverted drop out.<br>
     * Function selectively masks out certain percentage of node governed by parameter probability during training phase.<br>
     * During training phase it also compensates all remaining inputs by dividing by probability.<br>
     *
     * @param matrix matrix
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private void forward(Matrix matrix) throws MatrixException {
        if (isTraining) {
            matrix.multiply(1 / probability, matrix);
            matrix.setMask();
            matrix.getMask().setProbability(probability);
            matrix.getMask().maskRowByProbability();
        }
    }

    /**
     * Not used.
     *
     * @param weight weight matrix.
     * @return not used.
     */
    public double error(Matrix weight) {
        return 0;
    }

    /**
     * Not used.
     *
     * @param weight weight matrix.
     * @param weightGradientSum gradient sum of weight.
     */
    public void backward(Matrix weight, Matrix weightGradientSum) {
    }

    /**
     * Returns name of regularization.
     *
     * @return name of regularization.
     */
    public String getName() {
        return regularizationType.toString();
    }

}
