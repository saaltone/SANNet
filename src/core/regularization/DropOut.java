/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.regularization;

import utils.DynamicParam;
import utils.DynamicParamException;
import utils.Sequence;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Implements drop out regularization method for layer weights (parameters).<br>
 * Drop out is based on stochastic selection of layer nodes that are removed from training process at each training step.<br>
 * This forces other nodes to take over learning process reducing neural network's tendency to overfit.<br>
 * <br>
 * Reference: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf<br>
 *
 */

public class DropOut implements Regularization, Serializable {

    private static final long serialVersionUID = 1335548498128292515L;

    /**
     * If true neural network is in state otherwise false.
     *
     */
    private transient boolean isTraining;

    /**
     * Drop out probability of node.
     *
     */
    private double probability = 0.5;

    /**
     * Constructor for drop out class.
     *
     */
    public DropOut() {
    }

    /**
     * Constructor for drop out class.
     *
     * @param params parameters for drop out.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public DropOut(String params) throws DynamicParamException {
        this.setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for drop out.
     *
     * @return parameters used for drop out.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("probability", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
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
     * Not used.
     *
     */
    public void reset() {}

    /**
     * Sets flag for drop out if neural network is in training state.
     *
     * @param isTraining if true neural network is in state otherwise false.
     */
    public void setTraining(boolean isTraining) {
        this.isTraining = isTraining;
    }

    /**
     * Implements forward step for drop out.<br>
     * Function selectively masks out certain percentage of node governed by parameter probability during training phase.<br>
     * During inference phase it removes masking and compensates all weight by multipliying by probability.<br>
     *
     * @param sequence input sequence.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forward(Sequence sequence) throws MatrixException {
        for (Integer sampleIndex : sequence.keySet()) {
            for (Integer entryIndex : sequence.sampleKeySet()) {
                Matrix matrix = sequence.get(sampleIndex).get(entryIndex);
                if (isTraining) {
                    matrix.unsetScalingConstant();
                    matrix.setMask();
                    matrix.getMask().setMaskProba(probability);
                    matrix.getMask().maskRowByProba();
                }
                else {
                    matrix.setScalingConstant(probability);
                    matrix.unsetMask();
                }
            }
        }
    }

    /**
     * Not used.
     *
     * @param W weight matrix.
     */
    public void forward(Matrix W) {
    }

    /**
     * Not used.
     *
     * @param W weight matrix.
     * @return not used.
     */
    public double error(Matrix W) {
        return 0;
    }

    /**
     * Not used.
     *
     * @param W weight matrix.
     * @param dWSum gradient sum of weight.
     */
    public void backward(Matrix W, Matrix dWSum) {
    }

}
