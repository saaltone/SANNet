/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.regularization;

import core.layer.Connector;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.Matrix;
import utils.MatrixException;

import java.io.Serializable;
import java.util.HashMap;
import java.util.TreeMap;

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
     * Reference to connector between previous and next layer.
     *
     */
    private Connector connector;

    /**
     * True if next layer if hidden layer otherwise false.
     *
     */
    private boolean toHiddenLayer;

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
     * If true drop out masking is done on sample by sample.<br>
     * Otherwise same mask is used throughout single training batch.<br>
     *
     */
    private boolean maskBySample = false;

    /**
     * Constructor for drop out class.
     *
     * @param connector reference to connector between previous and next layer.
     * @param toHiddenLayer true if next layer if hidden layer otherwise false.
     */
    public DropOut(Connector connector, boolean toHiddenLayer) {
        this.connector = connector;
        this.toHiddenLayer = toHiddenLayer;
    }

    /**
     * Constructor for drop out class.
     *
     * @param connector reference to connector between previous and next layer.
     * @param toHiddenLayer true if next layer if hidden layer otherwise false.
     * @param params parameters for drop out.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public DropOut(Connector connector, boolean toHiddenLayer, String params) throws DynamicParamException {
        this(connector, toHiddenLayer);
        this.setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Gets parameters used for drop out.
     *
     * @return parameters used for drop out.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("probability", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("maskBySample", DynamicParam.ParamType.BOOLEAN);
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
        if (params.hasParam("maskBySample")) maskBySample = params.getValueAsBoolean("maskBySample");
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
     * Function selectively masks out certain percentage of node governed by parameter probability.<br>
     * It then compensates lost connections in other nodes by scaling up respectively.<br>
     *
     * @param ins input samples for forward step.
     * @param index executed only when index is -1 meaning for whole input sample batch at once.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forwardPre(TreeMap<Integer, Matrix> ins, int index) throws MatrixException {
        if (index != -1) return;
        if (!toHiddenLayer) return;
        for (Matrix W : connector.getReg()) {
            if (isTraining) {
                if (maskBySample || (!maskBySample && W.colMaskStackSize() == 0)) {
                    W.setMaskProba(probability);
                    W.setMask();
                    W.maskColByProba();
                    W.setScalingConstant(1 / probability);
                    W.stackColMask(false);
                }
            }
            else {
                W.setScalingConstant(probability);
            }
        }
    }

    /**
     * Unscales weight matrix for inference phase.
     *
     * @param outs output samples for forward step.
     */
    public void forwardPost(TreeMap<Integer, Matrix> outs) {
        if (!toHiddenLayer) return;
        if (!isTraining) {
            for (Matrix W : connector.getReg()) {
                W.unsetScalingConstant();
            }
        }
    }

    /**
     * Not used.
     *
     * @return not used.
     */
    public double error() {
        return 0;
    }

    /**
     * In case of mask sample by sample basis pops mask from stack per backpropagation step.
     *
     * @param index executed only when index is -1 meaning for whole input sample batch at once.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backward(int index) throws MatrixException {
        if (index != -1) return;
        if (!toHiddenLayer) return;
        if (isTraining) {
            for (Matrix W : connector.getReg()) {
                if (W.colMaskStackSize() > 0) W.unstackColMask();
            }
        }
    }

    /**
     * Prior weight update removes any drop out masking with this function.
     *
     */
    public void update() {
        if (!toHiddenLayer) return;
        if (isTraining) {
            for (Matrix W : connector.getReg()) {
                W.unsetMask();
                W.clearColMaskStack();
                W.unsetScalingConstant();
            }
        }
    }

}
