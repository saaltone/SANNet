/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.layer;

import utils.*;
import utils.matrix.Init;

import java.util.HashMap;

/**
 * Defines class for input layer of neural network.
 *
 */
public class InputLayer extends AbstractLayer {

    /**
     * Constructor for input layer.
     *
     * @param layerIndex index of layer.
     * @param params parameters for input layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public InputLayer(int layerIndex, String params) throws DynamicParamException {
        super(layerIndex);
        if (params != null) setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Executes parameter (weight) update for training step of neural network layer.
     *
     */
    public void update(){
        super.update();
        waitToComplete();
    }

    /**
     * Returns parameters used for input layer.
     *
     * @return parameters used for input layer.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("width", DynamicParam.ParamType.INT);
        paramDefs.put("height", DynamicParam.ParamType.INT);
        paramDefs.put("depth", DynamicParam.ParamType.INT);
        return paramDefs;
    }

    /**
     * Sets parameters used for input layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - width: width (number of nodes) for input layer.<br>
     *     - height: height for input layer (relevant only for convolutional layers).<br>
     *     - depth: depth for input layer (relevant only for convolutional layers).<br>
     *
     * @param params parameters used for input layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("width")) setWidth(params.getValueAsInteger("width"));
        if (params.hasParam("height")) setHeight(params.getValueAsInteger("height"));
        if (params.hasParam("depth")) setDepth(params.getValueAsInteger("depth"));
    }

    /**
     * Returns used initialization function.
     *
     * @return used initialization function.
     */
    public Init getInitialization() {
        return null;
    }

    /**
     * Executes forward processing step. Not relevant for input layer.
     *
     * @param training if true neural network is in training state otherwise neural network is in inference state.
     */
    protected void forwardProcess(boolean training) {}

    /**
     * Executes backward processing step. Not relevant for input layer.
     *
     */
    protected void backwardProcess() {}

    /**
     * Updates output error, relevant only for output layer.
     *
     */
    public void updateOutputError() {}

    /**
     * Returns gradients of next neural network layer.
     *
     * @return gradients of next neural network layer
     */
    public Sequence getdEosN() {
        return getForward().getNLayer().getdEos();
    }

}

