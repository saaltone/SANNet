/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.layer.recurrent;

import core.layer.AbstractExecutionLayer;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Initialization;

/**
 * Implements abstract recurrent layer providing functions common for all recurrent layers.<br>
 *
 */
public abstract class AbstractRecurrentLayer extends AbstractExecutionLayer {

    /**
     * Parameter name types for abstract recurrent layer.
     *     - truncateSteps: number of sequence steps taken in backpropagation phase (default -1 i.e. not used).<br>
     *     - reversedInput: if true layer input is reversed otherwise not. Default value false.<br>
     *
     */
    private final static String paramNameTypes = "(truncateSteps:INT), " +
            "(reversedInput:BOOLEAN)";

    /**
     * Limits number of backward propagation sequence steps.
     *
     */
    protected int truncateSteps;

    /**
     * If true input is reversed for the layer.
     *
     */
    private boolean reversedInput;

    /**
     * Constructor for abstract recurrent layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function.
     * @param params parameters for neural network layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception setting of activation function fails.
     */
    protected AbstractRecurrentLayer(int layerIndex, Initialization initialization, String params) throws DynamicParamException, NeuralNetworkException {
        super (layerIndex, initialization, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        truncateSteps = -1;
        reversedInput = false;
    }

    /**
     * Returns parameters used for abstract recurrent layer.
     *
     * @return parameters used for abstract recurrent layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + AbstractRecurrentLayer.paramNameTypes;
    }

    /**
     * Sets parameters used for abstract recurrent layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - truncateSteps: number of sequence steps taken in backpropagation phase (default -1 i.e. not used).<br>
     *     - reversedInput: if true layer input is reversed otherwise not. Default value false.<br>
     *
     * @param params parameters used for abstract recurrent layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("truncateSteps")) {
            truncateSteps = params.getValueAsInteger("truncateSteps");
            if (truncateSteps < 0) throw new NeuralNetworkException("Truncate steps cannot be less than 0.");
        }
        if (params.hasParam("reversedInput")) reversedInput = params.getValueAsBoolean("reversedInput");
    }

    /**
     * Checks if layer is recurrent layer type.
     *
     * @return always true.
     */
    public boolean isRecurrentLayer() { return true; }

    /**
     * Checks if layer works with recurrent layers.
     *
     * @return if true layer works with recurrent layers otherwise false.
     */
    public boolean worksWithRecurrentLayer() {
        return true;
    }

    /**
     * Check if layer input is reversed.
     *
     * @return if true input layer input is reversed otherwise not.
     */
    public boolean isReversedInput() { return reversedInput; }

    /**
     * Returns true if input is joined otherwise returns false.
     *
     * @return true if input is joined otherwise returns false.
     */
    protected boolean isJoinedInput() {
        return false;
    }

    /**
     * Returns width of neural network layer.
     *
     * @return width of neural network layer.
     */
    public int getLayerWidth() {
        return getInternalLayerWidth();
    }

    /**
     * Returns internal width of neural network layer.
     *
     * @return internal width of neural network layer.
     */
    protected int getInternalLayerWidth() {
        return super.getLayerWidth();
    }

    /**
     * Returns number of truncated steps for gradient calculation. -1 means no truncation.
     *
     * @return number of truncated steps.
     */
    protected int getTruncateSteps() {
        return truncateSteps;
    }

    /**
     * Returns number of layer parameters.
     *
     * @return number of layer parameters.
     */
    protected int getNumberOfParameters() {
        return getWeightSet().getNumberOfParameters();
    }

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        return "Reversed input: " + (isReversedInput() ? "Yes" : "No");
    }

}
