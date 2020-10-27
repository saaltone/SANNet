/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package core.layer.recurrent;

import core.NeuralNetworkException;
import core.layer.AbstractExecutionLayer;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.Sequence;
import utils.matrix.Initialization;
import utils.matrix.MatrixException;

import java.util.HashMap;

/**
 * Class that implements functions specific and common for all recurrent layers.
 *
 */
public abstract class AbstractRecurrentLayer extends AbstractExecutionLayer {

    /**
     * Flag if state is reset prior start of next training sequence.
     *
     */
    protected boolean resetStateTraining = false;

    /**
     * Flag if state is reset prior start of next test (validate, predict) sequence.
     *
     */
    protected boolean resetStateTesting = false;

    /**
     * Flag if state is restored prior start of next training phase.
     *
     */
    protected boolean restoreStateTraining = false;

    /**
     * Flag if state is restored prior start of next test (validate, predict) phase.
     *
     */
    protected boolean restoreStateTesting = false;

    /**
     * Previous state;
     *
     */
    private transient boolean previousState = false;

    /**
     * Limits number of backward propagation sequence steps.
     *
     */
    protected int truncateSteps = -1;

    /**
     * Constructor for AbstractRecurrentLayer.
     *
     * @param layerIndex layer Index.
     * @param initialization initialization function.
     * @param params parameters for neural network layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception setting of activation function fails.
     */
    protected AbstractRecurrentLayer(int layerIndex, Initialization initialization, String params) throws DynamicParamException, NeuralNetworkException {
        super (layerIndex, initialization, params);
        setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for AbstractRecurrentLayer.
     *
     * @return parameters used for AbstractRecurrentLayer.
     */
    public HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>(super.getParamDefs());
        paramDefs.put("resetStateTraining", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("resetStateTesting", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("restoreStateTraining", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("restoreStateTesting", DynamicParam.ParamType.BOOLEAN);
        return paramDefs;
    }

    /**
     * Sets parameters used for AbstractRecurrentLayer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - resetStateTraining: true if output is reset prior training forward step start otherwise false (default value false).<br>
     *     - resetStateTesting: true if output is reset prior test forward step start otherwise false (default value false).<br>
     *     - restoreStateTraining: true if output is restored prior training phase otherwise false (default value false).<br>
     *     - restoreStateTesting: true if output is restored prior test phase otherwise false (default value false).<br>
     *
     * @param params parameters used for AbstractRecurrentLayer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("resetStateTraining")) resetStateTraining = params.getValueAsBoolean("resetStateTraining");
        if (params.hasParam("resetStateTesting")) resetStateTesting = params.getValueAsBoolean("resetStateTesting");
        if (params.hasParam("restoreStateTraining")) restoreStateTraining = params.getValueAsBoolean("restoreStateTraining");
        if (params.hasParam("restoreStateTesting")) restoreStateTesting = params.getValueAsBoolean("restoreStateTesting");
    }

    /**
     * Checks if layer is recurrent layer type.
     *
     * @return always true.
     */
    public boolean isRecurrentLayer() { return true; }

    /**
     * Checks if layer is convolutional layer type.
     *
     * @return always false.
     */
    public boolean isConvolutionalLayer() { return false; }

    /**
     * Sets if recurrent inputs of layer are allowed to be reset during training.
     *
     * @param resetStateTraining if true allows reset.
     */
    public void resetStateTraining(boolean resetStateTraining) {
        this.resetStateTraining = resetStateTraining;
    }

    /**
     * Sets if recurrent inputs of layer are allowed to be reset during testing.
     *
     * @param resetStateTesting if true allows reset.
     */
    public void resetStateTesting(boolean resetStateTesting) {
        this.resetStateTesting = resetStateTesting;
    }

    /**
     * Sets if recurrent inputs of layer are allowed to be restored during training.
     *
     * @param restoreStateTraining if true allows restore.
     */
    public void restoreStateTraining(boolean restoreStateTraining) {
        this.restoreStateTraining = restoreStateTraining;
    }

    /**
     * Sets if recurrent inputs of layer are allowed to be restored during testing.
     *
     * @param restoreStateTesting if true allows restore.
     */
    public void restoreStateTesting(boolean restoreStateTesting) {
        this.restoreStateTesting = restoreStateTesting;
    }

    /**
     * Resets layer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void resetLayer() throws MatrixException {
        procedure.reset(resetStateTraining || resetStateTesting);
        resetLayerOutputs();
        resetNormalization();
    }

    /**
     * Takes single forward processing step process layer input(s).<br>
     * Additionally applies any normalization or regularization defined for layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void forwardProcess() throws MatrixException, DynamicParamException {
        Sequence previousOutputs = prepareForwardProcess();

        if (previousState != isTraining()) {
            procedure.reset(true);
            if ((previousState && restoreStateTraining) || ((!previousState && restoreStateTesting))) {
                procedure.storeDependencies(previousState ? 1 : 2);
            }
            if ((isTraining() && restoreStateTraining) || ((!isTraining() && restoreStateTesting))) {
                procedure.restoreDependencies(isTraining() ? 1 : 2);
            }
        }

        executeForwardProcess(previousOutputs);

        previousState = isTraining();
    }

    /**
     * Executes backward process step.
     *
     * @param nextLayerGradients next layer gradients.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected void executeBackwardProcess(Sequence nextLayerGradients) throws MatrixException, DynamicParamException {
        procedure.calculateGradient(nextLayerGradients, getLayerGradients(), truncateSteps);
    }

}
