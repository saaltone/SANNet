/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.layer.recurrent;

import core.layer.AbstractExecutionLayer;
import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Initialization;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashMap;

/**
 * Implements abstract recurrent layer providing functions common for all recurrent layers.<br>
 *
 */
public abstract class AbstractRecurrentLayer extends AbstractExecutionLayer {

    /**
     * Parameter name types for abstract recurrent layer.
     *     - resetStateTraining: true if output is reset prior training forward step start otherwise false (default value false).<br>
     *     - resetStateTesting: true if output is reset prior test forward step start otherwise false (default value false).<br>
     *     - restoreStateTraining: true if output is restored prior training phase otherwise false (default value false).<br>
     *     - restoreStateTesting: true if output is restored prior test phase otherwise false (default value false).<br>
     *     - truncateSteps: number of sequence steps taken in backpropagation phase (default -1 i.e. not used).<br>
     *
     */
    private final static String paramNameTypes = "(resetStateTraining:BOOLEAN), " +
            "(resetStateTesting:BOOLEAN), " +
            "(restoreStateTraining:BOOLEAN), " +
            "(restoreStateTesting:BOOLEAN), " +
            "(truncateSteps:INT)";

    /**
     * Flag if state is reset prior start of next training sequence.
     *
     */
    protected boolean resetStateTraining;

    /**
     * Flag if state is reset prior start of next test (validate, predict) sequence.
     *
     */
    protected boolean resetStateTesting;

    /**
     * Flag if state is restored prior start of next training phase.
     *
     */
    protected boolean restoreStateTraining;

    /**
     * Flag if state is restored prior start of next test (validate, predict) phase.
     *
     */
    protected boolean restoreStateTesting;

    /**
     * Previous state;
     *
     */
    private transient boolean previousState = false;

    /**
     * Limits number of backward propagation sequence steps.
     *
     */
    protected int truncateSteps;

    /**
     * If true recurrent layer is bidirectional otherwise false
     *
     */
    private final boolean isBirectional;

    /**
     * Constructor for abstract recurrent layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function.
     * @param isBirectional if true recurrent layer is bidirectional otherwise false
     * @param params parameters for neural network layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception setting of activation function fails.
     */
    protected AbstractRecurrentLayer(int layerIndex, Initialization initialization, boolean isBirectional, String params) throws DynamicParamException, NeuralNetworkException {
        super (layerIndex, initialization, params);
        this.isBirectional = isBirectional;
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        resetStateTraining = false;
        resetStateTesting = false;
        restoreStateTraining = false;
        restoreStateTesting = false;
        truncateSteps = -1;
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
     *     - resetStateTraining: true if output is reset prior training forward step start otherwise false (default value false).<br>
     *     - resetStateTesting: true if output is reset prior test forward step start otherwise false (default value false).<br>
     *     - restoreStateTraining: true if output is restored prior training phase otherwise false (default value false).<br>
     *     - restoreStateTesting: true if output is restored prior test phase otherwise false (default value false).<br>
     *     - truncateSteps: number of sequence steps taken in backpropagation phase (default -1 i.e. not used).<br>
     *
     * @param params parameters used for abstract recurrent layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("resetStateTraining")) resetStateTraining = params.getValueAsBoolean("resetStateTraining");
        if (params.hasParam("resetStateTesting")) resetStateTesting = params.getValueAsBoolean("resetStateTesting");
        if (params.hasParam("restoreStateTraining")) restoreStateTraining = params.getValueAsBoolean("restoreStateTraining");
        if (params.hasParam("restoreStateTesting")) restoreStateTesting = params.getValueAsBoolean("restoreStateTesting");
        if (params.hasParam("truncateSteps")) {
            truncateSteps = params.getValueAsInteger("truncateSteps");
            if (truncateSteps < 0) throw new NeuralNetworkException("Truncate steps cannot be less than 0.");
        }
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
     * Check if layer is bidirectional.
     *
     * @return true if layer is bidirectional otherwise returns false.
     */
    public boolean isBidirectional() {
        return isBirectional;
    }

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
        return getInternalLayerWidth() * (!isBidirectional() ? 1 : 2);
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
     * Returns current weight set.
     *
     * @return current weight set.
     */
    protected abstract WeightSet getCurrentWeightSet();

    /**
     * Takes single forward processing step process layer input(s).<br>
     * Applies additionally any regularization defined for layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void forwardProcess() throws MatrixException, DynamicParamException {
        if (previousState != isTraining()) {
            procedure.reset(true);
            if ((previousState && restoreStateTraining) || ((!previousState && restoreStateTesting))) {
                procedure.storeDependencies(previousState ? 1 : 2);
            }
            if ((isTraining() && restoreStateTraining) || ((!isTraining() && restoreStateTesting))) {
                procedure.restoreDependencies(isTraining() ? 1 : 2);
            }
        }
        previousState = isTraining();

        super.forwardProcess();
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
     * Returns neural network weight gradients.
     *
     * @return neural network weight gradients.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public HashMap<Matrix, Matrix> getLayerWeightGradients() throws MatrixException {
        return procedure.getGradients();
    }

    /**
     * Returns number of layer parameters.
     *
     * @return number of layer parameters.
     */
    protected int getNumberOfParameters() {
        return getWeightSet().getNumberOfParameters() * (!isBidirectional() ? 1 : 2);
    }

}
