/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.layer.recurrent;

import core.network.NeuralNetworkException;
import core.layer.AbstractExecutionLayer;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.procedure.Procedure;
import utils.sampling.Sequence;
import utils.matrix.Initialization;
import utils.matrix.MatrixException;

import java.util.HashMap;
import java.util.Map;

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
     * Reverse procedure for layer. Procedure contains chain of forward and backward expressions.
     *
     */
    protected Procedure reverseProcedure = null;

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
     * Resets layer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void reset() throws MatrixException {
        super.reset();
        if (isBidirectional()) reverseProcedure.reset((isTraining() && resetStateTraining) || (!isTraining() && resetStateTesting));
    }

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
                if (isBidirectional()) reverseProcedure.storeDependencies(previousState ? 1 : 2);
            }
            if ((isTraining() && restoreStateTraining) || ((!isTraining() && restoreStateTesting))) {
                procedure.restoreDependencies(isTraining() ? 1 : 2);
                if (isBidirectional()) reverseProcedure.restoreDependencies(isTraining() ? 1 : 2);
            }
        }
        previousState = isTraining();

        super.forwardProcess();
        if (isBidirectional()) {
            Sequence layerReverseOutputs = new Sequence();
            reverseProcedure.calculateExpression(getPreviousLayerOutputs(), layerReverseOutputs);
            setLayerOutputs(Sequence.join(new Sequence[] { getLayerOutputs(), layerReverseOutputs }, true));
        }
    }

    /**
     * Takes single backward processing step to process layer output gradient(s) towards input.<br>
     * Applies automated backward (automatic gradient) procedure when relevant to layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void backwardProcess() throws MatrixException, DynamicParamException {
        if (!isBidirectional()) super.backwardProcess();
        else {
            Sequence nextLayerGradients = getNextLayerGradients();
            Sequence[] unjoinedNextLayerGradients = Sequence.unjoin(nextLayerGradients);
            Sequence layerGradients = new Sequence();
            procedure.calculateGradient(unjoinedNextLayerGradients[0], layerGradients, getTruncateSteps());
            Sequence reverseLayerGradients = new Sequence();
            reverseProcedure.calculateGradient(unjoinedNextLayerGradients[1], reverseLayerGradients, getTruncateSteps());
            Sequence updatedLayerGradients = new Sequence();
            for (Map.Entry<Integer, MMatrix> entry : layerGradients.entrySet()) {
                int sampleIndex = entry.getKey();
                MMatrix layerGradient = entry.getValue();
                updatedLayerGradients.put(sampleIndex, layerGradient.add(reverseLayerGradients.get(sampleIndex)));
            }
            setLayerGradients(updatedLayerGradients);
        }
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
        HashMap<Matrix, Matrix> layerWeightGradients = new HashMap<>(procedure.getGradients());
        if (isBidirectional()) layerWeightGradients.putAll(reverseProcedure.getGradients());
        return layerWeightGradients;
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
