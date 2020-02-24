/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.layer;

import core.NeuralNetworkException;
import core.activation.ActivationFunction;
import core.normalization.Normalization;
import utils.*;
import utils.matrix.*;
import utils.procedure.Procedure;
import utils.procedure.ProcedureFactory;

import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;

/**
 * Abstract class that implements execution layer for actual neural network layers (feed forward layer, recurrent layer etc.)<br>
 * Provides supportive functions for actual neural network layers.<br>
 * Support automatic gradient i.e. backward gradient calculation for layers supporting it.<br>
 *
 */
public abstract class AbstractExecutionLayer implements Layer, Serializable {

    private static final long serialVersionUID = -2696526850302490503L;

    /**
     * Reference to parent layer that handles neural network layer state handling and initiates primary functions (train, predict, validate etc.)
     *
     */
    protected final AbstractLayer parent;

    /**
     * Activation function for neural network layer.
     *
     */
    protected ActivationFunction activation;

    /**
     * Initialization function for neural network layer.
     *
     */
    protected Init initialization = Init.UNIFORM_XAVIER;

    /**
     * Procedure for the layer. Procedure contains chain of forward and backward expressions.
     *
     */
    private LinkedList<Procedure> procedureList = null;

    /**
     * Current procedure ID.
     *
     */
    private final int currentProcedureID = 0;

    /**
     * Flag if state is reset prior start of next training sequence.
     *
     */
    protected boolean resetStateTraining = true;

    /**
     * Flag if state is reset prior start of next test (validate, predict) sequence.
     *
     */
    protected boolean resetStateTesting = true;

    /**
     * Previous state;
     *
     */
    private transient boolean previousStateTraining;

    /**
     * If true allows layer recurrent input to be reset between test (validate, predict) sequence.
     *
     */
    private boolean allowLayerReset = true;

    /**
     * Limits number of backward propagation sequence steps.
     *
     */
    protected int truncateSteps = -1;

    /**
     * Constructor for abstract execution layer.
     *
     * @param parent reference to parent abstract layer.
     * @param activation activation function.
     * @param initialization initialization function.
     * @param params parameters for neural network layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception setting of activation function fails.
     */
    protected AbstractExecutionLayer(AbstractLayer parent, ActivationFunction activation, Init initialization, String params) throws DynamicParamException, NeuralNetworkException {
        this.parent = parent;

        if (activation != null) this.activation = activation;
        else this.activation = new ActivationFunction(UnaryFunctionType.ELU);

        if (initialization != null) this.initialization = initialization;

        if (params != null) setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for layer.<br>
     * Implemented by actual neural network layer with parameters specific to that layer.<br>
     *
     * @return parameters used for recurrent layer.
     */
    protected abstract HashMap<String, DynamicParam.ParamType> getParamDefs();

    /**
     * Sets parameters used for layer.<br>
     * Implemented by actual neural network layer with parameters specific to that layer.<br>
     *
     * @param params parameters used for recurrent layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected abstract void setParams(DynamicParam params) throws DynamicParamException;

    /**
     * Returns layer type by name.
     *
     * @return layer type by name.
     * @throws NeuralNetworkException throws exception if operation fails.
     */
    public String getTypeByName() throws NeuralNetworkException  {
        return LayerFactory.getLayerTypeByName(this);
    }

    /**
     * Returns used initialization function.
     *
     * @return used initialization function.
     */
    public Init getInitialization() {
        return initialization;
    }

    /**
     * Sets if recurrent inputs of layer are allowed to be reset.
     *
     * @param allowLayerReset if true allows reset.
     */
    public void setAllowLayerReset(boolean allowLayerReset) {
        this.allowLayerReset = allowLayerReset;
    }

    /**
     * Takes single forward processing step process layer input(s).<br>
     * Applies automated forward procedure when relevant to layer.<br>
     * Additionally applies any normalization or regularization defined for the layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    public void forwardProcess() throws MatrixException, NeuralNetworkException {
        if (procedureList == null) defineProcedure();

        parent.getBackward().normalizerReset();

        Sequence outP = parent.getBackward().getPLayer().isConvolutionalLayer() && !isConvolutionalLayer() ? parent.getBackward().getPLayer().getOuts().flatten() : parent.getBackward().getPLayer().getOuts();

        parent.getBackward().regulateForward(outP);
        parent.getBackward().regulateForward();

        parent.getBackward().normalizeForward();

        parent.resetOuts();

        boolean hasDependencies = procedureList.get(currentProcedureID).hasDependencies();

        if (allowLayerReset && hasDependencies && currentProcedureID == 0) {
            if (previousStateTraining != parent.getBackward().isTraining()) procedureList.get(currentProcedureID).resetDependencies();
            else if(resetStateTraining || resetStateTesting) procedureList.get(currentProcedureID).resetDependencies();
        }
        previousStateTraining = parent.getBackward().isTraining();

        procedureList.get(currentProcedureID).calculateExpression(outP, parent.getOuts(),!parent.getBackward().isTraining() && !hasDependencies);

        if(!parent.getBackward().isTraining() && hasDependencies) procedureList.get(currentProcedureID).reset();

        if (parent.getForward() == null) parent.updateOutputError();

        if (!parent.getBackward().isTraining()) parent.getBackward().normalizeFinalizeForward();


    }

    /**
     * Defines layer procedure for forward and backward calculation (automatic gradient) by applying procedure factory.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private void defineProcedure() throws MatrixException {
        ProcedureFactory procedureFactory = new ProcedureFactory();

        procedureFactory.registerMatrix(parent.getBackward().getWs(), true);
        if (parent.getBackward().getNormalization().size() > 0) {
            for (Matrix W : parent.getBackward().getWs()) {
                if (parent.getBackward().getNorm().contains(W)) W.setNormalization(parent.getBackward().getNormalization());
            }
        }

        boolean reset = true;
        while (procedureList == null) {
            resetInput(reset);
            procedureFactory.newProcedure(getInputMatrices());
            procedureList = procedureFactory.endProcedure(getForwardProcedure(parent.getBackward().getNormalization()));
            reset = false;
        }
    }

    /**
     * Resets input.
     *
     * @param resetPreviousInput if true resets previous input.
     * @throws MatrixException throws exception is matrix operation fails.
     */
    protected abstract void resetInput(boolean resetPreviousInput) throws MatrixException;

    /**
     * Returns input matrix for procedure construction.
     *
     * @return input matrix for procedure construction.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract Sample getInputMatrices() throws MatrixException;

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @param normalizers normalizers for layer normalization.
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract Sample getForwardProcedure(HashSet<Normalization> normalizers) throws MatrixException;

    /**
     * Takes single backward processing step to process layer output gradient(s) towards input.<br>
     * Applies automated backward (automatic gradient) procedure when relevant to layer.<br>
     * Additionally applies any regularization defined for the layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backwardProcess() throws MatrixException {
        parent.resetOutGrads();

        parent.getBackward().resetGrad();

        Sequence dEosN = isConvolutionalLayer() && parent.getForward().hasNLayer() && !parent.getForward().getNLayer().isConvolutionalLayer() ? parent.getdEosN().unflatten(parent.getWidth(), parent.getHeight(), parent.getDepth()) : parent.getdEosN();

        procedureList.get(currentProcedureID).calculateGradient(dEosN, parent.getdEos(), truncateSteps);

        for (Integer sampleIndex : parent.getOuts().keySet()) {
            for (Matrix W : parent.getBackward().getdWs().keySet()) {
                parent.getBackward().getdWs(W).put(sampleIndex, procedureList.get(currentProcedureID).getNode(W).getGradient(sampleIndex));
            }
        }

        procedureList.get(currentProcedureID).reset();

        parent.getBackward().sumGrad();

        parent.getBackward().regulateBackward();

        parent.getBackward().normalizeBackward();

    }

}
