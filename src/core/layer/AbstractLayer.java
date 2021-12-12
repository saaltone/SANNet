/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.layer;

import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.sampling.Sequence;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Implements abstract layer that handles state management of neural network layer, thread management and primary functions (train, validate, predict) of neural network layer.<br>
 * Implementation is based on principle that each neural network is individual thread to allow concurrent execution of multiple neural network instances.<br>
 *
 */
public abstract class AbstractLayer implements NeuralNetworkLayer, Runnable, Serializable {

    @Serial
    private static final long serialVersionUID = -3851862716566007887L;

    /**
     * Defines states of neural network layer.<br>
     *   IDLE: neural network layer is idle ready for operation call.<br>
     *   TRAIN: initiates forward training procedure step.<br>
     *   PREDICT: initiates forward predict procedure step.<br>
     *   BACKWARD: initiates backward phase of training procedure step.<br>
     *   UPDATE: initiates weight update procedure step.<br>
     *   EXECUTING: neural network layer is executing procedure step.<br>
     *   TERMINATED: neural network layer is terminated (layer thread is terminated).<br>
     *
     */
    private enum ExecutionState {
        IDLE,
        TRAIN,
        PREDICT,
        BACKWARD,
        UPDATE,
        EXECUTING,
        TERMINATED
    }

    /**
     * Parameter name types for abstract layer.
     *     - width: width of layer. Default 1.<br>
     *     - height: height of layer. Default 1.<br>
     *     - depth: depth of layer. Default 1.<br>
     *
     */
    private final static String paramNameTypes = "(width:INT), " +
            "(height:INT), " +
            "(depth:INT)";

    /**
     * Index of layer.
     *
     */
    private final int layerIndex;

    /**
     * Lock for synchronizing neural network layer thread operations.
     *
     */
    private transient Lock executeLock;

    /**
     * Lock condition for synchronizing execution procedures (train, predict, backward, update).
     *
     */
    private transient Condition executeLockCondition;

    /**
     * Lock condition for synchronizing completion of procedure execution and shift to idle state.
     *
     */
    private transient Condition executeLockCompleteCondition;

    /**
     * Execution state of neural network layer.
     *
     */
    private transient ExecutionState executionState;

    /**
     * Execution thread for neural network layer.
     *
     */
    private transient Thread layerThread;

    /**
     * Reference to next layer
     *
     */
    private NeuralNetworkLayer nextLayer;

    /**
     * Reference to previous layer.
     *
     */
    private NeuralNetworkLayer previousLayer;

    /**
     * Width of neural network layer. Also known as number of neural network layer nodes.
     *
     */
    private int layerWidth = 1;

    /**
     * Height of neural network layer. Relevant for convolutional layers.
     *
     */
    private int layerHeight = 1;

    /**
     * Depth of neural network layer. Relevant for convolutional layers.
     *
     */
    private int layerDepth = 1;

    /**
     * Outputs of neural network layer.
     *
     */
    private transient Sequence layerOutputs;

    /**
     * Gradients of neural network layer.
     *
     */
    private transient Sequence layerGradients;

    /**
     * Default constructor for abstract layer.
     *
     * @param layerIndex layer index
     * @param params parameters
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    AbstractLayer(int layerIndex, String params) throws DynamicParamException, NeuralNetworkException {
        this.layerIndex = layerIndex;
        initializeDefaultParams();
        if (params != null) setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for abstract layer.
     *
     * @return parameters used for abstract layer.
     */
    public String getParamDefs() {
        return AbstractLayer.paramNameTypes;
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        layerWidth = -1;
        layerHeight = 1;
        layerDepth = 1;
    }

    /**
     * Initializes neural network layer dimensions.
     *
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     */
    public void initializeDimensions() throws NeuralNetworkException {
        if (getLayerWidth() == -1) {
            setLayerWidth(getPreviousLayerWidth());
            setLayerHeight(getPreviousLayerHeight());
            setLayerDepth(getPreviousLayerDepth());
        }
    }

    /**
     * Sets parameters used for abstract layer.<br>
     * <br>
     * Supported parameters are:<br>
     *     - width: width of layer. Default 1.<br>
     *     - height: height of layer. Default 1.<br>
     *     - depth: depth of layer. Default 1.<br>
     *
     * @param params parameters used for abstract layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        if (params.hasParam("width")) {
            layerWidth = params.getValueAsInteger("width");
            if (layerWidth < 1) throw new NeuralNetworkException("Width of layer must be at least 1.");
        }
        if (params.hasParam("height")) {
            layerHeight = params.getValueAsInteger("height");
            if (layerHeight < 1) throw new NeuralNetworkException("Height of layer must be at least 1.");
        }
        if (params.hasParam("depth")) {
            layerDepth = params.getValueAsInteger("depth");
            if (layerDepth < 1) throw new NeuralNetworkException("Depth of layer must be at least 1.");
        }
    }

    /**
     * Returns name of layer
     *
     * @return name of layer
     * @throws NeuralNetworkException throws exception if operation fails.
     */
    public String getLayerName() throws NeuralNetworkException {
        return switch (layerIndex) {
            case 0 -> "InputLayer";
            case -1 -> "OutputLayer";
            default -> "Hidden Layer " + layerIndex + " - " + getTypeByName();
        };
    }

    /**
     * Returns layer type by name
     *
     * @return layer type by name
     * @throws NeuralNetworkException throws exception if layer is of an unknown type.
     */
    protected abstract String getTypeByName() throws NeuralNetworkException;

    /**
     * Check if layer is bidirectional.
     *
     * @return true if layer is bidirectional otherwise returns false.
     */
    public boolean isBidirectional() {
        return false;
    }

    /**
     * Sets reference to next neural network layer.
     *
     * @param nextLayer reference to next neural network layer.
     */
    public void setNextLayer(NeuralNetworkLayer nextLayer) {
        this.nextLayer = nextLayer;
    }

    /**
     * Returns reference to next layer.
     *
     * @return reference to next layer.
     */
    public NeuralNetworkLayer getNextLayer() {
        return nextLayer;
    }

    /**
     * Returns if layer has next layer.
     *
     * @return true if layer has next layer otherwise false.
     */
    public boolean hasNextLayer() {
        return nextLayer != null;
    }

    /**
     * Sets reference to previous neural network layer.
     *
     * @param previousLayer reference to previous neural network layer.
     */
    public void setPreviousLayer(NeuralNetworkLayer previousLayer) {
        this.previousLayer = previousLayer;
    }

    /**
     * Returns reference to previous neural network layer.
     *
     * @return reference to previous neural network layer.
     */
    public NeuralNetworkLayer getPreviousLayer() {
        return previousLayer;
    }

    /**
     * Returns if layer has previous layer.
     *
     * @return true if layer has previous layer otherwise false.
     */
    public boolean hasPreviousLayer() {
        return previousLayer != null;
    }

    /**
     * Sets width of the neural network layer.
     *
     * @param layerWidth width of neural network layer.
     */
    protected void setLayerWidth(int layerWidth) {
        this.layerWidth = layerWidth;
    }

    /**
     * Returns width of neural network layer.
     *
     * @return width of neural network layer.
     */
    public int getLayerWidth() {
        return layerWidth;
    }

    /**
     * Sets height of the neural network layer. Relevant for convolutional layers.
     *
     * @param layerHeight height of neural network layer.
     */
    protected void setLayerHeight(int layerHeight) { this.layerHeight = layerHeight; }

    /**
     * Returns height of neural network layer. Relevant for convolutional layers.
     *
     * @return height of neural network layer.
     */
    public int getLayerHeight() {
        return layerHeight;
    }

    /**
     * Sets depth of the neural network layer. Relevant for convolutional layers.
     *
     * @param layerDepth depth of neural network layer.
     */
    protected void setLayerDepth(int layerDepth) { this.layerDepth = layerDepth; }

    /**
     * Returns depth of neural network layer. Relevant for convolutional layers.
     *
     * @return depth of neural network layer.
     */
    public int getLayerDepth() {
        return layerDepth;
    }

    /**
     * Returns width of previous layer.
     *
     * @return width of previous layer.
     */
    public int getPreviousLayerWidth() {
        return getPreviousLayer().isConvolutionalLayer() && !isConvolutionalLayer() ? getPreviousLayer().getLayerWidth() * getPreviousLayer().getLayerHeight() * getPreviousLayer().getLayerDepth() : getPreviousLayer().getLayerWidth();
    }

    /**
     * Returns height of previous layer.
     *
     * @return height of previous layer.
     */
    public int getPreviousLayerHeight() {
        return getPreviousLayer().isConvolutionalLayer() && !isConvolutionalLayer() ? 1 : getPreviousLayer().getLayerHeight();
    }

    /**
     * Returns depth of previous layer.
     *
     * @return depth of previous layer.
     */
    public int getPreviousLayerDepth() {
        return getPreviousLayer().isConvolutionalLayer() && !isConvolutionalLayer() ? 1 : getPreviousLayer().getLayerDepth();
    }

    /**
     * Defines layer procedure for forward and backward calculation (automatic gradient) by applying procedure factory.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     */
    protected abstract void defineProcedure() throws MatrixException, DynamicParamException, NeuralNetworkException;

    /**
     * Returns output of neural network.
     *
     * @return output of neural network.
     */
    public Sequence getOutput() {
        return hasNextLayer() ? getNextLayer().getOutput() : getLayerOutputs();
    }

    /**
     * Returns outputs of neural network layer.
     *
     * @return outputs of neural network layer.
     */
    public Sequence getLayerOutputs() {
        return layerOutputs;
    }

    /**
     * Sets layer outputs.
     *
     * @param layerOutputs layer outputs.
     */
    protected void setLayerOutputs(Sequence layerOutputs) {
        this.layerOutputs = layerOutputs;
    }

    /**
     * Returns previous layer outputs.
     *
     * @return previous layer outputs.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Sequence getPreviousLayerOutputs() throws MatrixException {
        return hasPreviousLayer() ? getPreviousLayer().isConvolutionalLayer() && !isConvolutionalLayer() ? getPreviousLayer().getLayerOutputs().flatten() : getPreviousLayer().getLayerOutputs() : getLayerOutputs();
    }

    /**
     * Resets outputs of neural network layer.
     *
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     */
    public void resetLayerOutputs() throws MatrixException {
        layerOutputs = new Sequence(getLayerDepth());
    }

    /**
     * Returns neural network layer gradients.
     *
     * @return neural network layer gradients.
     */
    public Sequence getLayerGradients() {
        return layerGradients;
    }

    /**
     * Sets layer gradients.
     *
     * @param layerGradients layer gradients.
     */
    protected void setLayerGradients(Sequence layerGradients) {
        this.layerGradients = layerGradients;
    }

    /**
     * Returns gradients of next neural network layer.
     *
     * @return gradients of next neural network layer
     */
    public Sequence getNextLayerGradients() {
        return getNextLayer() != null ? getNextLayer().getLayerGradients() : getLayerGradients();
    }

    /**
     * Resets gradients of neural network layer.
     *
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     */
    public void resetLayerGradients() throws MatrixException {
        layerGradients = new Sequence(getLayerDepth());
    }

    /**
     * Starts neural network layer and it's execution thread.
     *
     * @throws NeuralNetworkException throws exception if neural network layer name cannot be returned.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void start() throws NeuralNetworkException, MatrixException, DynamicParamException {
        if (layerThread != null) return;
        executeLock = new ReentrantLock();
        executeLockCondition = executeLock.newCondition();
        executeLockCompleteCondition = executeLock.newCondition();
        executionState = ExecutionState.IDLE;

        layerThread = new Thread(this);
        layerThread.setName(getLayerName());
        layerThread.start();

        if (hasNextLayer()) getNextLayer().start();

        defineProcedure();
    }

    /**
     * Stops neural network layer and terminates neural network layer execution thread.<br>
     * Sets layer state to TERMINATED.<br>
     *
     */
    public void stop() {
        executeLock.lock();
        executionState = ExecutionState.TERMINATED;
        executeLockCondition.signal();
        if (hasNextLayer()) getNextLayer().stop();
        executeLock.unlock();
    }

    /**
     * Executes training step for neural network layer and propagates procedure to next layer.
     *
     * @param inputs training inputs for layer.
     */
    public void train(Sequence inputs) {
        if (inputs.isEmpty()) return;
        layerOutputs = new Sequence(inputs);
        nextState(ExecutionState.TRAIN);
        waitToComplete();
    }

    /**
     * Executes training step for neural network layer and propagates procedure to next layer.<br>
     * Uses existing training inputs and outputs.<br>
     *
     */
    public void train() {
        nextState(ExecutionState.TRAIN);
    }

    /**
     * Executes predict step for neural network layer and propagates procedure to next layer.
     *
     * @param inputs predict inputs for layer.
     * @return output of next layer or this layer if next layer does not exist.
     */
    public Sequence predict(Sequence inputs) {
        if (inputs.isEmpty()) return null;
        layerOutputs = new Sequence(inputs);
        nextState(ExecutionState.PREDICT);
        waitToComplete();
        return getOutput();
    }

    /**
     * Executes predict step for neural network layer and propagates procedure to next layer.<br>
     * Uses existing testing inputs.<br>
     *
     */
    public void predict() {
        nextState(ExecutionState.PREDICT);
    }

    /**
     * Executes backward (gradient) propagation phase for training step of neural network layer.
     *
     * @throws NeuralNetworkException throws exception if backward operation fails.
     */
    public void backward() throws NeuralNetworkException {
        nextState(ExecutionState.BACKWARD);
        if (!hasNextLayer()) waitToComplete();
    }

    /**
     * Executes parameter (weight) update for training step of neural network layer.
     *
     */
    public void update() {
        nextState(ExecutionState.UPDATE);
    }

    /**
     * Sets next execution state.
     * @param executionState next execution state.
     *
     */
    private void nextState(ExecutionState executionState) {
        executeLock.lock();
        this.executionState = executionState;
        executeLockCondition.signal();
        executeLock.unlock();
    }

    /**
     * Marks state completed and propagates information to forward or backward direction depending on given flag.
     *
     * @param forwardDirection if true propagates state completion signal to forward direction otherwise propagates to backward direction.
     */
    public void stateCompleted(boolean forwardDirection) {
        if (forwardDirection) {
            if (hasNextLayer()) getNextLayer().stateCompleted(true);
            else stateCompleted();
        }
        else {
            if (hasPreviousLayer()) getPreviousLayer().stateCompleted(false);
            else stateCompleted();
        }
    }

    /**
     * Changes layer execution state to IDLE.
     *
     */
    private void stateCompleted() {
        executeLock.lock();
        executionState = ExecutionState.IDLE;
        executeLockCompleteCondition.signal();
        executeLock.unlock();
    }

    /**
     * Waits that layer execution step is completed.
     *
     */
    void waitToComplete() {
        executeLock.lock();
        if (executionState != ExecutionState.IDLE) executeLockCompleteCondition.awaitUninterruptibly();
        executeLock.unlock();
    }

    /**
     * Thread run function.<br>
     * Executes given neural network procedures and synchronizes their execution via layer thread execution lock.<br>
     *
     */
    public void run() {
        while (true) {
            executeLock.lock();
            if (executionState == ExecutionState.IDLE || executionState == ExecutionState.EXECUTING) executeLockCondition.awaitUninterruptibly();
            try {
                switch (executionState) {
                    case TRAIN:
                        setTraining(true);
                        forwardProcess();
                        if (hasNextLayer()) getNextLayer().train();
                        else if (hasPreviousLayer()) getPreviousLayer().stateCompleted(false);
                        break;
                    case PREDICT:
                        setTraining(false);
                        forwardProcess();
                        if (hasNextLayer()) getNextLayer().predict();
                        else if (hasPreviousLayer()) getPreviousLayer().stateCompleted(false);
                        break;
                    case BACKWARD:
                        if (hasPreviousLayer()) {
                            backwardProcess();
                            getPreviousLayer().backward();
                        }
                        else if (hasNextLayer()) getNextLayer().stateCompleted(true);
                        break;
                    case UPDATE:
                        optimize();
                        if (hasNextLayer()) getNextLayer().update();
                        else if (hasPreviousLayer()) getPreviousLayer().stateCompleted(false);
                        break;
                    case TERMINATED:
                        if (hasNextLayer()) getNextLayer().stop();
                        layerThread = null;
                        executionState = ExecutionState.IDLE;
                        executeLock.unlock();
                        return;
                }
            }
            catch (Exception exception) {
                exception.printStackTrace();
                System.exit(-1);
            }
            executionState = ExecutionState.EXECUTING;
            executeLock.unlock();
        }
    }


    /**
     * Sets training flag.
     *
     * @param training if true layer is training otherwise false.
     */
    protected abstract void setTraining(boolean training);

    /**
     * Executes optimization step.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected abstract void optimize() throws MatrixException, DynamicParamException;

}
