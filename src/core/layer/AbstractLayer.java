/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.layer;

import core.NeuralNetworkException;
import utils.DynamicParamException;
import utils.Sequence;
import utils.matrix.Init;
import utils.matrix.MatrixException;

import java.io.Serializable;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Implements abstract layer that handles state management of neural network layer, thread management and primary functions (train, validate, predict) of neural network layer.<br>
 * Implementation is based on principle that each neural network is individual thread to allow concurrent execution of multiple neural network instances.<br>
 *
 */
public abstract class AbstractLayer implements Runnable, Serializable {

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
     * Index of layer.
     *
     */
    private final int layerIndex;

    /**
     * Lock for synchronizing neural network layer thread operations.
     *
     */
    private transient Lock lock;

    /**
     * Lock condition for synchronizing execution procedures (train, predict, backward, update).
     *
     */
    private transient Condition execute;

    /**
     * Lock condition for synchronizing completion of procedure execution and shift to idle state.
     *
     */
    private transient Condition complete;

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
     * Reference to execution layer implementing actual neural network layer functions.
     *
     */
    private Layer executionLayer;

    /**
     * Connector forward to next layer.
     *
     */
    private Connector forward;

    /**
     * Connector backward to previous layer.
     *
     */
    private Connector backward;

    /**
     * Width of neural network layer. Also known as number of neural network layer nodes.
     *
     */
    private int width;

    /**
     * Height of neural network layer. Relevant for convolutional layers.
     *
     */
    private int height = 1;

    /**
     * Depth of neural network layer. Relevant for convolutional layers.
     *
     */
    private int depth = 1;

    /**
     * Tree map for storing outputs of neural network layer.
     *
     */
    private transient Sequence outs;

    /**
     * Gradients for neural network layer.
     *
     */
    private transient Sequence dEos;

    /**
     * Static function to create connector between two layers.
     *
     * @param previousLayer previous layer of connector.
     * @param nextLayer next layer of connector
     * @return created connector between previous and next layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public static Connector Connect(AbstractLayer previousLayer, AbstractLayer nextLayer) throws DynamicParamException {
        Connector connector = new Connector(previousLayer, nextLayer);
        previousLayer.setForward(connector);
        nextLayer.setBackward(connector);
        return connector;
    }

    /**
     * Default constructor for abstract layer.
     *
     */
    AbstractLayer(int layerIndex) {
        this.layerIndex = layerIndex;
    }

    /**
     * Returns name of layer
     *
     * @return name of layer
     * @throws NeuralNetworkException throws exception if operation fails.
     */
    private String getLayerName() throws NeuralNetworkException {
        String layerName;
        switch (layerIndex) {
            case 0:
                layerName = "InputLayer";
                break;
            case -1:
                layerName = "OutputLayer - ";
                break;
            default:
                layerName = "HiddenLayer " + layerIndex + " - ";
                break;
        }
        if (executionLayer != null && layerIndex != 0) layerName += executionLayer.getTypeByName();
        return layerName;
    }

    /**
     * Sets execution layer for layer.
     *
     * @param executionLayer execution layer reference to be set.
     */
    void setExecutionLayer(Layer executionLayer) {
        this.executionLayer = executionLayer;
    }

    /**
     * Sets forward connector with link to next neural network layer.
     *
     * @param forward reference to forward connector.
     */
    private void setForward(Connector forward) {
        this.forward = forward;
    }

    /**
     * Returns forward connector to next layer.
     *
     * @return forward connector.
     */
    public Connector getForward() {
        return forward;
    }

    /**
     * Sets backward connector with link to previous neural network layer.
     *
     * @param backward reference to backward connector.
     */
    private void setBackward(Connector backward) {
        this.backward = backward;
    }

    /**
     * Returns backward connector to previous layer.
     *
     * @return backward connector.
     */
    public Connector getBackward() {
        return backward;
    }

    /**
     * Sets width of the neural network layer.
     *
     * @param width width of neural network layer.
     */
    public void setWidth(int width) { this.width = width; }

    /**
     * Returns width of neural network layer.
     *
     * @return width of neural network layer.
     */
    public int getWidth() {
        return width;
    }

    /**
     * Sets height of the neural network layer. Relevant for convolutional layers.
     *
     * @param height height of neural network layer.
     */
    public void setHeight(int height) { this.height = height; }

    /**
     * Returns height of neural network layer.
     *
     * @return height of neural network layer.
     */
    public int getHeight() {
        return height;
    }

    /**
     * Sets depth of the neural network layer. Relevant for convolutional layers.
     *
     * @param depth depth of neural network layer.
     */
    public void setDepth(int depth) { this.depth = depth; }

    /**
     * Returns depth of neural network layer. Relevant for convolutional layers.
     *
     * @return depth of neural network layer.
     */
    public int getDepth() {
        return depth;
    }

    /**
     * Get used initialization function.
     *
     * @return used initialization function. Relevant for convolutional layers.
     */
    public Init getInitialization() {
        return executionLayer.getInitialization();
    }

    /**
     * Checks if execution layer is recurrent layer type.
     *
     * @return true if execution layer is recurrent layer type otherwise false.
     */
    public boolean isRecurrentLayer() {
        return executionLayer != null && executionLayer.isRecurrentLayer();
    }

    /**
     * Checks if execution layer is convolutional layer type.
     *
     * @return true if execution layer is convolutional layer type otherwise false.
     */
    public boolean isConvolutionalLayer() {
        return executionLayer != null && executionLayer.isConvolutionalLayer();
    }

    /**
     * Returns outputs of neural network layer.
     *
     * @return outputs of neural network layer.
     */
    public Sequence getOuts() {
        return outs;
    }

    /**
     * Sets if recurrent inputs of layer are allowed to be reset.
     *
     * @param allowLayerReset if true allows reset of recurrent inputs.
     */
    public void setAllowLayerReset(boolean allowLayerReset) {
        if (executionLayer != null) executionLayer.setAllowLayerReset(allowLayerReset);
        if (forward != null) forward.setAllowLayerReset(allowLayerReset);
    }

    /**
     * Resets outputs of neural network layer.
     *
     */
    public void resetOuts() {
        outs = new Sequence(depth);
    }

    /**
     * Returns neural network layer gradients.
     *
     * @return neural network layer gradients.
     */
    public Sequence getdEos() {
        return dEos;
    }

    /**
     * Resets gradients of neural network layer.
     *
     */
    public void resetOutGrads() {
        dEos = new Sequence(depth);
    }

    /**
     * Initializes execution layer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if intialization fails.
     */
    public void initialize() throws MatrixException, NeuralNetworkException {
        if (executionLayer != null) executionLayer.initialize();
    }

    /**
     * Starts neural network layer and it's execution thread.
     *
     * @throws NeuralNetworkException throws exception if neural network layer name cannot be returned.
     */
    public void start() throws NeuralNetworkException {
        if (layerThread != null) return;
        lock = new ReentrantLock();
        execute = lock.newCondition();
        complete = lock.newCondition();
        executionState = ExecutionState.IDLE;

        layerThread = new Thread(this);
        layerThread.setName(getLayerName());
        layerThread.start();

        outs = new Sequence(depth);
        resetOutGrads();

        if (forward != null) forward.start();
    }

    /**
     * Executes training step for neural network layer and propagates procedure to next layer.
     *
     * @param inputs training inputs for the layer.
     */
    public void train(Sequence inputs) {
        if (inputs.isEmpty()) return;
        nextState(ExecutionState.TRAIN, false);
        outs = new Sequence(inputs);
        lock.unlock();
        waitToComplete();
    }

    /**
     * Executes training step for neural network layer and propagates procedure to next layer.<br>
     * Uses existing training inputs and outputs.<br>
     *
     */
    public void train() {
        nextState(ExecutionState.TRAIN, true);
    }

    /**
     * Executes predict step for neural network layer and propagates procedure to next layer.
     *
     * @param inputs predict inputs for the layer.
     * @return output of next layer or this layer if next layer does not exist.
     */
    public Sequence predict(Sequence inputs) {
        if (inputs.isEmpty()) return null;
        nextState(ExecutionState.PREDICT, false);
        outs = new Sequence(inputs);
        lock.unlock();
        waitToComplete();
        return getOutput();
    }

    /**
     * Executes predict step for neural network layer and propagates procedure to next layer.<br>
     * Uses existing training inputs and outputs.<br>
     *
     */
    public void predict() {
        nextState(ExecutionState.PREDICT, true);
    }

    /**
     * Executes backward (gradient) propagation phase for training step of neural network layer.
     *
     * @throws NeuralNetworkException throws exception if backward operation fails.
     */
    public void backward() throws NeuralNetworkException {
        nextState(ExecutionState.BACKWARD, true);
        if (forward == null) waitToComplete();
    }

    /**
     * Executes parameter (weight) update for training step of neural network layer.
     *
     */
    public void update(){
        nextState(ExecutionState.UPDATE, true);
    }

    /**
     * Sets next execution state.
     *
     * @param executionState next execution state.
     * @param unlock if true unlocks neural network thread lock otherwise keeps lock locked.
     */
    private void nextState(ExecutionState executionState, boolean unlock) {
        lock.lock();
        this.executionState = executionState;
        execute.signal();
        if (unlock) lock.unlock();
    }

    /**
     * Stops neural network layer and terminates neural network layer execution thread.<br>
     * Sets layer state to TERMINATED.<br>
     *
     */
    public void stop(){
        lock.lock();
        executionState = ExecutionState.TERMINATED;
        execute.signal();
        if (forward != null) forward.stop();
        lock.unlock();
    }

    /**
     * Marks state completed and propagates information to forward or backward direction depending on given flag.
     *
     * @param forwardDirection if true propagates state completion signal to forward direction otherwise propagates to backward direction.
     */
    public void stateCompleted(boolean forwardDirection) {
        if (forwardDirection) {
            if (forward != null) forward.stateCompleted(true);
            else stateCompleted();
        }
        else {
            if (backward != null) backward.stateCompleted(false);
            else stateCompleted();
        }
    }

    /**
     * Changes layer execution state to IDLE.
     *
     */
    private void stateCompleted() {
        lock.lock();
        executionState = ExecutionState.IDLE;
        complete.signal();
        lock.unlock();
    }

    /**
     * Waits that layer execution step is completed.
     *
     */
    void waitToComplete() {
        lock.lock();
        try {
            while (executionState != ExecutionState.IDLE) complete.await();
        }
        catch (InterruptedException exception) {}
        lock.unlock();
    }

    /**
     * Returns output of next layer or this layer if next layer does not exist.<br>
     * Effectively output of neural network.<br>
     *
     * @return output of neural network.
     */
    public Sequence getOutput() {
        if (forward != null) return forward.getOutput();
        else return outs;
    }

    /**
     * Thread run function.<br>
     * Executes given neural network procedures and synchronizes their execution via layer thread execution lock.<br>
     *
     */
    public void run() {
        while (true) {
            lock.lock();
            try {
                while (executionState == ExecutionState.IDLE || executionState == ExecutionState.EXECUTING) execute.await();
            }
            catch (InterruptedException exception) {}
            try {
                switch (executionState) {
                    case TRAIN:
                        if (backward != null) forwardProcess();
                        if (forward != null) forward.train();
                        else if (backward != null) backward.stateCompleted(false);
                        break;
                    case PREDICT:
                        if (backward != null) forwardProcess();
                        if (forward != null) forward.predict();
                        else if (backward != null) backward.stateCompleted(false);
                        break;
                    case BACKWARD:
                        if (backward != null) {
                            backwardProcess();
                            backward.backward();
                        }
                        else if (forward != null) forward.stateCompleted(true);
                        break;
                    case UPDATE:
                        if (forward != null) forward.update();
                        else if (backward != null) backward.stateCompleted(false);
                        break;
                    case TERMINATED:
                        if (forward != null) forward.stop();
                        layerThread = null;
                        executionState = ExecutionState.IDLE;
                        lock.unlock();
                        return;
                }
            }
            catch (Exception exception) {
                exception.printStackTrace();
                System.exit(-1);
            }
            executionState = ExecutionState.EXECUTING;
            lock.unlock();
        }
    }

    /**
     * Executes forward processing step of execution layer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    private void forwardProcess() throws MatrixException, NeuralNetworkException {
        executionLayer.forwardProcess();
    }

    /**
     * Executes backward processing step of execution layer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void backwardProcess() throws MatrixException {
        executionLayer.backwardProcess();
    }

    /**
     * Updates output error of a layer.<br>
     * Implemented by actual neural network layer (input, hidden or output layer).<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    public abstract void updateOutputError() throws MatrixException, NeuralNetworkException;

    /**
     * Returns gradients of next neural network layer.<br>
     * Implemented by actual neural network layer (input, hidden or output layer).<br>
     *
     * @return gradients of next neural network layer
     */
    public abstract Sequence getdEosN();

}
