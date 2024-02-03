/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.layer;

import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;
import utils.sampling.Sequence;

import java.io.Serial;
import java.io.Serializable;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.*;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Implements abstract layer that handles state management of neural network layer, thread management and primary functions (train, validate, predict) of neural network layer.<br>
 * Implementation is based on principle that each neural network is individual thread to allow concurrent execution of multiple neural network instances.<br>
 *
 */
public abstract class AbstractLayer implements NeuralNetworkLayer, Serializable {

    @Serial
    private static final long serialVersionUID = -3851862716566007887L;

    /**
     * Defines states of neural network layer.<br>
     *   IDLE: neural network layer is idle ready for operation call.<br>
     *   TRAIN: initiates forward training procedure step.<br>
     *   PREDICT: initiates forward predict procedure step.<br>
     *   BACKWARD: initiates backward phase of training procedure step.<br>
     *   UPDATE: initiates weight update procedure step.<br>
     *   TERMINATED: neural network layer is terminated (layer thread is terminated).<br>
     *
     */
    private enum ExecutionState {
        IDLE,
        TRAIN,
        PREDICT,
        BACKWARD,
        UPDATE,
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
     * Lock-condition for synchronizing execution procedures (train, predict, backward, update).
     *
     */
    private transient Condition executeLockCondition;

    /**
     * Lock for synchronizing neural network layer complete operations.
     *
     */
    private transient Lock completeLock;

    /**
     * Lock-condition for synchronizing procedure completion.
     *
     */
    private transient Condition completeLockCondition;

    /**
     * Execution state of neural network layer.
     *
     */
    private transient ExecutionState executionState;

    /**
     * Reference to next layer
     *
     */
    private final TreeMap<Integer, NeuralNetworkLayer> nextLayers = new TreeMap<>();

    /**
     * Reference to previous layer.
     *
     */
    private final TreeMap<Integer, NeuralNetworkLayer> previousLayers = new TreeMap<>();

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
    private final Sequence layerOutputs = new Sequence();

    /**
     * Gradients of neural network layer.
     *
     */
    private final Sequence layerOutputGradients = new Sequence();

    /**
     * Input sequences.
     *
     */
    private final TreeMap<Integer, Sequence> inputSequences = new TreeMap<>();

    /**
     * Input gradient sequences.
     *
     */
    private final TreeMap<Integer, Sequence> inputGradientSequences = new TreeMap<>();

    /**
     * Count for execution start requests from peer layers.
     *
     */
    private transient int executionStartCount;

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
     * Returns index of a layer.
     *
     * @return index of a layer.
     */
    public int getLayerIndex() {
        return layerIndex;
    }

    /**
     * Checks if layer can have multiple previous layers.
     *
     * @return  if true layer can have multiple previous layers otherwise false.
     */
    public boolean canHaveMultiplePreviousLayers() {
        return false;
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
        layerHeight = -1;
        layerDepth = -1;
    }

    /**
     * Initializes neural network layer dimensions.
     *
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     */
    public void initializeDimensions() throws NeuralNetworkException {
        if (getLayerWidth() == -1) {
            if (getDefaultPreviousLayer().getLayerWidth() < 1) throw new NeuralNetworkException("Default previous layer width must be positive. Invalid value: " + getDefaultPreviousLayer().getLayerWidth());
            setLayerWidth(getDefaultPreviousLayer().getLayerWidth());
        }
        if (getLayerHeight() == -1) {
            if (getDefaultPreviousLayer().getLayerHeight() < 1) throw new NeuralNetworkException("Default previous height width must be positive. Invalid value: " + getDefaultPreviousLayer().getLayerHeight());
            setLayerHeight(getDefaultPreviousLayer().getLayerHeight());
        }
        if (getLayerDepth() == -1) {
            if (getDefaultPreviousLayer().getLayerDepth() < 1) throw new NeuralNetworkException("Default previous depth width must be positive. Invalid value: " + getDefaultPreviousLayer().getLayerDepth());
            setLayerDepth(getDefaultPreviousLayer().getLayerDepth());
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
        if (this.getClass().equals(InputLayer.class)) return "InputLayer " + getLayerIndex();
        if (this.getClass().equals(OutputLayer.class)) return "OutputLayer " + getLayerIndex();
        return "Hidden Layer " + getLayerIndex() + " - " + getTypeByName();
    }

    /**
     * Returns layer type by name
     *
     * @return layer type by name
     * @throws NeuralNetworkException throws exception if layer is of an unknown type.
     */
    protected abstract String getTypeByName() throws NeuralNetworkException;

    /**
     * Adds reference to next neural network layer.
     *
     * @param nextLayer reference to next neural network layer.
     * @throws NeuralNetworkException throws exception if next layer is attempted to be added to output layer.
     */
    public void addNextLayer(NeuralNetworkLayer nextLayer) throws NeuralNetworkException {
        int nextLayerIndex = nextLayers.size();
        nextLayers.put(nextLayerIndex, nextLayer);
    }

    /**
     * Returns references to next layers.
     *
     * @return references to next layers.
     */
    public TreeMap<Integer, NeuralNetworkLayer> getNextLayers() {
        return nextLayers;
    }

    /**
     * Returns if layer has next layers.
     *
     * @return true if layer has next layers otherwise false.
     */
    public boolean hasNextLayers() {
        return !nextLayers.isEmpty();
    }

    /**
     * Removes next neural network layer
     *
     * @param neuralNetworkLayer neural network layer.
     * @throws NeuralNetworkException throws exception if next neural network layer is not found.
     */
    public void removeNextLayer(NeuralNetworkLayer neuralNetworkLayer) throws NeuralNetworkException {
        int nextLayerIndex = getLayerIndex(neuralNetworkLayer, nextLayers);
        nextLayers.remove(nextLayerIndex);
    }

    /**
     * Adds reference to previous neural network layer.
     *
     * @param previousLayer reference to previous neural network layer.
     * @throws NeuralNetworkException throws exception if previous layer is attempted to be added to input layer or layer cannot have multiple previous layers.
     */
    public void addPreviousLayer(NeuralNetworkLayer previousLayer) throws NeuralNetworkException {
        int previousLayerIndex = previousLayers.size();
        if (!canHaveMultiplePreviousLayers() && previousLayerIndex > 1) throw new NeuralNetworkException("Layer cannot have multiple previous layers.");
        previousLayers.put(previousLayerIndex, previousLayer);
        inputSequences.put(previousLayerIndex, previousLayer.getLayerOutputs());
        inputGradientSequences.put(previousLayerIndex, previousLayer.getLayerOutputGradients());
    }

    /**
     * Returns references to previous neural network layers.
     *
     * @return references to previous neural network layers.
     */
    public TreeMap<Integer, NeuralNetworkLayer> getPreviousLayers() {
        return previousLayers;
    }

    /**
     * Returns default previous layer.
     *
     * @return default previous layer.
     */
    protected NeuralNetworkLayer getDefaultPreviousLayer() {
        return getPreviousLayers().get(getPreviousLayers().firstKey());
    }

    /**
     * Returns if layer has previous layers.
     *
     * @return true if layer has previous layers otherwise false.
     */
    public boolean hasPreviousLayers() {
        return !previousLayers.isEmpty();
    }

    /**
     * Removes previous neural network layer
     *
     * @param neuralNetworkLayer neural network layer.
     * @throws NeuralNetworkException throws exception if previous neural network layer is not found.
     */
    public void removePreviousLayer(NeuralNetworkLayer neuralNetworkLayer) throws NeuralNetworkException {
        int previousLayerIndex = getLayerIndex(neuralNetworkLayer, previousLayers);
        previousLayers.remove(previousLayerIndex);
        inputSequences.remove(previousLayerIndex);
        inputGradientSequences.remove(previousLayerIndex);
    }

    /**
     * Return index of neural network layer.
     *
     * @param neuralNetworkLayer neural network layer.
     * @param layers layers.
     * @return index of neural network layer.
     * @throws NeuralNetworkException throws exception if neural network layer is not found.
     */
    private int getLayerIndex(NeuralNetworkLayer neuralNetworkLayer, TreeMap<Integer, NeuralNetworkLayer> layers) throws NeuralNetworkException {
        for (Map.Entry<Integer, NeuralNetworkLayer> entry : layers.entrySet()) {
            if (entry.getValue() == neuralNetworkLayer) return entry.getKey();
        }
        throw new NeuralNetworkException("No layer found.");
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
     * Defines layer procedure for forward and backward calculation (automatic gradient) by applying procedure factory.<br>
     *
     * @throws MatrixException        throws exception if matrix operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     */
    protected abstract void defineProcedure() throws MatrixException, DynamicParamException, NeuralNetworkException;

    /**
     * Resets layer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void reset() throws MatrixException {
        if (getLayerOutputs() != null) getLayerOutputs().reset();
        if (getLayerOutputGradients() != null) getLayerOutputGradients().reset();
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
     * @param newLayerOutputs layer outputs.
     */
    protected void setLayerOutputs(Sequence newLayerOutputs) {
        layerOutputs.reset();
        layerOutputs.putAll(newLayerOutputs);
    }

    /**
     * Passes inputs from previous layer to this layer.
     *
     * @throws MatrixException throws exception if depth of sequence is not matching depth of this sequence.
     */
    protected void passLayerOutputs() throws MatrixException {
        this.reset();
        setLayerOutputs(getDefaultLayerInput());
    }

    /**
     * Returns neural network layer input gradients.
     *
     * @return neural network layer input gradients.
     */
    public Sequence getLayerOutputGradients() {
        return layerOutputGradients;
    }

    /**
     * Passes output gradients from this layer to previous layer.
     *
     * @throws MatrixException throws exception if depth of sequence is not matching depth of this sequence.
     */
    protected void passLayerOutputGradients() throws MatrixException {
        getDefaultLayerInputGradient().increment(getLayerOutputGradients());
    }

    /**
     * Returns default layer input.
     *
     * @return default layer input.
     */
    protected Sequence getDefaultLayerInput() {
        return getInputSequences().get(0);
    }

    /**
     * Returns input sequences.
     *
     * @return input sequences.
     */
    protected TreeMap<Integer, Sequence> getInputSequences() {
        return inputSequences;
    }

    /**
     * Returns default layer input gradient.
     *
     * @return default layer input gradient.
     */
    protected Sequence getDefaultLayerInputGradient() {
        return getInputGradientSequences().get(0);
    }

    /**
     * Returns input gradient sequences.
     *
     * @return input gradient sequences.
     */
    protected TreeMap<Integer, Sequence> getInputGradientSequences() {
        return inputGradientSequences;
    }

    /**
     * Starts neural network layer and it's execution thread.
     *
     * @param executorService executor service.
     * @throws NeuralNetworkException throws exception if neural network layer name cannot be returned.
     * @throws MatrixException        throws exception if depth of matrix is less than 1.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     */
    public void start(ExecutorService executorService) throws NeuralNetworkException, MatrixException, DynamicParamException {
        executeLock = new ReentrantLock();
        executeLockCondition = executeLock.newCondition();
        executionState = ExecutionState.IDLE;
        executionStartCount = -1;

        completeLock = new ReentrantLock();
        completeLockCondition = completeLock.newCondition();

        executeLayer(executorService);

        defineProcedure();
    }

    /**
     * Stops neural network layer and terminates neural network layer execution thread.<br>
     * Sets layer state to TERMINATED.<br>
     *
     */
    public void stop() {
        nextState(ExecutionState.TERMINATED, true);
    }

    /**
     * Executes training step for neural network layer and propagates procedure to next layer.
     *
     * @param inputs training inputs for layer.
     */
    public void train(Sequence inputs) {
        if (inputs.isEmpty()) return;
        setLayerOutputs(inputs);
        train(true);
    }

    /**
     * Executes training step for neural network layer and propagates procedure to next layer.<br>
     * Uses existing training inputs and outputs.<br>
     *
     * @param waitToComplete if true wait for layer execution to complete otherwise not.
     */
    public void train(boolean waitToComplete) {
        setTraining(true);
        nextState(ExecutionState.TRAIN, waitToComplete);
    }

    /**
     * Executes predict step for neural network layer and propagates procedure to next layer.
     *
     * @param inputs predict inputs for layer.
     */
    public void predict(Sequence inputs) {
        if (inputs.isEmpty()) return;
        setLayerOutputs(inputs);
        predict(true);
    }

    /**
     * Executes predict step for neural network layer and propagates procedure to next layer.<br>
     * Uses existing testing inputs.<br>
     *
     * @param waitToComplete if true wait for layer execution to complete otherwise not.
     */
    public void predict(boolean waitToComplete) {
        setTraining(false);
        nextState(ExecutionState.PREDICT, waitToComplete);
    }

    /**
     * Executes backward (gradient) propagation phase for training step of neural network layer.
     *
     * @param waitToComplete if true wait for layer execution to complete otherwise not.
     */
    public void backward(boolean waitToComplete) {
        nextState(ExecutionState.BACKWARD, waitToComplete);
    }

    /**
     * Executes parameter (weight) update for training step of neural network layer.
     *
     */
    public void update() {
        nextState(ExecutionState.UPDATE, true);
    }

    /**
     * Executes parameter (weight) update for training step of neural network layer.
     *
     * @param waitToComplete if true wait for layer execution to complete otherwise not.
     */
    public void update(boolean waitToComplete) {
        nextState(ExecutionState.UPDATE, waitToComplete);
    }

    /**
     * Sets next execution state.
     *
     * @param executionState next execution state.
     * @param waitToComplete if true wait for layer execution to complete otherwise not.
     */
    private void nextState(ExecutionState executionState, boolean waitToComplete) {
        waitToComplete();
        try {
            executeLock.lock();
            if (executionStartCount == -1) {
                switch (executionState) {
                    case TRAIN, PREDICT, UPDATE, TERMINATED -> executionStartCount = hasPreviousLayers() ? previousLayers.size() : 0;
                    case BACKWARD -> executionStartCount = hasNextLayers() ? nextLayers.size() : 0;
                    default -> {}
                }
            }
            if (--executionStartCount <= 0) {
                this.executionState = executionState;
                executeLockCondition.signalAll();
                executionStartCount = -1;
            }
        }
        finally {
            executeLock.unlock();
        }

        if (waitToComplete) waitToComplete();
    }

    /**
     * Wait for layer to complete.
     *
     */
    public void waitToComplete() {
        try {
            completeLock.lock();
            while (executionState != ExecutionState.IDLE) completeLockCondition.await();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        } finally {
            completeLock.unlock();
        }
    }

    /**
     * Sets layer completed and notifies other layers.
     *
     */
    private void complete() {
        try {
            completeLock.lock();
            this.executionState = ExecutionState.IDLE;
            completeLockCondition.signalAll();
        }
        finally {
            completeLock.unlock();
        }
    }

    /**
     * Executes layer.
     *
     * @param executorService executor service
     * @throws RuntimeException throws runtime exception in case any exception happens.
     */
    private void executeLayer(ExecutorService executorService) throws RuntimeException {
        executorService.execute(() -> {
            try {
                boolean isExecuting = true;
                while (isExecuting) {
                    isExecuting = executeLayerOperation();
                }
            } catch (Exception exception) {
                throw new RuntimeException(exception);
            }
        });
    }

    /**
     * Executes layer operation.
     *
     * @return return false if layer has been terminated otherwise returns true.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException  throws exception if parameter (params) setting fails.
     */
    private boolean executeLayerOperation() throws MatrixException, DynamicParamException {
        try {
            executeLock.lock();
            switch (executionState) {
                case TRAIN -> {
                    forwardProcess();
                    if (hasNextLayers()) {
                        for (NeuralNetworkLayer nextLayer : getNextLayers().values()) nextLayer.train(false);
                        for (NeuralNetworkLayer nextLayer : getNextLayers().values()) nextLayer.waitToComplete();
                    }
                    complete();
                }
                case PREDICT -> {
                    forwardProcess();
                    if (hasNextLayers()) {
                        for (NeuralNetworkLayer nextLayer : getNextLayers().values()) nextLayer.predict(false);
                        for (NeuralNetworkLayer nextLayer : getNextLayers().values()) nextLayer.waitToComplete();
                    }
                    complete();
                }
                case BACKWARD -> {
                    backwardProcess();
                    if (hasPreviousLayers()) {
                        for (NeuralNetworkLayer previousLayer : getPreviousLayers().values()) previousLayer.backward(false);
                        for (NeuralNetworkLayer previousLayer : getPreviousLayers().values()) previousLayer.waitToComplete();
                    }
                    complete();
                }
                case UPDATE -> {
                    optimize();
                    if (hasNextLayers()) {
                        for (NeuralNetworkLayer nextLayer : getNextLayers().values()) nextLayer.update(false);
                        for (NeuralNetworkLayer nextLayer : getNextLayers().values()) nextLayer.waitToComplete();
                    }
                    complete();
                }
                case TERMINATED -> {
                    if (hasNextLayers()) {
                        for (NeuralNetworkLayer nextLayer : getNextLayers().values()) nextLayer.stop();
                        for (NeuralNetworkLayer nextLayer : getNextLayers().values()) nextLayer.waitToComplete();
                    }
                    complete();
                    return false;
                }
                case IDLE -> executeLockCondition.await();
            }
            return true;
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        finally {
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
