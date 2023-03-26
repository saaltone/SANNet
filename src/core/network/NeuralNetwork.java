/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.network;

import core.layer.*;
import core.metrics.ClassificationMetric;
import core.metrics.Metric;
import core.metrics.RegressionMetric;
import core.metrics.SingleRegressionMetric;
import core.optimization.OptimizationType;
import core.optimization.OptimizerFactory;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.sampling.Sampler;
import utils.sampling.Sequence;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Implements neural network.<br>
 * Used to define, construct and execute neural network.<br>
 * Can support multiple layer of different types including regularization, normalization and optimization methods.<br>
 *
 */
public class NeuralNetwork implements Runnable, Serializable {

    @Serial
    private static final long serialVersionUID = -1075977720550636471L;

    /**
     * Defines states of neural network.
     *   IDLE: neural network is idle ready to execute procedure.
     *   TRAIN: initiates training procedure with single or multiple steps.
     *   PREDICT: initiates predict procedure step.
     *   VALIDATE: initiates validate procedure step.
     *   TERMINATED: neural network is terminated (neural network thread is terminated).
     *
     */
    private enum ExecutionState {
        IDLE,
        TRAIN,
        PREDICT,
        VALIDATE,
        TERMINATED
    }

    /**
     * Lock for synchronizing neural network thread operations.
     *
     */
    private transient Lock executeLock;

    /**
     * Lock-condition for synchronizing execution procedures (train, predict, validate).
     *
     */
    private transient Condition executeLockCondition;

    /**
     * Lock-condition for synchronizing completion of procedure execution and shift to idle state.
     *
     */
    private transient Condition completeLockCondition;

    /**
     * Execution state of neural network.
     *
     */
    private transient ExecutionState executionState;

    /**
     * Lock for synchronizing stopping of neural network thread operations.
     *
     */
    private transient Lock stopLock;

    /**
     * Flag if neural network procedure execution has been stopped.
     *
     */
    private transient boolean stopExecution;

    /**
     * Neural network execution thread.
     *
     */
    private transient Thread neuralNetworkThread;

    /**
     * Name of neural network instance.
     *
     */
    private String neuralNetworkName;

    /**
     * Reference to input layer of neural network.
     *
     */
    private final TreeMap<Integer, InputLayer> inputLayers = new TreeMap<>();

    /**
     * List containing hidden layers for neural network.
     *
     */
    private final TreeMap<Integer, AbstractLayer> hiddenLayers = new TreeMap<>();

    /**
     * Reference to output layer of neural network.
     *
     */
    private final TreeMap<Integer, OutputLayer> outputLayers = new TreeMap<>();

    /**
     * List of neural network layers.
     *
     */
    private final TreeMap<Integer, NeuralNetworkLayer> neuralNetworkLayers = new TreeMap<>();

    /**
     * Reference to early stopping condition.
     *
     */
    private final TreeMap<Integer, EarlyStopping> earlyStoppingMap = new TreeMap<>();

    /**
     * Reference to training error metric.
     *
     */
    private final TreeMap<Integer, SingleRegressionMetric> trainingMetrics = new TreeMap<>();

    /**
     * Reference to validation error metric. Default Regression.
     *
     */
    private final TreeMap<Integer, Metric> validationMetrics = new TreeMap<>();

    /**
     * Structure containing prediction input sequence.
     *
     */
    private final TreeMap<Integer, Sequence> predictInputs = new TreeMap<>();

    /**
     * Flag is neural network and it's layers are to be reset prior training phase.
     *
     */
    private transient boolean reset;

    /**
     * Count of total neural network training iterations.
     *
     */
    private int totalIterations = 0;

    /**
     * Total training time of neural network in nanoseconds.
     *
     */
    private long trainingTime = 0;

    /**
     * Length of automatic validation cycle in iterations.
     *
     */
    private int autoValidationCycle = 0;

    /**
     * Iteration count automatic validation cycle.
     *
     */
    private transient int autoValidationCount = 0;

    /**
     * Sampler for training phase.
     *
     */
    private transient Sampler trainingSampler;

    /**
     * Sampler for validation phase.
     *
     */
    private transient Sampler validationSampler;

    /**
     * Reference to neural network persistence instance.
     *
     */
    private transient Persistence persistence;

    /**
     * Flag if neural network training progress is to be verbosed.
     *
     */
    private boolean verboseTraining;

    /**
     * Flag if neural network validation progress is to be verbosed.
     *
     */
    private boolean verboseValidation;

    /**
     * Cycle length as iterators for neural network verbosing.
     *
     */
    private int verboseCycle;

    /**
     * Constructor for neural network.<br>
     * Builds neural network based on neural network configuration.<br>
     *
     * @param neuralNetworkConfiguration neural network configuration.
     * @throws NeuralNetworkException thrown if initialization of layer fails or neural network is already built.
     */
    public NeuralNetwork(NeuralNetworkConfiguration neuralNetworkConfiguration) throws NeuralNetworkException {
        build(neuralNetworkConfiguration);
    }

    /**
     * Constructor for neural network.
     *
     * @param neuralNetworkConfiguration neural network configuration.
     * @param neuralNetworkName name for neural network instance.
     * @throws NeuralNetworkException thrown if initialization of layer fails or neural network is already built.
     */
    public NeuralNetwork(NeuralNetworkConfiguration neuralNetworkConfiguration, String neuralNetworkName) throws NeuralNetworkException {
        this.neuralNetworkName = neuralNetworkName;
        build(neuralNetworkConfiguration);
    }

    /**
     * Sets name for neural network instance.
     *
     * @param neuralNetworkName name for neural network instance.
     */
    public void setNeuralNetworkName(String neuralNetworkName) {
        waitToComplete();
        this.neuralNetworkName = neuralNetworkName;
    }

    /**
     * Returns name of neural network instance.
     *
     * @return name of neural network instance.
     */
    public String getNeuralNetworkName() {
        waitToComplete();
        return neuralNetworkName;
    }

    /**
     * Sets optimizer to specific neural network layer.
     *
     * @param neuralNetworkLayerIndex neural network layer. Input layer is index 0.
     * @param optimization type of optimizer.
     * @param params parameters for optimizer.
     * @throws NeuralNetworkException throws neural network exception if setting of optimizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setOptimizer(int neuralNetworkLayerIndex, OptimizationType optimization, String params) throws NeuralNetworkException, DynamicParamException {
        checkStarted();
        if (neuralNetworkLayerIndex < 0 || neuralNetworkLayerIndex > neuralNetworkLayers.size() - 1) throw new NeuralNetworkException("No neural network layer index: " + neuralNetworkLayerIndex + " exists.");
        NeuralNetworkLayer neuralNetworkLayer = neuralNetworkLayers.get(neuralNetworkLayerIndex);
        if (neuralNetworkLayer instanceof InputLayer) throw new NeuralNetworkException("Input layer cannot be optimized.");
        if (neuralNetworkLayer instanceof OutputLayer) throw new NeuralNetworkException("Output layer cannot be optimized.");
        neuralNetworkLayer.setOptimizer(OptimizerFactory.create(optimization, params));
    }

    /**
     * Sets optimizer to all neural network layers.
     *
     * @param optimization type of optimizer.
     * @param params parameters for optimizer.
     * @throws NeuralNetworkException throws neural network exception if setting of optimizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setOptimizer(OptimizationType optimization, String params) throws NeuralNetworkException, DynamicParamException {
        checkStarted();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers.values()) {
            if (!(neuralNetworkLayer instanceof InputLayer) && !(neuralNetworkLayer instanceof OutputLayer)) neuralNetworkLayer.setOptimizer(OptimizerFactory.create(optimization, params));
        }
    }

    /**
     * Sets optimizer to specific neural network layer.
     *
     * @param neuralNetworkLayerIndex neural network layer. Input layer is index 0.
     * @param optimization type of optimizer.
     * @throws NeuralNetworkException throws neural network exception if setting of optimizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setOptimizer(int neuralNetworkLayerIndex, OptimizationType optimization) throws NeuralNetworkException, DynamicParamException {
        setOptimizer(neuralNetworkLayerIndex, optimization, null);
    }

    /**
     * Sets optimizer to all neural network layers.
     *
     * @param optimization type of optimizer.
     * @throws NeuralNetworkException throws neural network exception if setting of optimizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setOptimizer(OptimizationType optimization) throws NeuralNetworkException, DynamicParamException {
        setOptimizer(optimization, null);
    }

    /**
     * Returns inputs layers.
     *
     * @return input layers.
     */
    public TreeMap<Integer, InputLayer> getInputLayers() {
        return inputLayers;
    }

    /**
     * Returns output layers of neural network.
     *
     * @return output layers of neural network.
     */
    public TreeMap<Integer, OutputLayer> getOutputLayers() {
        return outputLayers;
    }

    /**
     * Returns map of neural network layers.
     *
     * @return map of neural network layers.
     */
    public TreeMap<Integer, NeuralNetworkLayer> getNeuralNetworkLayers() {
        return neuralNetworkLayers;
    }

    /**
     * Builds neural network from neural network configuration.
     *
     * @param neuralNetworkConfiguration neural network configuration.
     * @throws NeuralNetworkException thrown if initialization of layer fails or neural network is already built.
     */
    private void build(NeuralNetworkConfiguration neuralNetworkConfiguration) throws NeuralNetworkException {
        if (!neuralNetworkLayers.isEmpty()) throw new NeuralNetworkException("Neural network is already built.");

        neuralNetworkConfiguration.validate();

        inputLayers.putAll(neuralNetworkConfiguration.getInputLayers());
        hiddenLayers.putAll(neuralNetworkConfiguration.getHiddenLayers());
        outputLayers.putAll(neuralNetworkConfiguration.getOutputLayers());
        neuralNetworkLayers.putAll(neuralNetworkConfiguration.getNeuralNetworkLayers());
    }

    /**
     * Removes last hidden layer.
     *
     * @throws NeuralNetworkException throws exception if removal of last hidden layer fails.
     */
    public void removeLastHiddenLayer() throws NeuralNetworkException {
        waitToComplete();
        if (isStarted()) throw new NeuralNetworkException("Cannot remove hidden layer when neural network is running");
        int lastLayerIndex = hiddenLayers.lastKey();
        int lastNeuralNetworkLayerIndex = -1;
        NeuralNetworkLayer lastNeuralNetworkLayer = null;
        for (Map.Entry<Integer, NeuralNetworkLayer> entry : neuralNetworkLayers.entrySet()) {
            if (entry.getValue() == hiddenLayers.get(lastLayerIndex)) {
                lastNeuralNetworkLayer = entry.getValue();
                lastNeuralNetworkLayerIndex = entry.getKey();
                break;
            }
        }
        if (lastNeuralNetworkLayer == null) throw new NeuralNetworkException("No last neural network layer found.");

        TreeMap<Integer, NeuralNetworkLayer> previousLayers = lastNeuralNetworkLayer.getPreviousLayers();
        TreeMap<Integer, NeuralNetworkLayer> nextLayers = lastNeuralNetworkLayer.getNextLayers();
        for (NeuralNetworkLayer previousLayer : previousLayers.values()) {
            previousLayer.removeNextLayer(lastNeuralNetworkLayer);
            for (NeuralNetworkLayer nextLayer : nextLayers.values()) previousLayer.addNextLayer(nextLayer);
        }
        for (NeuralNetworkLayer nextLayer : nextLayers.values()) {
            nextLayer.removePreviousLayer(lastNeuralNetworkLayer);
            for (NeuralNetworkLayer previousLayer : previousLayers.values()) nextLayer.addPreviousLayer(previousLayer);
        }

        hiddenLayers.remove(lastLayerIndex);
        neuralNetworkLayers.remove(lastNeuralNetworkLayerIndex);
    }

    /**
     * Removes last hidden layers.
     *
     * @param numberOfHiddenLayers number of hidden layers.
     * @throws NeuralNetworkException throws exception if removal of last hidden layer fails.
     */
    public void removeLastHiddenLayers(int numberOfHiddenLayers) throws NeuralNetworkException {
        for (int hiddenLayer = 0; hiddenLayer < numberOfHiddenLayers; hiddenLayer++) removeLastHiddenLayer();
    }

    /**
     * Sets persistence instance to neural network.<br>
     * Persistence is used to serialize neural network instance and store to file or deserialize neural network from file.<br>
     *
     * @param persistence persistence instance.
     */
    public void setPersistence(Persistence persistence) {
        waitToComplete();
        this.persistence = persistence;
    }

    /**
     * Removes persistence instance.
     *
     */
    public void removePersistence() {
        waitToComplete();
        this.persistence = null;
    }

    /**
     * Returns persistence instance of neural network
     *
     * @return persistence instance.
     */
    public Persistence getPersistence() {
        waitToComplete();
        return persistence;
    }

    /**
     * Initializes neural network and it's layers.<br>
     * Starts neural network thread and neural network layer threads.<br>
     *
     * @throws NeuralNetworkException throws exception if starting of neural network fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void start() throws NeuralNetworkException, MatrixException, DynamicParamException {
        if (neuralNetworkLayers.isEmpty()) throw new NeuralNetworkException("Neural network is not built.");
        checkStarted();

        trainingMetrics.clear();
        for (Integer outputLayerIndex : getOutputLayers().keySet()) trainingMetrics.put(outputLayerIndex, new SingleRegressionMetric());
        for (Integer outputLayerIndex : getOutputLayers().keySet()) if (validationMetrics.get(outputLayerIndex) != null) validationMetrics.put(outputLayerIndex, validationMetrics.get(outputLayerIndex).reference());

        if (!earlyStoppingMap.isEmpty()) {
            for (Integer outputLayerIndex : getOutputLayers().keySet()) {
                earlyStoppingMap.get(outputLayerIndex).setTrainingMetric(trainingMetrics.get(outputLayerIndex));
                earlyStoppingMap.get(outputLayerIndex).setValidationMetric(validationMetrics.get(outputLayerIndex));
            }
        }

        executeLock = new ReentrantLock();
        executeLockCondition = executeLock.newCondition();
        completeLockCondition = executeLock.newCondition();
        executionState = ExecutionState.IDLE;

        stopLock = new ReentrantLock();
        stopExecution = false;

        neuralNetworkThread = new Thread(this);
        neuralNetworkThread.setName("NeuralNetwork" + (neuralNetworkName != null ? " (" + neuralNetworkName + ")" : ""));
        neuralNetworkThread.start();

        for (InputLayer inputLayer : inputLayers.values()) inputLayer.start();
    }

    /**
     * Stops neural network.
     *
     */
    public void stop() {
        if (!isStarted()) return;
        waitToComplete();
        executeLock.lock();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers.values()) neuralNetworkLayer.stop();
        executionState = ExecutionState.TERMINATED;
        executeLockCondition.signal();
        executeLock.unlock();

        try {
            neuralNetworkThread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        executeLock = null;
        executeLockCondition = null;
        completeLockCondition = null;
        stopLock = null;
        neuralNetworkThread = null;
    }

    /**
     * Checks if neural network is already started.
     *
     * @return returns true if neural network is started otherwise false.
     */
    public boolean isStarted() {
        return neuralNetworkThread != null && (neuralNetworkThread.getState() != Thread.State.NEW);
    }

    /**
     * Checks if neural network is started (running).
     *
     * @throws NeuralNetworkException throws exception is neural network is started.
     */
    private void checkStarted() throws NeuralNetworkException {
        if (isStarted()) throw new NeuralNetworkException("Neural network is started");
    }

    /**
     * Checks if neural network is not started (running).
     *
     * @throws NeuralNetworkException throws exception is neural network is not started.
     */
    private void checkNotStarted() throws NeuralNetworkException {
        if (!isStarted()) throw new NeuralNetworkException("Neural network is not started");
    }

    /**
     * Aborts execution of neural network.<br>
     * Execution is aborted after execution of last single operation is completed.<br>
     * Useful when neural network is executing multiple training iterations.<br>
     *
     */
    public void abortExecution() {
        if (!isStarted()) return;
        stopLock.lock();
        stopExecution = true;
        stopLock.unlock();
    }

    /**
     * Sets neural network into completed state and makes state transition to idle state.
     *
     */
    private void stateCompleted() {
        executeLock.lock();
        executionState = ExecutionState.IDLE;
        completeLockCondition.signal();
        executeLock.unlock();
    }

    /**
     * Checks if neural network is executing (processing).
     *
     * @return true if neural network is executing otherwise false.
     */
    public boolean isProcessing() {
        if (!isStarted()) return false;
        executeLock.lock();
        boolean isProcessing;
        isProcessing = !(executionState == ExecutionState.IDLE || executionState == ExecutionState.TERMINATED);
        executeLock.unlock();
        return isProcessing;
    }

    /**
     * Waits for neural network to finalize it execution (processing).
     *
     */
    public void waitToComplete() {
        if (!isStarted()) return;
        executeLock.lock();
        if (isProcessing()) completeLockCondition.awaitUninterruptibly();
        executeLock.unlock();
    }

    /**
     * Checks if neural network execution (processing) is stopped.
     *
     * @return true if execution is stopped otherwise false.
     */
    private boolean stoppedExecution() {
        stopLock.lock();
        boolean stopped = false;
        if (stopExecution) {
            stopped = true;
            stopExecution = false;
        }
        stopLock.unlock();
        return stopped;
    }

    /**
     * Sets next state for neural network.
     *
     * @param executionState next state for neural network.
     */
    private void nextState(ExecutionState executionState) {
        this.executionState = executionState;
        executeLockCondition.signal();
        executeLock.unlock();
    }

    /**
     * Sets training sample sets of neural network via sampler.<br>
     * Equal indices of input and output reflect input output pairs of neural network.<br>
     *
     * @param trainingSampler training sampler containing training data set.
     * @throws NeuralNetworkException throws exception if setting of training sample sets fail.
     */
    public void setTrainingData(Sampler trainingSampler) throws NeuralNetworkException {
        checkNotStarted();
        waitToComplete();
        executeLock.lock();
        if (trainingSampler == null) {
            executeLock.unlock();
            throw new NeuralNetworkException("Training sampler is not set.");
        }
        this.trainingSampler = trainingSampler;
        executeLock.unlock();
    }

    /**
     * Sets early stopping conditions.
     *
     * @param earlyStoppingMap early stopping instances.
     * @throws NeuralNetworkException throws exception if early stopping is not defined.
     */
    public void setTrainingEarlyStopping(TreeMap<Integer, EarlyStopping> earlyStoppingMap) throws NeuralNetworkException {
        waitToComplete();
        if (earlyStoppingMap == null) throw new NeuralNetworkException("Early stopping is not defined.");
        this.earlyStoppingMap.clear();
        this.earlyStoppingMap.putAll(earlyStoppingMap);
        for (Integer earlyStoppingIndex : this.earlyStoppingMap.keySet()) {
            earlyStoppingMap.get(earlyStoppingIndex).setTrainingMetric(trainingMetrics.get(earlyStoppingIndex));
            earlyStoppingMap.get(earlyStoppingIndex).setValidationMetric(validationMetrics.get(earlyStoppingIndex));
        }
    }

    /**
     * Sets auto validation on.
     *
     * @param autoValidationCycle validation cycle in iterations.
     * @throws NeuralNetworkException throws exception if number of auto validation cycles are below 1.
     */
    public void setAutoValidate(int autoValidationCycle) throws NeuralNetworkException {
        waitToComplete();
        if (autoValidationCycle < 1) throw new NeuralNetworkException("Auto validation cycle size must be at least 1.");
        this.autoValidationCycle = autoValidationCycle;
    }

    /**
     * Unsets auto validation.
     *
     */
    public void unsetAutoValidate() {
        waitToComplete();
        autoValidationCycle = 0;
    }

    /**
     * Trains neural network.
     *
     * @throws NeuralNetworkException throws exception if training procedure fails.
     */
    public void train() throws NeuralNetworkException {
        train(null, false, true);
    }

    /**
     * Trains neural network and sets specific training input and output samples.
     *
     * @param trainingSampler training sampler containing training data.
     * @throws NeuralNetworkException throws exception if training procedure fails.
     */
    public void train(Sampler trainingSampler) throws NeuralNetworkException {
        train(trainingSampler, false, true);
    }

    /**
     * Trains neural network with option to reset neural network and it's layers.
     *
     * @param reset if true resets neural network and it's layers.
     * @throws NeuralNetworkException throws exception if training procedure fails.
     */
    public void train(boolean reset) throws NeuralNetworkException {
        train(null, reset, true);
    }

    /**
     * Trains neural network with option to reset neural network and it's layers.<br>
     * Sets specific training input and output samples.<br>
     *
     * @param trainingSampler training sampler containing training data.
     * @param reset if true resets neural network and it's layers.
     * @throws NeuralNetworkException throws exception if training procedure fails.
     */
    public void train(Sampler trainingSampler, boolean reset) throws NeuralNetworkException {
        train(trainingSampler, reset, true);
    }

    /**
     * Trains neural network with option to reset neural network and it's layers.<br>
     * Optionally waits neural network training procedure to complete.<br>
     *
     * @param reset if true resets neural network and it's layers.
     * @param waitToComplete if true waits for neural network execution complete otherwise returns function prior training completion.
     * @throws NeuralNetworkException throws exception if training procedure fails.
     */
    public void train(boolean reset, boolean waitToComplete) throws NeuralNetworkException {
        train(null, reset, waitToComplete);
    }

    /**
     * Trains neural network with option to reset neural network and it's layers.<br>
     * Sets specific training input and output samples.<br>
     * Optionally waits neural network training procedure to complete.<br>
     *
     * @param trainingSampler training sampler containing training data.
     * @param reset if true resets neural network and it's layers.
     * @param waitToComplete if true waits for neural network execution complete otherwise returns function prior training completion.
     * @throws NeuralNetworkException throws exception if training procedure fails.
     */
    public void train(Sampler trainingSampler, boolean reset, boolean waitToComplete) throws NeuralNetworkException {
        checkNotStarted();
        waitToComplete();
        executeLock.lock();
        if (trainingSampler != null) this.trainingSampler = trainingSampler;
        if (this.trainingSampler == null) throw new NeuralNetworkException("Training sampler is not set.");
        this.reset = reset;
        nextState(ExecutionState.TRAIN);
        if (waitToComplete) waitToComplete();
    }

    /**
     * Verboses (prints to console) neural network training progress.<br>
     * Print information of neural network training iteration count, training time and training error.<br>
     *
     * @param verboseCycle verbose cycle in iterations.
     */
    public void verboseTraining(int verboseCycle) {
        waitToComplete();
        verboseTraining = true;
        this.verboseCycle = verboseCycle;
    }

    /**
     * Unsets verbosing of neural network training progress.
     *
     */
    public void unverboseTraining() {
        waitToComplete();
        verboseTraining = false;
    }

    /**
     * Returns neural network training time in milliseconds.
     *
     * @return neural network training time in milliseconds.
     */
    public long getTrainingTimeInMilliseconds() {
        waitToComplete();
        return trainingTime / 1000000;
    }

    /**
     * Returns neural network training time in seconds.
     *
     * @return neural network training time in seconds.
     */
    public long getTrainingTimeInSeconds() {
        waitToComplete();
        return trainingTime / 1000000000;
    }

    /**
     * Sets reset flag for procedure expression dependencies.
     *
     * @param resetDependencies if true procedure expression dependencies are reset otherwise false.
     */
    public void resetDependencies(boolean resetDependencies) {
        waitToComplete();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers.values()) neuralNetworkLayer.resetDependencies(resetDependencies);
    }

    /**
     * Predicts values based on current test set inputs.
     *
     * @return predicted values (neural network output).
     * @throws NeuralNetworkException throws exception if prediction fails.
     */
    public TreeMap<Integer, Sequence>  predict() throws NeuralNetworkException {
        return predict(null, true);
    }

    /**
     * Predicts values based on given input.
     *
     * @param inputs inputs for prediction.
     * @return predicted values (neural network outputs).
     * @throws NeuralNetworkException throws exception if prediction fails.
     */
    public TreeMap<Integer, Matrix> predictMatrix(TreeMap<Integer, Matrix> inputs) throws NeuralNetworkException {
        if (inputs.isEmpty()) throw new NeuralNetworkException("No prediction inputs set");
        TreeMap<Integer, Matrix> outputs = new TreeMap<>();
        for (Map.Entry<Integer, Sequence> entry : predict(Sequence.getSequencesFromMatrices(inputs), true).entrySet()) outputs.put(entry.getKey(), entry.getValue().get(0));
        return outputs;
    }

    /**
     * Predicts values based on current test set inputs.<br>
     * Sets specific inputs for prediction.<br>
     *
     * @param inputs test input set for prediction.
     * @return predicted values (neural network output).
     * @throws NeuralNetworkException throws exception if prediction fails.
     */
    public TreeMap<Integer, Sequence> predict(TreeMap<Integer, Sequence>  inputs) throws NeuralNetworkException {
        return predict(inputs, true);
    }

    /**
     * Predicts values based on current test set inputs.<br>
     * Optionally waits neural network prediction procedure to complete.<br>
     *
     * @param waitToComplete if true waits for neural network execution complete otherwise returns function prior prediction completion.
     * @return predicted values (neural network output).
     * @throws NeuralNetworkException throws exception if prediction fails.
     */
    public TreeMap<Integer, Sequence> predict(boolean waitToComplete) throws NeuralNetworkException {
        return predict(null, waitToComplete);
    }

    /**
     * Predicts values based on given inputs.<br>
     * Optionally waits neural network prediction procedure to complete.<br>
     *
     * @param inputs test input set for prediction.
     * @param waitToComplete if true waits for neural network execution complete otherwise returns function prior prediction completion.
     * @return predicted values (neural network output).
     * @throws NeuralNetworkException throws exception if prediction fails.
     */
    public TreeMap<Integer, Sequence> predict(TreeMap<Integer, Sequence> inputs, boolean waitToComplete) throws NeuralNetworkException {
        checkNotStarted();
        waitToComplete();
        if (inputs == null) throw new NeuralNetworkException("No prediction inputs set");
        executeLock.lock();
        predictInputs.clear();
        predictInputs.putAll(inputs);
        nextState(ExecutionState.PREDICT);
        return waitToComplete ? getOutput() : null;
    }

    /**
     * Sets verbosing for validation phase.<br>
     * Follows training verbosing cycle.<br>
     *
     */
    public void verboseValidation() {
        waitToComplete();
        verboseValidation = true;
    }

    /**
     * Unsets verbosing for validation phase.
     *
     */
    public void unverboseValidation() {
        waitToComplete();
        verboseValidation = false;
    }

    /**
     * Validates neural network with current test input and output (actual true values) sample set.
     *
     * @throws NeuralNetworkException throws exception if validation fails.
     */
    public void validate() throws NeuralNetworkException {
        validate(null, true);
    }

    /**
     * Validates neural network with current test input and output (actual true values) sample set.<br>
     * Optionally waits neural network validation procedure to complete.<br>
     *
     * @param waitToComplete if true waits for neural network execution complete otherwise returns function prior validation completion.
     * @throws NeuralNetworkException throws exception if validation fails.
     */
    public void validate(boolean waitToComplete) throws NeuralNetworkException {
        validate(null, waitToComplete);
    }

    /**
     * Validates neural network.<br>
     * Sets specific input and output (actual true values) test samples.<br>
     *
     * @param validationSampler validation sampler containing validation data set.
     * @throws NeuralNetworkException throws exception if validation fails.
     */
    public void validate(Sampler validationSampler) throws NeuralNetworkException {
        validate(validationSampler, true);
    }

    /**
     * Sets validation data.
     *
     * @param validationSampler validation sampler containing validation data set.
     * @throws NeuralNetworkException throws exception if setting of validation data fails.
     */
    public void setValidationData(Sampler validationSampler) throws NeuralNetworkException {
        checkNotStarted();
        waitToComplete();
        executeLock.lock();
        if (validationSampler == null) {
            executeLock.unlock();
            throw new NeuralNetworkException("Validation sampler is not set.");
        }
        this.validationSampler = validationSampler;
        executeLock.unlock();
    }

    /**
     * Validates neural network.<br>
     * Sets specific input and output (actual true values) test samples.<br>
     * Optionally waits neural network validation procedure to complete.<br>
     *
     * @param validationSampler validation sampler containing validation data set.
     * @param waitToComplete if true waits for neural network execution complete otherwise returns function prior validation completion.
     * @throws NeuralNetworkException throws exception if validation fails.
     */
    public void validate(Sampler validationSampler, boolean waitToComplete) throws NeuralNetworkException {
        checkNotStarted();
        waitToComplete();
        executeLock.lock();
        if (validationSampler != null) this.validationSampler = validationSampler;
        if (this.validationSampler == null) throw new NeuralNetworkException("Validation sampler is not set.");
        nextState(ExecutionState.VALIDATE);
        if (waitToComplete) waitToComplete();
    }

    /**
     * Returns output of neural network (output layer).
     *
     * @return output of neural network.
     */
    public TreeMap<Integer, Sequence> getOutput() {
        waitToComplete();
        TreeMap<Integer, Sequence> outputs = new TreeMap<>();
        for (Map.Entry<Integer, OutputLayer> entry : getOutputLayers().entrySet()) outputs.put(entry.getKey(), entry.getValue().getLayerOutputs());
        return outputs;
    }

    /**
     * Thread run function.<br>
     * Executes given neural network procedures and synchronizes their execution via neural network thread execution lock.<br>
     *
     */
    public void run() {
        while (true) {
            executeLock.lock();
            try {
                switch (executionState) {
                    case TRAIN -> trainIterations();
                    case PREDICT -> predictInput();
                    case VALIDATE -> validateInput(true);
                    case TERMINATED -> {
                        return;
                    }
                }
            }
            catch (Exception exception) {
                exception.printStackTrace();
                System.exit(-1);
            }
            executeLock.unlock();
        }
    }

    /**
     * Trains neural network with defined number of iterations.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws IOException throws exception if neural network persistence operation fails.
     * @throws NeuralNetworkException throws exception if neural network training fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private void trainIterations() throws MatrixException, IOException, NeuralNetworkException, DynamicParamException {
        trainingSampler.reset();
        int numberOfIterations = trainingSampler.getNumberOfIterations();
        for (int iteration = 0; iteration < numberOfIterations; iteration++) {
            trainIteration();
            if (stoppedExecution()) break;
            if (!earlyStoppingMap.isEmpty()) {
                boolean stopTraining = true;
                for (EarlyStopping earlyStopping : earlyStoppingMap.values()) if (!earlyStopping.stopTraining()) stopTraining = false;
                if (stopTraining) break;
            }
        }
        stateCompleted();
    }

    /**
     * Trains single neural network iteration.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws IOException throws exception if neural network persistence operation fails.
     * @throws NeuralNetworkException throws exception if neural network training fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private void trainIteration() throws MatrixException, IOException, NeuralNetworkException, DynamicParamException {
        long startTime = System.nanoTime();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers.values()) if (reset) neuralNetworkLayer.resetOptimizer();
        for (SingleRegressionMetric trainingMetric : trainingMetrics.values()) trainingMetric.reset();
        TreeMap<Integer, Sequence> inputSequences = new TreeMap<>();
        TreeMap<Integer, Sequence> outputSequences = new TreeMap<>();
        trainingSampler.getSamples(inputSequences, outputSequences);
        for (Map.Entry<Integer, OutputLayer> entry : getOutputLayers().entrySet()) entry.getValue().setTargets(outputSequences.get(entry.getKey()));
        for (Map.Entry<Integer, InputLayer> entry : getInputLayers().entrySet()) entry.getValue().train(inputSequences.get(entry.getKey()));
        for (Map.Entry<Integer, OutputLayer> entry : getOutputLayers().entrySet()) entry.getValue().backward();
        for (Map.Entry<Integer, InputLayer> entry : getInputLayers().entrySet()) entry.getValue().update();
        long endTime = System.nanoTime();
        trainingTime += endTime - startTime;
        for (Map.Entry<Integer, SingleRegressionMetric> entry : trainingMetrics.entrySet()) entry.getValue().report(getOutputLayers().get(entry.getKey()).getTotalError());
        totalIterations++;
        if (autoValidationCycle > 0) {
            autoValidationCount++;
            if (autoValidationCount >= autoValidationCycle) {
                validateInput(false);
                if (!earlyStoppingMap.isEmpty()) for (EarlyStopping earlyStopping : earlyStoppingMap.values()) earlyStopping.evaluateValidationCondition(totalIterations);
                autoValidationCount = 0;
            }
        }
        if (!earlyStoppingMap.isEmpty()) for (EarlyStopping earlyStopping : earlyStoppingMap.values()) earlyStopping.evaluateTrainingCondition(totalIterations);
        if (verboseTraining) verboseTrainingStatus();
        if (persistence != null) persistence.cycle();
    }

    /**
     * Verboses (prints to console) neural network training status.<br>
     * Prints number of iteration, neural network training time and training error.<br>
     *
     */
    private void verboseTrainingStatus() {
        StringBuilder meanSquaredError = new StringBuilder("[ ");
        for (SingleRegressionMetric trainingMetric : trainingMetrics.values()) {
            meanSquaredError.append(String.format("%.4f", trainingMetric.getMeanSquaredError())).append(" ");
        }
        meanSquaredError.append("]");
        if (totalIterations % verboseCycle == 0) System.out.println((neuralNetworkName != null ? neuralNetworkName + ": " : "") + "Training error (iteration #" + totalIterations +"): " + meanSquaredError + ", Training time in seconds: " + (trainingTime / 1000000000));
    }

    /**
     * Predicts using given test set inputs.
     *
     */
    private void predictInput() {
        for (Map.Entry<Integer, InputLayer> entry : getInputLayers().entrySet()) entry.getValue().predict(predictInputs.get(entry.getKey()));
        stateCompleted();
    }

    /**
     * Validates with given test set inputs and outputs.
     *
     * @param stateCompleted if flag is sets calls stateCompleted function.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if validation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private void validateInput(boolean stateCompleted) throws MatrixException, NeuralNetworkException, DynamicParamException {
        for (Metric validationMetric : validationMetrics.values()) validationMetric.reset();
        validationSampler.reset();
        int numberOfIterations = validationSampler.getNumberOfIterations();
        for (int sampleIndex = 0; sampleIndex < numberOfIterations; sampleIndex++) {
            TreeMap<Integer, Sequence> inputSequences = new TreeMap<>();
            TreeMap<Integer, Sequence> outputSequences = new TreeMap<>();
            validationSampler.getSamples(inputSequences, outputSequences);
            for (Map.Entry<Integer, InputLayer> entry : getInputLayers().entrySet()) entry.getValue().predict(inputSequences.get(entry.getKey()));
            for (Map.Entry<Integer, OutputLayer> entry : getOutputLayers().entrySet()) validationMetrics.get(entry.getKey()).report(entry.getValue().getLayerOutputs(), outputSequences.get(entry.getKey()));
        }
        if (verboseValidation && (totalIterations % verboseCycle == 0)) verboseValidationStatus();
        if (stateCompleted) stateCompleted();
    }

    /**
     * Returns training metrics instance.
     *
     * @return training metrics instance.
     */
    public TreeMap<Integer, SingleRegressionMetric> getTrainingMetrics() {
        waitToComplete();
        return trainingMetrics;
    }

    /**
     * Returns total neural network training iterations count.
     *
     * @return total neural network training iterations count.
     */
    public int getTotalIterations() {
        waitToComplete();
        return totalIterations;
    }

    /**
     * Returns neural network output error.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return neural network output error.
     */
    public TreeMap<Integer, Double> getOutputError() throws MatrixException, DynamicParamException {
        waitToComplete();
        TreeMap<Integer, Double> outputErrors = new TreeMap<>();
        for (Map.Entry<Integer, OutputLayer> entry : getOutputLayers().entrySet()) outputErrors.put(entry.getKey(), entry.getValue().getTotalError());
        return outputErrors;
    }

    /**
     * Sets regression as validation metric.
     *
     */
    public void setAsRegression() {
        waitToComplete();
        validationMetrics.clear();
        for (Integer outputLayerIndex : getOutputLayers().keySet()) validationMetrics.put(outputLayerIndex, new RegressionMetric());
    }

    /**
     * Sets regression as validation metric.
     *
     * @param outputLayerIndex output layer index.
     */
    public void setAsRegression(int outputLayerIndex) {
        waitToComplete();
        validationMetrics.put(outputLayerIndex, new RegressionMetric());
    }

    /**
     * Sets regression as validation metric.
     *
     * @param useR2AsLastError if true uses R2 as last error otherwise uses MSE.
     */
    public void setAsRegression(boolean useR2AsLastError) {
        waitToComplete();
        validationMetrics.clear();
        for (Integer outputLayerIndex : getOutputLayers().keySet()) validationMetrics.put(outputLayerIndex, new RegressionMetric(useR2AsLastError));
    }

    /**
     * Sets regression as validation metric.
     *
     * @param outputLayerIndex output layer index.
     * @param useR2AsLastError if true uses R2 as last error otherwise uses MSE.
     */
    public void setAsRegression(int outputLayerIndex, boolean useR2AsLastError) {
        waitToComplete();
        validationMetrics.put(outputLayerIndex, new RegressionMetric(useR2AsLastError));
    }

    /**
     * Sets classification as validation metric.
     *
     */
    public void setAsClassification() {
        waitToComplete();
        validationMetrics.clear();
        for (Integer outputLayerIndex : getOutputLayers().keySet()) validationMetrics.put(outputLayerIndex, new ClassificationMetric());
    }

    /**
     * Sets classification as validation metric.
     *
     * @param outputLayerIndex output layer index.
     */
    public void setAsClassification(int outputLayerIndex) {
        waitToComplete();
        validationMetrics.put(outputLayerIndex, new ClassificationMetric());
    }

    /**
     * Sets regression as validation metric.
     *
     * @param multiClass if true metrics assumes multi class classification otherwise single class classification.
     */
    public void setAsClassification(boolean multiClass) {
        waitToComplete();
        validationMetrics.clear();
        for (Integer outputLayerIndex : getOutputLayers().keySet()) validationMetrics.put(outputLayerIndex, new ClassificationMetric(multiClass));
    }

    /**
     * Sets regression as validation metric.
     *
     * @param outputLayerIndex output layer index.
     * @param multiClass if true metrics assumes multi class classification otherwise single class classification.
     */
    public void setAsClassification(int outputLayerIndex, boolean multiClass) {
        waitToComplete();
        validationMetrics.put(outputLayerIndex, new ClassificationMetric(multiClass));
    }

    /**
     * Sets if confusion matrix is printed along other classification metrics.
     *
     * @param printConfusionMatrix if true confusion matrix is printed along other classification metrics.
     */
    public void printConfusionMatrix(boolean printConfusionMatrix) {
        waitToComplete();
        for (Metric validationMetric : validationMetrics.values()) if (validationMetric instanceof  ClassificationMetric) ((ClassificationMetric) validationMetric).setPrintConfusionMatrix(printConfusionMatrix);
    }

    /**
     * Returns validation metrics instance.
     *
     * @return validation metrics instance.
     */
    public TreeMap<Integer, Metric> getValidationMetrics() {
        waitToComplete();
        return validationMetrics;
    }

    /**
     * Verboses validation metrics.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private void verboseValidationStatus() throws MatrixException, DynamicParamException {
        System.out.println((neuralNetworkName != null ? neuralNetworkName + ": " : "") + "Validating...");
        for (Metric validationMetric : validationMetrics.values()) validationMetric.printReport();
    }

    /**
     * Makes deep copy of neural network by using object serialization.
     *
     * @return copy of this neural network.
     * @throws IOException throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public NeuralNetwork copy() throws IOException, ClassNotFoundException, MatrixException {
        waitToComplete();
        predictInputs.clear();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers.values()) neuralNetworkLayer.reset();
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        ObjectOutputStream objectOutputStream = new ObjectOutputStream(byteArrayOutputStream);
        objectOutputStream.writeObject(this);
        objectOutputStream.flush();
        ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(byteArrayOutputStream.toByteArray());
        ObjectInputStream objectInputStream = new ObjectInputStream(byteArrayInputStream);
        NeuralNetwork neuralNetwork = (NeuralNetwork)objectInputStream.readObject();
        objectInputStream.close();
        objectOutputStream.close();
        return neuralNetwork;
    }

    /**
     * Returns reference to neural network.
     *
     * @return reference to neural network.
     * @throws IOException throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public NeuralNetwork reference() throws IOException, ClassNotFoundException, MatrixException, DynamicParamException {
        NeuralNetwork neuralNetwork = copy();
        neuralNetwork.reinitialize();
        return neuralNetwork;
    }

    /**
     * Reinitializes neural network.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void reinitialize() throws MatrixException, DynamicParamException {
        waitToComplete();
        predictInputs.clear();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers.values()) neuralNetworkLayer.reinitialize();
        for (Map.Entry<Integer, EarlyStopping> entry : earlyStoppingMap.entrySet()) earlyStoppingMap.put(entry.getKey(), entry.getValue().reference());
        for (Metric validationMetric : validationMetrics.values()) validationMetric.reset();
        totalIterations = 0;
        trainingTime = 0;
    }

    /**
     * Appends other neural network to this neural network by weight tau. Effectively appends each weight and bias matrix of each layer by this weight factor.
     *
     * @param otherNeuralNetwork other neural network that contributes to this neural network.
     * @param tau tau which controls contribution of other neural network.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void append(NeuralNetwork otherNeuralNetwork, double tau) throws MatrixException {
        waitToComplete();
        int neuralNetworkLayersSize = neuralNetworkLayers.size();
        for (int index = 0; index < neuralNetworkLayersSize; index++) {
            neuralNetworkLayers.get(index).append(otherNeuralNetwork.getNeuralNetworkLayers().get(index), tau);
        }
    }

    /**
     * Sets importance sampling weights to output layer.
     *
     * @param importanceSamplingWeights importance sampling weights
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    public void setImportanceSamplingWeights(TreeMap<Integer, HashMap<Integer, Double>> importanceSamplingWeights) throws NeuralNetworkException {
        checkNotStarted();
        waitToComplete();
        executeLock.lock();
        for (Map.Entry<Integer, OutputLayer> entry : getOutputLayers().entrySet()) entry.getValue().setImportanceSamplingWeights(importanceSamplingWeights.get(entry.getKey()));
        executeLock.unlock();
    }


    /**
     * Prints structure and metadata of neural network.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    public void print() throws NeuralNetworkException {
        checkNotStarted();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers.values()) {
            neuralNetworkLayer.print();
            System.out.println();
        }
        System.out.println("Apply early stopping: " + (!earlyStoppingMap.isEmpty() ? "Yes" : "No"));
        System.out.println();
    }

    /**
     * Prints expression chains of neural network.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    public void printExpressions() throws NeuralNetworkException {
        checkNotStarted();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers.values()) neuralNetworkLayer.printExpressions();
    }

    /**
     * Prints gradient chains of neural network.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    public void printGradients() throws NeuralNetworkException {
        checkNotStarted();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers.values()) neuralNetworkLayer.printGradients();
    }

}

