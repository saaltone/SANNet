/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.network;

import core.layer.AbstractLayer;
import core.layer.InputLayer;
import core.layer.NeuralNetworkLayer;
import core.layer.OutputLayer;
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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Implements neural network.<br>
 * Used to define, construct and execute neural network.<br>
 * Can support multiple layer of different types including regularization, normalization and optimization methods.<br>
 *
 */
public class NeuralNetwork implements Serializable {

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
     * Lock for synchronizing neural network thread operations.
     *
     */
    private transient Lock completeLock;

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
     * Network thread pool.
     *
     */
    private transient ExecutorService networkThreadPool;

    /**
     * Layer thread pool.
     *
     */
    private transient ExecutorService layerThreadPool;

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
     * Reference to input layer groups of neural network.
     *
     */
    private final TreeMap<Integer, TreeMap<Integer, InputLayer>> inputLayerGroups = new TreeMap<>();

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
     * Reference to output layer groups of neural network.
     *
     */
    private final TreeMap<Integer, TreeMap<Integer, OutputLayer>> outputLayerGroups = new TreeMap<>();

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
     * If true shows training metrics otherwise not.
     *
     */
    private boolean showTrainingMetrics = false;

    /**
     * Reference to validation error metric. Default Regression.
     *
     */
    private final TreeMap<Integer, Metric> validationMetrics = new TreeMap<>();

    /**
     * Structure containing prediction input sequence.
     *
     */
    private transient TreeMap<Integer, Sequence> predictInputs;

    /**
     * Flag is neural network and it's layers are to be reset prior training phase.
     *
     */
    private transient boolean reset;

    /**
     * Count of total neural network training iterations.
     *
     */
    private int totalTrainingIterations = 0;

    /**
     * Total training time of neural network in nanoseconds.
     *
     */
    private long trainingTime = 0;

    /**
     * Total validation time of neural network in nanoseconds.
     *
     */
    private long validationTime = 0;

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
     * Returns input layers.
     *
     * @return input layers.
     */
    public TreeMap<Integer, InputLayer> getInputLayers() {
        return inputLayers;
    }

    /**
     * Returns inputs layer groups.
     *
     * @return inputs layer groups.
     */
    public TreeMap<Integer, TreeMap<Integer, InputLayer>> getInputLayerGroups() {
        return new TreeMap<>() {{ putAll(inputLayerGroups); }};
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
     * Returns output layer groups.
     *
     * @return output layer groups.
     */
    public TreeMap<Integer, TreeMap<Integer, OutputLayer>> getOutputLayerGroups() {
        return new TreeMap<>() {{ putAll(outputLayerGroups); }};
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
        inputLayerGroups.putAll(neuralNetworkConfiguration.getInputLayerGroups());
        hiddenLayers.putAll(neuralNetworkConfiguration.getHiddenLayers());
        outputLayers.putAll(neuralNetworkConfiguration.getOutputLayers());
        outputLayerGroups.putAll(neuralNetworkConfiguration.getOutputLayerGroups());
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

        if (persistence != null) persistence = persistence.reference(this);
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
        checkStarted();
        if (neuralNetworkLayers.isEmpty()) throw new NeuralNetworkException("Neural network is not built.");

        for (Integer outputLayerIndex : getOutputLayers().keySet()) {
            trainingMetrics.put(outputLayerIndex, new SingleRegressionMetric(showTrainingMetrics));
            if (validationMetrics.get(outputLayerIndex) != null) validationMetrics.put(outputLayerIndex, validationMetrics.get(outputLayerIndex).reference());
            if (!earlyStoppingMap.isEmpty()) {
                earlyStoppingMap.get(outputLayerIndex).setTrainingMetric(trainingMetrics.get(outputLayerIndex));
                earlyStoppingMap.get(outputLayerIndex).setValidationMetric(validationMetrics.get(outputLayerIndex));
            }
        }

        executeLock = new ReentrantLock();
        executeLockCondition = executeLock.newCondition();
        completeLock = new ReentrantLock();
        completeLockCondition = completeLock.newCondition();
        executionState = ExecutionState.IDLE;
        stopLock = new ReentrantLock();
        stopExecution = false;

        networkThreadPool = Executors.newSingleThreadExecutor();
        executeLayer(networkThreadPool);

        layerThreadPool = Executors.newCachedThreadPool();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers.values()) neuralNetworkLayer.start(layerThreadPool);
    }

    /**
     * Stops neural network.
     *
     */
    public void stop() {
        if (!isStarted()) return;
        waitToComplete();
        nextState(ExecutionState.TERMINATED);
        for (NeuralNetworkLayer neuralNetworkLayer : inputLayers.values()) neuralNetworkLayer.stop();

        try {
            layerThreadPool.shutdownNow();
            networkThreadPool.shutdownNow();
            if (!layerThreadPool.awaitTermination(10, TimeUnit.SECONDS) || !networkThreadPool.awaitTermination(10, TimeUnit.SECONDS)) {
                System.out.println("Failed to shut down neural network.");
            }
        }
        catch (InterruptedException ignored) {
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
                while (!executeLayerOperation()) {}
            } catch (Exception exception) {
                throw new RuntimeException(exception);
            }
        });
    }

    /**
     * Thread run function.<br>
     * Executes given neural network procedures and synchronizes their execution via neural network thread execution lock.<br>
     *
     * @return return true if layer has been terminated otherwise returns true.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws IOException throws exception if neural network persistence operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private boolean executeLayerOperation() throws MatrixException, NeuralNetworkException, DynamicParamException, IOException {
        try {
            executeLock.lock();
            switch (executionState) {
                case TRAIN -> {
                    trainIterations();
                    complete();
                }
                case VALIDATE -> {
                    validateInput();
                    complete();
                }
                case PREDICT -> {
                    predictInput();
                    complete();
                }
                case TERMINATED -> {
                    complete();
                    return true;
                }
            }
        }
        finally {
            executeLock.unlock();
        }
        return false;
    }

    /**
     * Checks if neural network is already started.
     *
     * @return returns true if neural network is started otherwise false.
     */
    public boolean isStarted() {
        return networkThreadPool != null && !networkThreadPool.isTerminated();
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
     * Sets neural network into completed state and makes state to idle state.
     *
     */
    private void complete() {
        try {
            completeLock.lock();
            executionState = ExecutionState.IDLE;
            completeLockCondition.signal();
        }
        finally {
            completeLock.unlock();
        }
    }

    /**
     * Waits for neural network to finalize it execution (processing).
     *
     */
    public void waitToComplete() {
        if (!isStarted()) return;
        try {
            completeLock.lock();
            while (executionState != ExecutionState.IDLE) completeLockCondition.await();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        finally {
            completeLock.unlock();
        }
    }

    /**
     * Stops execution.
     *
     */
    public void stopExecution() {
        try {
            stopLock.lock();
            stopExecution = true;
        }
        finally {
            stopLock.unlock();
        }
    }

    /**
     * Checks if neural network execution (processing) is stopped.
     *
     * @return true if execution is stopped otherwise false.
     */
    private boolean stoppedExecution() {
        try {
            stopLock.lock();
            return stopExecution;
        }
        finally {
            stopExecution = false;
            stopLock.unlock();
        }
    }

    /**
     * Sets next state for neural network.
     *
     * @param executionState next state for neural network.
     */
    private void nextState(ExecutionState executionState) {
        try {
            executeLock.lock();
            this.executionState = executionState;
            executeLockCondition.signal();
        }
        finally {
            executeLock.unlock();
        }
    }

    /**
     * Sets training sample sets of neural network via sampler.<br>
     * Equal indices of input and output reflect input output pairs of neural network.<br>
     *
     * @param trainingSampler training sampler containing training data set.
     * @throws NeuralNetworkException throws exception if setting of training sample sets fail.
     */
    public void setTrainingData(Sampler trainingSampler) throws NeuralNetworkException {
        waitToComplete();
        if (trainingSampler == null) throw new NeuralNetworkException("Training sampler is not set.");
        this.trainingSampler = trainingSampler;
    }

    /**
     * Sets early stopping conditions.
     *
     * @param newEarlyStoppingMap early stopping instances.
     * @throws NeuralNetworkException throws exception if early stopping is not defined.
     */
    public void setTrainingEarlyStopping(TreeMap<Integer, EarlyStopping> newEarlyStoppingMap) throws NeuralNetworkException {
        waitToComplete();
        if (newEarlyStoppingMap == null) throw new NeuralNetworkException("Early stopping is not defined.");
        earlyStoppingMap.clear();
        earlyStoppingMap.putAll(newEarlyStoppingMap);
        for (Integer earlyStoppingIndex : newEarlyStoppingMap.keySet()) {
            earlyStoppingMap.get(earlyStoppingIndex).setTrainingMetric(trainingMetrics.get(earlyStoppingIndex));
        }
        for (Integer earlyStoppingIndex : newEarlyStoppingMap.keySet()) {
            earlyStoppingMap.get(earlyStoppingIndex).setValidationMetric(validationMetrics.get(earlyStoppingIndex));
        }
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
        if (trainingSampler != null) this.trainingSampler = trainingSampler;
        if (this.trainingSampler == null) throw new NeuralNetworkException("Training sampler is not set.");
        this.reset = reset;
        nextState(ExecutionState.TRAIN);
        if (waitToComplete) waitToComplete();
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
            if (stoppedExecution()) return;
            if (!earlyStoppingMap.isEmpty()) {
                boolean stopTraining = true;
                for (EarlyStopping earlyStopping : earlyStoppingMap.values()) if (!earlyStopping.stopTraining()) stopTraining = false;
                if (stopTraining) return;
            }
        }
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
        totalTrainingIterations++;
        long trainingStartTime = System.nanoTime();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers.values()) if (reset) neuralNetworkLayer.resetOptimizer();
        TreeMap<Integer, Sequence> inputSequences = new TreeMap<>();
        TreeMap<Integer, Sequence> outputSequences = new TreeMap<>();
        trainingSampler.getSamples(inputSequences, outputSequences);
        for (Map.Entry<Integer, OutputLayer> entry : getOutputLayers().entrySet()) entry.getValue().setTargets(outputSequences.get(entry.getKey()));
        for (Map.Entry<Integer, InputLayer> entry : getInputLayers().entrySet()) entry.getValue().train(inputSequences.get(entry.getKey()));
        for (Map.Entry<Integer, OutputLayer> entry : getOutputLayers().entrySet()) entry.getValue().backward();
        for (Map.Entry<Integer, InputLayer> entry : getInputLayers().entrySet()) entry.getValue().update();
        long trainingEndTime = System.nanoTime();
        trainingTime += trainingEndTime - trainingStartTime;
        for (Map.Entry<Integer, SingleRegressionMetric> entry : trainingMetrics.entrySet()) entry.getValue().report(getOutputLayers().get(entry.getKey()).getTotalError());
        if (!earlyStoppingMap.isEmpty()) for (EarlyStopping earlyStopping : earlyStoppingMap.values()) earlyStopping.evaluateTrainingCondition(totalTrainingIterations);
        if (autoValidationCycle > 0) {
            autoValidationCount++;
            if (autoValidationCount >= autoValidationCycle) {
                long validationStartTime = System.nanoTime();
                validateInput();
                long validationEndTime = System.nanoTime();
                validationTime += validationEndTime - validationStartTime;
                if (!earlyStoppingMap.isEmpty()) for (EarlyStopping earlyStopping : earlyStoppingMap.values()) earlyStopping.evaluateValidationCondition(totalTrainingIterations);
                autoValidationCount = 0;
            }
        }
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
            meanSquaredError.append(String.format("%.4f", trainingMetric.getLastAbsoluteError())).append(" ");
        }
        meanSquaredError.append("]");
        if (totalTrainingIterations % verboseCycle == 0) System.out.println((neuralNetworkName != null ? neuralNetworkName + ": " : "") + "Training error (iteration #" + totalTrainingIterations +"): " + meanSquaredError + ", Training time: " + String.format("%.3f", trainingTime / Math.pow(10, 9)) + "s" + (autoValidationCycle > 0 ? ", Validation time: " + String.format("%.3f", validationTime / Math.pow(10, 9)) + "s" : ""));
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
     * Sets reset flag for procedure expression dependencies.
     *
     * @param resetDependencies if true procedure expression dependencies are reset otherwise false.
     */
    public void resetDependencies(boolean resetDependencies) {
        waitToComplete();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers.values()) neuralNetworkLayer.resetDependencies(resetDependencies);
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
     * Returns training metrics instance.
     *
     * @return training metrics instance.
     */
    public TreeMap<Integer, SingleRegressionMetric> getTrainingMetrics() {
        waitToComplete();
        return trainingMetrics;
    }

    /**
     * Sets if training metrics is shown.
     *
     * @param showTrainingMetrics if true training metrics is shown otherwise not.
     * @throws NeuralNetworkException throws exception if parameter is attempted to be set when neural network is already started.
     */
    public void setShowTrainingMetrics(boolean showTrainingMetrics) throws NeuralNetworkException {
        if(isStarted()) throw new NeuralNetworkException("Training metrics can be only enabled / disabled when neural network is not started.");
        this.showTrainingMetrics = showTrainingMetrics;
    }

    /**
     * Returns total neural network training iterations count.
     *
     * @return total neural network training iterations count.
     */
    public int getTotalTrainingIterations() {
        waitToComplete();
        return totalTrainingIterations;
    }

    /**
     * Sets validation data.
     *
     * @param validationSampler validation sampler containing validation data set.
     * @throws NeuralNetworkException throws exception if setting of validation data fails.
     */
    public void setValidationData(Sampler validationSampler) throws NeuralNetworkException {
        waitToComplete();
        if (validationSampler == null) throw new NeuralNetworkException("Validation sampler is not set.");
        this.validationSampler = validationSampler;
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
        if (validationSampler != null) this.validationSampler = validationSampler;
        if (this.validationSampler == null) throw new NeuralNetworkException("Validation sampler is not set.");
        nextState(ExecutionState.VALIDATE);
        if (waitToComplete) waitToComplete();
    }

    /**
     * Validates with given test set inputs and outputs.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if validation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    private void validateInput() throws MatrixException, NeuralNetworkException, DynamicParamException {
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
        if (verboseValidation && (totalTrainingIterations % verboseCycle == 0)) verboseValidationStatus();
    }

    /**
     * Sets regression as validation metric.
     *
     * @param showMetric if true shows metric otherwise not.
     */
    public void setAsRegression(boolean showMetric) {
        setAsRegression(false, showMetric);
    }

    /**
     * Sets regression as validation metric.
     *
     * @param outputLayerIndex output layer index.
     * @param showMetric if true shows metric otherwise not.
     */
    public void setAsRegression(int outputLayerIndex, boolean showMetric) {
        setAsRegression(outputLayerIndex, false, showMetric);
    }

    /**
     * Sets regression as validation metric.
     *
     * @param useR2AsLastError if true uses R2 as last error otherwise uses MSE.
     * @param showMetric if true shows metric otherwise not.
     */
    public void setAsRegression(boolean useR2AsLastError, boolean showMetric) {
        waitToComplete();
        validationMetrics.clear();
        for (Integer outputLayerIndex : getOutputLayers().keySet()) setAsRegression(outputLayerIndex, useR2AsLastError, showMetric);
    }

    /**
     * Sets regression as validation metric.
     *
     * @param outputLayerIndex output layer index.
     * @param useR2AsLastError if true uses R2 as last error otherwise uses MSE.
     * @param showMetric if true shows metric otherwise not.
     */
    public void setAsRegression(int outputLayerIndex, boolean useR2AsLastError, boolean showMetric) {
        waitToComplete();
        setValidationMetric(outputLayerIndex, new RegressionMetric(useR2AsLastError, showMetric));
    }

    /**
     * Sets classification as validation metric.
     *
     * @param showMetric if true shows metric otherwise not.
     */
    public void setAsClassification(boolean showMetric) {
        setAsClassification(false, showMetric);
    }

    /**
     * Sets classification as validation metric.
     *
     * @param outputLayerIndex output layer index.
     * @param showMetric if true shows metric otherwise not.
     */
    public void setAsClassification(int outputLayerIndex, boolean showMetric) {
        setAsClassification(outputLayerIndex, false, showMetric);
    }

    /**
     * Sets regression as validation metric.
     *
     * @param multiClass if true metrics assumes multi class classification otherwise single class classification.
     * @param showMetric if true shows metric otherwise not.
     */
    public void setAsClassification(boolean multiClass, boolean showMetric) {
        waitToComplete();
        validationMetrics.clear();
        for (Integer outputLayerIndex : getOutputLayers().keySet()) setAsClassification(outputLayerIndex, multiClass, showMetric);
    }

    /**
     * Sets regression as validation metric.
     *
     * @param outputLayerIndex output layer index.
     * @param multiClass if true metrics assumes multi class classification otherwise single class classification.
     * @param showMetric if true shows metric otherwise not.
     */
    public void setAsClassification(int outputLayerIndex, boolean multiClass, boolean showMetric) {
        waitToComplete();
        setValidationMetric(outputLayerIndex, new ClassificationMetric(multiClass, showMetric));
    }

    /**
     * Sets validation metric.
     *
     * @param outputLayerIndex output layer index.
     * @param metric metric.
     */
    private void setValidationMetric(int outputLayerIndex, Metric metric) {
        validationMetrics.put(outputLayerIndex, metric);
        if (earlyStoppingMap.get(outputLayerIndex) != null) earlyStoppingMap.get(outputLayerIndex).setValidationMetric(metric);
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
     * Sets if confusion matrix is printed along other classification metrics.
     *
     * @param printConfusionMatrix if true confusion matrix is printed along other classification metrics.
     */
    public void printConfusionMatrix(boolean printConfusionMatrix) {
        waitToComplete();
        for (Metric validationMetric : validationMetrics.values()) if (validationMetric instanceof ClassificationMetric classificationMetric) {
            classificationMetric.setPrintConfusionMatrix(printConfusionMatrix);
        }
    }

    /**
     * Sets if confusion matrix is shown along other classification metrics.
     *
     * @param showConfusionMatrix if true confusion matrix is shown along other classification metrics.
     */
    public void showConfusionMatrix(boolean showConfusionMatrix) {
        waitToComplete();
        for (Metric validationMetric : validationMetrics.values()) if (validationMetric instanceof ClassificationMetric classificationMetric) {
            classificationMetric.setShowConfusionMatrix(showConfusionMatrix);
        }
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
    public TreeMap<Integer, Sequence> predict(TreeMap<Integer, Sequence> inputs) throws NeuralNetworkException {
        return predict(inputs, true);
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
        predictInputs = inputs;
        nextState(ExecutionState.PREDICT);
        return waitToComplete ? getOutput() : null;
    }

    /**
     * Predicts using given test set inputs.
     *
     */
    private void predictInput() {
        for (Map.Entry<Integer, InputLayer> entry : getInputLayers().entrySet()) entry.getValue().predict(predictInputs.get(entry.getKey()));
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
        predictInputs = new TreeMap<>();
        trainingSampler = null;
        validationSampler = null;
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers.values()) neuralNetworkLayer.reinitialize();
        for (Map.Entry<Integer, EarlyStopping> entry : earlyStoppingMap.entrySet()) earlyStoppingMap.put(entry.getKey(), entry.getValue().reference());
        for (Metric validationMetric : validationMetrics.values()) validationMetric.reinitialize();
        totalTrainingIterations = 0;
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
        for (Map.Entry<Integer, NeuralNetworkLayer> entry : neuralNetworkLayers.entrySet()) {
            entry.getValue().append(otherNeuralNetwork.getNeuralNetworkLayers().get(entry.getKey()), tau);
        }
    }


    /**
     * Prints structure and metadata of neural network.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    public void print() throws NeuralNetworkException {
        checkNotStarted();
        System.out.println("Number of layers: " + neuralNetworkLayers.size() + " [ Input layers: " + inputLayers.size() + ", Hidden layers: " + hiddenLayers.size() + ", Output layers: " + outputLayers.size() + " ]");
        int totalNumberOfParameters = 0;
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers.values()) {
            totalNumberOfParameters += neuralNetworkLayer.getNumberOfParameters();
        }
        System.out.println("Total number of parameters: " + totalNumberOfParameters);
        System.out.println("Apply early stopping: " + (!earlyStoppingMap.isEmpty() ? "Yes" : "No"));
        System.out.println();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers.values()) {
            neuralNetworkLayer.print();
            System.out.println();
        }
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

