/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.network;

import java.io.*;
import java.util.*;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import core.activation.ActivationFunction;
import core.layer.*;
import core.loss.LossFunction;
import core.normalization.NormalizationType;
import core.optimization.OptimizationType;
import core.optimization.OptimizerFactory;
import core.regularization.*;
import core.metrics.*;
import utils.*;
import utils.matrix.*;
import utils.sampling.Sampler;

/**
 * Defines main class for neural network.<br>
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
     * Lock condition for synchronizing execution procedures (train, predict, validate).
     *
     */
    private transient Condition executeLockCondition;

    /**
     * Lock condition for synchronizing completion of procedure execution and shift to idle state.
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
    private InputLayer inputLayer;

    /**
     * List containing hidden layers for neural network.
     *
     */
    private final ArrayList<AbstractLayer> hiddenLayers = new ArrayList<>();

    /**
     * Reference to output layer of neural network.
     *
     */
    private OutputLayer outputLayer;

    /**
     * List of neural network layers.
     *
     */
    private final ArrayList<NeuralNetworkLayer> neuralNetworkLayers = new ArrayList<>();

    /**
     * Reference to early stopping condition.
     *
     */
    private EarlyStopping earlyStopping;

    /**
     * Reference to training error metric.
     *
     */
    private transient Metrics trainingMetrics;

    /**
     * Reference to validation error metric.
     *
     */
    private transient Metrics validationMetrics;

    /**
     * Validation error metric type. Default REGRESSION.
     *
     */
    private MetricsType validationMetricsType = MetricsType.REGRESSION;

    /**
     * Structure containing prediction input sequence.
     *
     */
    private transient Sequence predictInputs;

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
     * Total training time of neural network in nano seconds.
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
     * Constructor for neural network.
     *
     */
    public NeuralNetwork() {
    }

    /**
     * Constructor for neural network.
     *
     * @param neuralNetworkName name for neural network instance.
     */
    public NeuralNetwork(String neuralNetworkName) {
        this.neuralNetworkName = neuralNetworkName;
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
     * Adds regularizer to specific neural network layer.
     *
     * @param neuralNetworkLayerIndex neural network layer. Input layer has index 0.
     * @param regularizationType type of regularizer.
     * @throws NeuralNetworkException throws neural network exception if adding of regularizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addRegularizer(int neuralNetworkLayerIndex, RegularizationType regularizationType) throws NeuralNetworkException, DynamicParamException {
        addRegularizer(neuralNetworkLayerIndex, regularizationType, null);
    }

    /**
     * Adds regularizer to all neural network layers.
     *
     * @param regularizationType type of regularizer.
     * @throws NeuralNetworkException throws neural network exception if adding of regularizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addRegularizer(RegularizationType regularizationType) throws NeuralNetworkException, DynamicParamException {
        addRegularizer(regularizationType, null);
    }

    /**
     * Adds regularizer to specific neural network layer.
     *
     * @param neuralNetworkLayerIndex neural network layer. Input layer has index 0.
     * @param regularizationType type of regularizer.
     * @param params parameters for regularizer.
     * @throws NeuralNetworkException throws neural network exception if adding of regularizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addRegularizer(int neuralNetworkLayerIndex, RegularizationType regularizationType, String params) throws NeuralNetworkException, DynamicParamException {
        checkStarted();
        if (neuralNetworkLayerIndex < 0 || neuralNetworkLayerIndex > neuralNetworkLayers.size() - 1) throw new NeuralNetworkException("No neural network layer index: " + neuralNetworkLayerIndex + " exists.");
        NeuralNetworkLayer neuralNetworkLayer = neuralNetworkLayers.get(neuralNetworkLayerIndex);
        if (neuralNetworkLayer instanceof InputLayer) throw new NeuralNetworkException("Input layer cannot be regularized.");
        if (neuralNetworkLayer instanceof OutputLayer) throw new NeuralNetworkException("Output layer cannot be regularized.");
        neuralNetworkLayer.addRegularization(regularizationType, params);
    }

    /**
     * Adds regularizer to all neural network layers.
     *
     * @param regularizationType type of regularizer.
     * @param params parameters for regularizer.
     * @throws NeuralNetworkException throws neural network exception if adding of regularizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addRegularizer(RegularizationType regularizationType, String params) throws NeuralNetworkException, DynamicParamException {
        checkStarted();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers) {
            if (!(neuralNetworkLayer instanceof InputLayer) && !(neuralNetworkLayer instanceof OutputLayer)) neuralNetworkLayer.addRegularization(regularizationType, params);
        }
    }

    /**
     * Removes regularizer from specific neural network layer.
     *
     * @param neuralNetworkLayerIndex neural network layer. Input layer has index 0.
     * @param regularizationType type of regularizer.
     * @throws NeuralNetworkException throws neural network exception if adding of regularizer fails.
     */
    public void removeRegularizer(int neuralNetworkLayerIndex, RegularizationType regularizationType) throws NeuralNetworkException {
        checkStarted();
        if (neuralNetworkLayerIndex < 0 || neuralNetworkLayerIndex > neuralNetworkLayers.size() - 1) throw new NeuralNetworkException("No neural network layer index: " + neuralNetworkLayerIndex + " exists.");
        NeuralNetworkLayer neuralNetworkLayer = neuralNetworkLayers.get(neuralNetworkLayerIndex);
        if (neuralNetworkLayer instanceof InputLayer) throw new NeuralNetworkException("Input layer cannot be regularized.");
        if (neuralNetworkLayer instanceof OutputLayer) throw new NeuralNetworkException("Output layer cannot be regularized.");
        neuralNetworkLayer.removeRegularization(regularizationType);
    }

    /**
     * Removes regularizer from all neural network layers.
     *
     * @param regularizationType type of regularizer.
     * @throws NeuralNetworkException throws neural network exception if removal of regularizer fails.
     */
    public void removeRegularizer(RegularizationType regularizationType) throws NeuralNetworkException {
        checkStarted();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers) {
            if (!(neuralNetworkLayer instanceof InputLayer) && !(neuralNetworkLayer instanceof OutputLayer)) neuralNetworkLayer.removeRegularization(regularizationType);
        }
    }

    /**
     * Removes regularizer from specific neural network layer.
     *
     * @param neuralNetworkLayerIndex neural network layer. Input layer has index 0.
     * @throws NeuralNetworkException throws neural network exception if removal of regularizer fails.
     */
    public void removeRegularization(int neuralNetworkLayerIndex) throws NeuralNetworkException {
        checkStarted();
        if (neuralNetworkLayerIndex < 0 || neuralNetworkLayerIndex > neuralNetworkLayers.size() - 1) throw new NeuralNetworkException("No neural network layer index: " + neuralNetworkLayerIndex + " exists.");
        NeuralNetworkLayer neuralNetworkLayer = neuralNetworkLayers.get(neuralNetworkLayerIndex);
        if (neuralNetworkLayer instanceof InputLayer) throw new NeuralNetworkException("Input layer cannot be regularized.");
        if (neuralNetworkLayer instanceof OutputLayer) throw new NeuralNetworkException("Output layer cannot be regularized.");
        neuralNetworkLayer.removeRegularization();
    }

    /**
     * Removes all regularizers from neural network (layers).
     *
     * @throws NeuralNetworkException throws neural network exception if removal of regularizers fails.
     */
    public void removeRegularization() throws NeuralNetworkException {
        checkStarted();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers) {
            if (!(neuralNetworkLayer instanceof InputLayer) && !(neuralNetworkLayer instanceof OutputLayer)) neuralNetworkLayer.removeRegularization();
        }
    }

    /**
     * Adds normalizer to specific neural network layer.
     *
     * @param neuralNetworkLayerIndex neural network layer. Input layer has index 0.
     * @param normalizationType type of normalizer.
     * @throws NeuralNetworkException throws neural network exception if adding of normalizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addNormalizer(int neuralNetworkLayerIndex, NormalizationType normalizationType) throws NeuralNetworkException, DynamicParamException {
        addNormalizer(neuralNetworkLayerIndex, normalizationType, null);
    }

    /**
     * Adds normalizer to all neural network layers.
     *
     * @param normalizationType type of normalizer.
     * @throws NeuralNetworkException throws neural network exception if adding of normalizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addNormalizer(NormalizationType normalizationType) throws NeuralNetworkException, DynamicParamException {
        addNormalizer(normalizationType, null);
    }

    /**
     * Adds normalizer to specific neural network layer.
     *
     * @param neuralNetworkLayerIndex neural network layer. Input layer has index 0.
     * @param normalizationType type of normalizer.
     * @param params parameters for normalizer.
     * @throws NeuralNetworkException throws neural network exception if adding of normalizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addNormalizer(int neuralNetworkLayerIndex, NormalizationType normalizationType, String params) throws NeuralNetworkException, DynamicParamException {
        checkStarted();
        if (neuralNetworkLayerIndex < 0 || neuralNetworkLayerIndex > neuralNetworkLayers.size() - 1) throw new NeuralNetworkException("No neural network layer index: " + neuralNetworkLayerIndex + " exists.");
        NeuralNetworkLayer neuralNetworkLayer = neuralNetworkLayers.get(neuralNetworkLayerIndex);
        if (neuralNetworkLayer instanceof InputLayer) throw new NeuralNetworkException("Input layer cannot be normalized.");
        if (neuralNetworkLayer instanceof OutputLayer) throw new NeuralNetworkException("Output layer cannot be normalized.");
        neuralNetworkLayer.addNormalization(normalizationType, params);
    }

    /**
     * Adds normalizer to all neural network layers.
     *
     * @param normalizationType type of normalizer.
     * @param params parameters for normalizer.
     * @throws NeuralNetworkException throws neural network exception if adding of normalizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addNormalizer(NormalizationType normalizationType, String params) throws NeuralNetworkException, DynamicParamException {
        checkStarted();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers) {
            if (!(neuralNetworkLayer instanceof InputLayer) && !(neuralNetworkLayer instanceof OutputLayer)) neuralNetworkLayer.addNormalization(normalizationType, params);
        }
    }

    /**
     * Removes normalizer from specific neural network layer.
     *
     * @param neuralNetworkLayerIndex neural network layer. Input layer has index 0.
     * @param normalizationType type of normalizer.
     * @throws NeuralNetworkException throws neural network exception if adding of normalizer fails.
     */
    public void removeNormalizer(int neuralNetworkLayerIndex, NormalizationType normalizationType) throws NeuralNetworkException {
        checkStarted();
        if (neuralNetworkLayerIndex < 0 || neuralNetworkLayerIndex > neuralNetworkLayers.size() - 1) throw new NeuralNetworkException("No neural network layer index: " + neuralNetworkLayerIndex + " exists.");
        NeuralNetworkLayer neuralNetworkLayer = neuralNetworkLayers.get(neuralNetworkLayerIndex);
        if (neuralNetworkLayer instanceof InputLayer) throw new NeuralNetworkException("Input layer cannot be normalized.");
        if (neuralNetworkLayer instanceof OutputLayer) throw new NeuralNetworkException("Output layer cannot be normalized.");
        neuralNetworkLayer.removeNormalization(normalizationType);
    }

    /**
     * Removes normalizer from all neural network layers.
     *
     * @param normalizationType type of normalizer.
     * @throws NeuralNetworkException throws neural network exception if removal of normalizer fails.
     */
    public void removeNormalizer(NormalizationType normalizationType) throws NeuralNetworkException {
        checkStarted();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers) {
            if (!(neuralNetworkLayer instanceof InputLayer) && !(neuralNetworkLayer instanceof OutputLayer)) neuralNetworkLayer.removeNormalization(normalizationType);
        }
    }

    /**
     * Removes normalizer from specific neural network layer.
     *
     * @param neuralNetworkLayerIndex neural network layer. Input layer has index 0.
     * @throws NeuralNetworkException throws neural network exception if removal of normalizer fails.
     */
    public void removeNormalization(int neuralNetworkLayerIndex) throws NeuralNetworkException {
        checkStarted();
        if (neuralNetworkLayerIndex < 0 || neuralNetworkLayerIndex > neuralNetworkLayers.size() - 1) throw new NeuralNetworkException("No neural network layer index: " + neuralNetworkLayerIndex + " exists.");
        NeuralNetworkLayer neuralNetworkLayer = neuralNetworkLayers.get(neuralNetworkLayerIndex);
        if (neuralNetworkLayer instanceof InputLayer) throw new NeuralNetworkException("Input layer cannot be normalized.");
        if (neuralNetworkLayer instanceof OutputLayer) throw new NeuralNetworkException("Output layer cannot be normalized.");
        neuralNetworkLayer.removeNormalization();
    }

    /**
     * Removes all normalizers from neural network (layers).
     *
     * @throws NeuralNetworkException throws neural network exception if removal of normalizers fails.
     */
    public void removeNormalization() throws NeuralNetworkException {
        checkStarted();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers) {
            if (!(neuralNetworkLayer instanceof InputLayer) && !(neuralNetworkLayer instanceof OutputLayer)) neuralNetworkLayer.removeNormalization();
        }
    }

    /**
     * Resets normalization of all neural network layers.
     *
     * @throws NeuralNetworkException throws neural network exception if resetting of normalization fails.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public void resetNormalization() throws NeuralNetworkException, MatrixException {
        checkStarted();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers) {
            if (!(neuralNetworkLayer instanceof InputLayer) && !(neuralNetworkLayer instanceof OutputLayer)) neuralNetworkLayer.resetNormalization();
        }
    }

    /**
     * Resets normalization of specific type for all neural network layers.
     *
     * @param normalizationType normalization method to be reset.
     * @throws NeuralNetworkException throws neural network exception if resetting of normalization fails.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public void resetNormalization(NormalizationType normalizationType) throws NeuralNetworkException, MatrixException {
        checkStarted();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers) {
            if (!(neuralNetworkLayer instanceof InputLayer) && !(neuralNetworkLayer instanceof OutputLayer)) neuralNetworkLayer.resetNormalization(normalizationType);
        }
    }

    /**
     * Resets normalization of specific neural network layer.
     *
     * @param neuralNetworkLayerIndex neural network layer. Input layer has index 0.
     * @throws NeuralNetworkException throws neural network exception if resetting of normalization fails.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public void resetNormalization(int neuralNetworkLayerIndex) throws NeuralNetworkException, MatrixException {
        checkStarted();
        if (neuralNetworkLayerIndex < 0 || neuralNetworkLayerIndex > neuralNetworkLayers.size() - 1) throw new NeuralNetworkException("No neural network layer index: " + neuralNetworkLayerIndex + " exists.");
        NeuralNetworkLayer neuralNetworkLayer = neuralNetworkLayers.get(neuralNetworkLayerIndex);
        if (neuralNetworkLayer instanceof InputLayer) throw new NeuralNetworkException("Input layer cannot be normalized.");
        if (neuralNetworkLayer instanceof OutputLayer) throw new NeuralNetworkException("Output layer cannot be normalized.");
        neuralNetworkLayer.resetNormalization();
    }

    /**
     * Resets normalization of specific type for specific neural network layer.
     *
     * @param neuralNetworkLayerIndex neural network layer. Input layer has index 0.
     * @param normalizationType normalization method to be reset.
     * @throws NeuralNetworkException throws neural network exception if resetting of normalization fails.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public void resetNormalization(int neuralNetworkLayerIndex, NormalizationType normalizationType) throws NeuralNetworkException, MatrixException {
        checkStarted();
        if (neuralNetworkLayerIndex < 0 || neuralNetworkLayerIndex > neuralNetworkLayers.size() - 1) throw new NeuralNetworkException("No neural network layer index: " + neuralNetworkLayerIndex + " exists.");
        NeuralNetworkLayer neuralNetworkLayer = neuralNetworkLayers.get(neuralNetworkLayerIndex);
        if (neuralNetworkLayer instanceof InputLayer) throw new NeuralNetworkException("Input layer cannot be normalized.");
        if (neuralNetworkLayer instanceof OutputLayer) throw new NeuralNetworkException("Output layer cannot be normalized.");
        neuralNetworkLayer.resetNormalization(normalizationType);
    }

    /**
     * Sets optimizer to specific neural network layer.
     *
     * @param neuralNetworkLayerIndex neural network layer. Input layer has index 0.
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
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers) {
            if (!(neuralNetworkLayer instanceof InputLayer) && !(neuralNetworkLayer instanceof OutputLayer)) neuralNetworkLayer.setOptimizer(OptimizerFactory.create(optimization, params));
        }
    }

    /**
     * Sets optimizer to specific neural network layer.
     *
     * @param neuralNetworkLayerIndex neural network layer. Input layer has index 0.
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
     * Resets optimizer of all neural network layers.
     *
     * @throws NeuralNetworkException throws neural network exception if resetting of optimizer fails.
     */
    public void resetOptimizer() throws NeuralNetworkException {
        checkStarted();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers) {
            if (!(neuralNetworkLayer instanceof InputLayer) && !(neuralNetworkLayer instanceof OutputLayer)) neuralNetworkLayer.resetOptimizer();
        }
    }

    /**
     * Resets optimizer of specific neural network layer.
     *
     * @param neuralNetworkLayerIndex neural network layer. Input layer has index 0.
     * @throws NeuralNetworkException throws neural network exception if resetting of optimizer fails.
     */
    public void resetOptimizer(int neuralNetworkLayerIndex) throws NeuralNetworkException {
        checkStarted();
        if (neuralNetworkLayerIndex < 0 || neuralNetworkLayerIndex > neuralNetworkLayers.size() - 1) throw new NeuralNetworkException("No neural network layer index: " + neuralNetworkLayerIndex + " exists.");
        NeuralNetworkLayer neuralNetworkLayer = neuralNetworkLayers.get(neuralNetworkLayerIndex);
        if (neuralNetworkLayer instanceof InputLayer) throw new NeuralNetworkException("Input layer cannot be optimized.");
        if (neuralNetworkLayer instanceof OutputLayer) throw new NeuralNetworkException("Output layer cannot be optimized.");
        neuralNetworkLayer.resetOptimizer();
    }

    /**
     * Adds input layer to neural network.
     *
     * @param params parameters for input layer.
     * @throws NeuralNetworkException throws neural network exception if adding of input layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addInputLayer(String params) throws NeuralNetworkException, DynamicParamException {
        checkStarted();
        inputLayer = new InputLayer(0, params);
    }

    /**
     * Returns inputs layer.
     *
     * @return input layer.
     */
    public InputLayer getInputLayer() {
        return inputLayer;
    }

    /**
     * Adds hidden layer to neural network. Layers are executed in order which they are added.
     *
     * @param layerType type of layer.
     * @throws NeuralNetworkException throws neural network exception if adding of layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public void addHiddenLayer(LayerType layerType) throws NeuralNetworkException, DynamicParamException, MatrixException {
        addHiddenLayer(layerType, null, null, null);
    }

    /**
     * Adds hidden layer to neural network. Layers are executed in order which they are added.
     *
     * @param layerType type of layer.
     * @param params parameters for layer.
     * @throws NeuralNetworkException throws neural network exception if adding of layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public void addHiddenLayer(LayerType layerType, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        addHiddenLayer(layerType, null, null, params);
    }

    /**
     * Adds hidden layer to neural network. Layers are executed in order which they are added.
     *
     * @param layerType type of layer.
     * @param activationFunction activation function for layer.
     * @throws NeuralNetworkException throws neural network exception if adding of layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public void addHiddenLayer(LayerType layerType, ActivationFunction activationFunction) throws NeuralNetworkException, DynamicParamException, MatrixException {
        addHiddenLayer(layerType, activationFunction, null, null);
    }

    /**
     * Adds hidden layer to neural network. Layers are executed in order which they are added.
     *
     * @param layerType type of layer.
     * @param activationFunction activation function for layer.
     * @param params parameters for layer.
     * @throws NeuralNetworkException throws neural network exception if adding of layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public void addHiddenLayer(LayerType layerType, ActivationFunction activationFunction, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        addHiddenLayer(layerType, activationFunction, null, params);
    }

    /**
     * Adds hidden layer to neural network. Layers are executed in order which they are added.
     *
     * @param layerType type of layer.
     * @param activationFunction activation function for layer.
     * @param initialization layer parameter initialization function for layer.
     * @throws NeuralNetworkException throws neural network exception if adding of layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public void addHiddenLayer(LayerType layerType, ActivationFunction activationFunction, Initialization initialization) throws NeuralNetworkException, DynamicParamException, MatrixException {
        addHiddenLayer(layerType, activationFunction, initialization, null);
    }

    /**
     * Adds hidden layer to neural network. Layers are executed in order which they are added.
     *
     * @param layerType type of layer.
     * @param activationFunction activation function for layer.
     * @param initialization layer parameter initialization function for layer.
     * @param params parameters for layer.
     * @throws NeuralNetworkException throws neural network exception if adding of layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public void addHiddenLayer(LayerType layerType, ActivationFunction activationFunction, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        checkStarted();
        hiddenLayers.add(LayerFactory.create(hiddenLayers.size() + 1, layerType, activationFunction, initialization, params));
    }

    /**
     * Adds output layer to neural network.
     *
     * @param lossFunctionType loss function type for output layer.
     * @throws NeuralNetworkException throws neural network exception if adding of output layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public void addOutputLayer(BinaryFunctionType lossFunctionType) throws NeuralNetworkException, DynamicParamException, MatrixException {
        addOutputLayer(lossFunctionType, null);
    }

    /**
     * Adds output layer to neural network.
     *
     * @param lossFunctionType loss function type for output layer.
     * @param params parameters for loss function.
     * @throws NeuralNetworkException throws neural network exception if adding of output layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public void addOutputLayer(BinaryFunctionType lossFunctionType, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        checkStarted();
        outputLayer = params != null ? new OutputLayer(-1, new LossFunction(lossFunctionType, params)) : new OutputLayer(-1, new LossFunction(lossFunctionType));
    }

    /**
     * Adds output layer to neural network.
     *
     * @param lossFunctionTypes loss function types for output layer.
     * @throws NeuralNetworkException throws neural network exception if adding of output layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public void addOutputLayer(ArrayList<BinaryFunctionType> lossFunctionTypes) throws NeuralNetworkException, DynamicParamException, MatrixException {
        addOutputLayer(lossFunctionTypes, null);
    }

    /**
     * Adds output layer to neural network.
     *
     * @param lossFunctionTypes loss function types for output layer.
     * @param params parameters for loss function.
     * @throws NeuralNetworkException throws neural network exception if adding of output layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public void addOutputLayer(ArrayList<BinaryFunctionType> lossFunctionTypes, ArrayList<String> params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        checkStarted();
        ArrayList<LossFunction> lossFunctions = new ArrayList<>();
        for (int index = 0; index < lossFunctionTypes.size(); index++) {
            lossFunctions.add(params != null ? new LossFunction(lossFunctionTypes.get(index), params.get(index)) : new LossFunction(lossFunctionTypes.get(index)));
        }
        outputLayer = new OutputLayer(-1, lossFunctions);
    }

    /**
     * Adds output layer to neural network.
     *
     * @param lossFunctionTypes loss function types for output layer.
     * @throws NeuralNetworkException throws neural network exception if adding of output layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public void addOutputLayer(BinaryFunctionType[] lossFunctionTypes) throws NeuralNetworkException, DynamicParamException, MatrixException {
        addOutputLayer(lossFunctionTypes, null);
    }

    /**
     * Adds output layer to neural network.
     *
     * @param lossFunctionTypes loss function types for output layer.
     * @param params parameters for loss function.
     * @throws NeuralNetworkException throws neural network exception if adding of output layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public void addOutputLayer(BinaryFunctionType[] lossFunctionTypes, String[] params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        checkStarted();
        ArrayList<LossFunction> lossFunctions = new ArrayList<>();
        for (int index = 0; index < lossFunctionTypes.length; index++) {
            lossFunctions.add(params != null ? new LossFunction(lossFunctionTypes[index], params[index]) : new LossFunction(lossFunctionTypes[index]));
        }
        outputLayer = new OutputLayer(-1, lossFunctions);
    }

    /**
     * Returns output layer of neural network.
     *
     * @return output layer of neural network.
     */
    public OutputLayer getOutputLayer() {
        return outputLayer;
    }

    /**
     * Returns list of neural network layers.
     *
     * @return list of neural network layers.
     */
    public ArrayList<NeuralNetworkLayer> getNeuralNetworkLayers() {
        return neuralNetworkLayers;
    }

    /**
     * Builds neural network.<br>
     * Connects layers to each other.<br>
     * Initializes layers.<br>
     *
     * @throws NeuralNetworkException throws neural network exception if neural network is already built.
     */
    public void build() throws NeuralNetworkException {
        if (!neuralNetworkLayers.isEmpty()) throw new NeuralNetworkException("Neural network is already built.");

        neuralNetworkLayers.add(inputLayer);
        neuralNetworkLayers.addAll(hiddenLayers);
        neuralNetworkLayers.add(outputLayer);

        for (int layerIndex = 0; layerIndex < neuralNetworkLayers.size() - 1; layerIndex++) {
            neuralNetworkLayers.get(layerIndex).setNextLayer(neuralNetworkLayers.get(layerIndex + 1));
            neuralNetworkLayers.get(layerIndex + 1).setPreviousLayer(neuralNetworkLayers.get(layerIndex));
        }

        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers) neuralNetworkLayer.initialize();
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

        trainingMetrics = new Metrics(MetricsType.REGRESSION);
        if (validationMetrics == null) validationMetrics = new Metrics(validationMetricsType);

        if (earlyStopping != null) {
            earlyStopping.setTrainingError(trainingMetrics);
            earlyStopping.setValidationError(validationMetrics);
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

        getInputLayer().start();

    }

    /**
     * Stops neural network.
     *
     */
    public void stop() {
        if (!isStarted()) return;
        waitToComplete();
        executeLock.lock();
        getInputLayer().stop();
        executionState = ExecutionState.TERMINATED;
        executeLockCondition.signal();
        neuralNetworkThread = null;
        executeLock.unlock();
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
     * Sets if recurrent inputs of layer are allowed to be reset during training.
     *
     * @param resetStateTraining if true recurrent inputs are allowed to be reset.
     * @throws NeuralNetworkException throws exception if neural network is not started.
     */
    public void setResetStateTraining(boolean resetStateTraining) throws NeuralNetworkException {
        checkNotStarted();
        waitToComplete();
        getInputLayer().setResetStateTraining(resetStateTraining);
    }

    /**
     * Sets if recurrent inputs of layer are allowed to be reset during testing.<br>
     * Useful when long input sequence is applied over multiple inferences (prediction) requests.<br>
     *
     * @param resetStateTesting if true recurrent inputs are allowed to be reset.
     * @throws NeuralNetworkException throws exception if neural network is not started.
     */
    public void setResetStateTesting(boolean resetStateTesting) throws NeuralNetworkException {
        checkNotStarted();
        waitToComplete();
        getInputLayer().setResetStateTesting(resetStateTesting);
    }

    /**
     * Sets if recurrent inputs of layer are allowed to be restored during training.
     *
     * @param restoreStateTraining if true recurrent inputs are allowed to be restored.
     * @throws NeuralNetworkException throws exception if neural network is not started.
     */
    public void setRestoreStateTraining(boolean restoreStateTraining) throws NeuralNetworkException {
        checkNotStarted();
        waitToComplete();
        getInputLayer().setRestoreStateTraining(restoreStateTraining);
    }

    /**
     * Sets if recurrent inputs of layer are allowed to be restored during testing.<br>
     * Useful when long input sequence is applied over multiple inferences (prediction) requests.<br>
     *
     * @param restoreStateTesting if true recurrent inputs are allowed to be restored.
     * @throws NeuralNetworkException throws exception if neural network is not started.
     */
    public void setRestoreStateTesting(boolean restoreStateTesting) throws NeuralNetworkException {
        checkNotStarted();
        waitToComplete();
        getInputLayer().setRestoreStateTesting(restoreStateTesting);
    }

    /**
     * Sets training sample sets of neural network via sampler.<br>
     * Equal indices of input and output reflect input output pair of neural network.<br>
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
     * Sets early stopping condition.
     *
     * @param earlyStopping early stopping instance.
     * @throws NeuralNetworkException throws exception if early stopping is not defined.
     */
    public void setTrainingEarlyStopping(EarlyStopping earlyStopping) throws NeuralNetworkException {
        waitToComplete();
        if (earlyStopping == null) throw new NeuralNetworkException("Early stopping is not defined.");
        this.earlyStopping = earlyStopping;
        earlyStopping.setTrainingError(trainingMetrics);
        earlyStopping.setValidationError(validationMetrics);
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
     * Returns neural network training time in milli seconds.
     *
     * @return neural network training time in milli seconds.
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
     * Predicts values based on current test set inputs.
     *
     * @return predicted values (neural network output).
     * @throws NeuralNetworkException throws exception if prediction fails.
     */
    public Sequence predict() throws NeuralNetworkException {
        return predict(null, true);
    }

    /**
     * Predicts values based on given input.
     *
     * @param input input for prediction.
     * @return predicted value (neural network output).
     * @throws NeuralNetworkException throws exception if prediction fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     */
    public Matrix predict(Matrix input) throws NeuralNetworkException, MatrixException {
        return predict(new MMatrix(input)).get(0);
    }

    /**
     * Predicts values based on given input.
     *
     * @param input input for prediction.
     * @return predicted value (neural network output).
     * @throws NeuralNetworkException throws exception if prediction fails.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     */
    public MMatrix predict(MMatrix input) throws NeuralNetworkException, MatrixException {
        if (input == null) throw new NeuralNetworkException("No prediction inputs set");
        Sequence inputs = new Sequence(input.getCapacity());
        inputs.put(0, input);
        return predict(inputs, true).get(0);
    }

    /**
     * Predicts values based on current test set inputs.<br>
     * Sets specific inputs for prediction.<br>
     *
     * @param inputs test input set for prediction.
     * @return predicted values (neural network output).
     * @throws NeuralNetworkException throws exception if prediction fails.
     */
    public Sequence predict(Sequence inputs) throws NeuralNetworkException {
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
    public Sequence predict(boolean waitToComplete) throws NeuralNetworkException {
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
    public Sequence predict(Sequence inputs, boolean waitToComplete) throws NeuralNetworkException {
        checkNotStarted();
        waitToComplete();
        executeLock.lock();
        if (inputs == null) {
            executeLock.unlock();
            throw new NeuralNetworkException("No prediction inputs set");
        }
        predictInputs = inputs;
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
    public Sequence getOutput() {
        waitToComplete();
        return getInputLayer().getOutput();
    }

    /**
     * Thread run function.<br>
     * Executes given neural network procedures and synchronizes their execution via neural network thread execution lock.<br>
     *
     */
    public void run() {
        while (true) {
            executeLock.lock();
            if (executionState == ExecutionState.IDLE) executeLockCondition.awaitUninterruptibly();
            try {
                switch (executionState) {
                    case TRAIN -> trainIterations();
                    case PREDICT -> predictInput();
                    case VALIDATE -> validateInput(true);
                    case TERMINATED -> {
                        getInputLayer().stop();
                        neuralNetworkThread = null;
                        completeLockCondition.signal();
                        executeLock.unlock();
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
        trainingMetrics.resetError();
        trainingSampler.reset();
        for (int iteration = 0; iteration < trainingSampler.getNumberOfIterations(); iteration++) {
            trainIteration();
            if (stoppedExecution()) break;
            if (earlyStopping != null) if (earlyStopping.stopTraining()) break;
        }
        trainingMetrics.store(totalIterations);
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
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers) if (reset) neuralNetworkLayer.reset();
        Sequence inputSequence = new Sequence(trainingSampler.getDepth());
        Sequence outputSequence = new Sequence(trainingSampler.getDepth());
        trainingSampler.getSamples(inputSequence, outputSequence);
        getOutputLayer().setTargets(outputSequence);
        getInputLayer().train(inputSequence);
        getOutputLayer().backward();
        getInputLayer().update();
        long endTime = System.nanoTime();
        trainingTime += endTime - startTime;
        trainingMetrics.report(getOutputLayer().getTotalError());
        trainingMetrics.store(totalIterations, true);
        totalIterations++;
        if (autoValidationCycle > 0) {
            autoValidationCount++;
            if (autoValidationCount >= autoValidationCycle) {
                validateInput(false);
                if (earlyStopping != null) earlyStopping.evaluateValidationCondition(totalIterations);
                autoValidationCount = 0;
            }
        }
        if (earlyStopping != null) earlyStopping.evaluateTrainingCondition(totalIterations);
        if (verboseTraining) verboseTrainingStatus();
        if (persistence != null) persistence.cycle();
    }

    /**
     * Verboses (prints to console) neural network training status.<br>
     * Prints number of iteration, neural network training time and training error.<br>
     *
     */
    private void verboseTrainingStatus() {
        if (totalIterations % verboseCycle == 0) System.out.println((neuralNetworkName != null ? neuralNetworkName + ": " : "") + "Training error (iteration #" + totalIterations +"): " + String.format("%.4f", trainingMetrics.getAbsolute()) + ", Training time in seconds: " + (trainingTime / 1000000000));
    }

    /**
     * Predicts using given test set inputs.
     *
     */
    private void predictInput() {
        getInputLayer().predict(predictInputs);
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
        validationMetrics.resetError();
        validationSampler.reset();
        for (int sampleIndex = 0; sampleIndex < validationSampler.getNumberOfIterations(); sampleIndex++) {
            Sequence inputSequence = new Sequence(validationSampler.getDepth());
            Sequence outputSequence = new Sequence(validationSampler.getDepth());
            validationSampler.getSamples(inputSequence, outputSequence);
            validationMetrics.report(getInputLayer().predict(inputSequence), outputSequence);
        }
        if (verboseValidation && (totalIterations % verboseCycle == 0)) verboseValidationStatus();
        validationMetrics.store(totalIterations, true);
        if (stateCompleted) stateCompleted();
    }

    /**
     * Returns training metrics instance.
     *
     * @return training metrics instance.
     */
    public Metrics getTrainingMetrics() {
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
    public double getOutputError() throws MatrixException, DynamicParamException {
        waitToComplete();
        return getOutputLayer().getTotalError();
    }

    /**
     * Sets neural network task (metrics) type (CLASSIFICATION or REGRESSION).
     *
     * @param metricsType neural network metrics type.
     * @param multiClass if true metrics assumes multi class classification otherwise single class classification.
     * @throws NeuralNetworkException throws exception if task type setting fails.
     */
    public void setTaskType(MetricsType metricsType, boolean multiClass) throws NeuralNetworkException {
        waitToComplete();
        validationMetricsType = metricsType;
        validationMetrics = new Metrics(metricsType, multiClass);
    }

    /**
     * Sets neural network task (metrics) type (CLASSIFICATION or REGRESSION).
     *
     * @param metricsType neural network metrics type.
     * @throws NeuralNetworkException throws exception if task type setting fails.
     */
    public void setTaskType(MetricsType metricsType) throws NeuralNetworkException {
        waitToComplete();
        validationMetricsType = metricsType;
        validationMetrics = new Metrics(metricsType);
    }

    /**
     * Returns validation metrics instance.
     *
     * @return validation metrics instance.
     */
    public Metrics getValidationMetrics() {
        waitToComplete();
        return validationMetrics;
    }

    /**
     * Verboses validation metrics.
     *
     * @throws NeuralNetworkException throws exception of printing of validation error fails.
     */
    private void verboseValidationStatus() throws NeuralNetworkException {
        System.out.println((neuralNetworkName != null ? neuralNetworkName + ": " : "") + "Validating...");
        validationMetrics.printReport();
    }

    /**
     * Makes deep copy of neural network by using object serialization.
     *
     * @return copy of this neural network.
     * @throws IOException throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     */
    public NeuralNetwork copy() throws IOException, ClassNotFoundException {
        waitToComplete();
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
     * Reinitializes neural network.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void reinitialize() throws MatrixException, NeuralNetworkException {
        waitToComplete();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers) neuralNetworkLayer.reinitialize();
    }

    /**
     * Appends other neural network to this neural network by weight tau. Effectively appends each weight and bias matrix of each layer by this weight factor.
     *
     * @param otherNeuralNetwork other neural network that contributes to this neural network.
     * @param tau tau which controls contribution of other layer.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void append(NeuralNetwork otherNeuralNetwork, double tau) throws MatrixException {
        waitToComplete();
        for (int index = 0; index < neuralNetworkLayers.size(); index++) {
            neuralNetworkLayers.get(index).append(otherNeuralNetwork.getNeuralNetworkLayers().get(index), tau);
        }
    }

    /**
     * Sets importance sampling weights to output layer.
     *
     * @param importanceSamplingWeights importance sampling weights
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    public void setImportanceSamplingWeights(TreeMap<Integer, Double> importanceSamplingWeights) throws NeuralNetworkException {
        checkNotStarted();
        waitToComplete();
        executeLock.lock();
        getOutputLayer().setImportanceSamplingWeights(importanceSamplingWeights);
        executeLock.unlock();
    }


    /**
     * Prints structure and metadata of neural network.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    public void print() throws NeuralNetworkException {
        checkNotStarted();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers) {
            neuralNetworkLayer.print();
            System.out.println();
        }
        System.out.println("Apply early stopping: " + (earlyStopping != null ? "Yes" : "No"));
        System.out.println();
    }

    /**
     * Prints expression chains of neural network.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    public void printExpressions() throws NeuralNetworkException {
        checkNotStarted();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers) neuralNetworkLayer.printExpressions();
    }

    /**
     * Prints gradient chains of neural network.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    public void printGradients() throws NeuralNetworkException {
        checkNotStarted();
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers) neuralNetworkLayer.printGradients();
    }

}

