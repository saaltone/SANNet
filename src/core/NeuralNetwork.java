/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core;

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
import utils.matrix.BinaryFunctionType;
import utils.matrix.Init;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Defines main class for neural network.<br>
 * Used to define, construct and execute neural network.<br>
 * Can support multiple layer of different types including regularization, normalization and optimization methods.<br>
 *
 */
public class NeuralNetwork implements Runnable, Serializable {

    private static final long serialVersionUID = -1075977720550636471L;

    /**
     * Defines states of neural network.
     *   IDLE: neural network is idle ready for operation call.
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
     * Name of neural network instance.
     *
     */
    private String neuralNetworkName;

    /**
     * Lock for synchronizing neural network thread operations.
     *
     */
    private transient Lock lock;

    /**
     * Lock condition for synchronizing execution procedures (train, predict, validate).
     *
     */
    private transient Condition execute;

    /**
     * Lock condition for synchronizing completion of procedure execution and shift to idle state.
     *
     */
    private transient Condition complete;

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
     * Reference to input layer of neural network.
     *
     */
    private InputLayer inputLayer;

    /**
     * List containing neural network layer in order starting from input layer ending to output layer.
     *
     */
    private final ArrayList<AbstractLayer> layers = new ArrayList<>();

    /**
     * Number of hidden layers.
     *
     */
    private int numberOfHiddenLayers = 0;

    /**
     * List of connectors between neural network layers.
     *
     */
    private final ArrayList<Connector> connectors = new ArrayList<>();

    /**
     * Reference to output layer of neural network.
     *
     */
    private OutputLayer outputLayer;

    /**
     * Reference to early stopping condition.
     *
     */
    private EarlyStopping earlyStopping;

    /**
     * Reference to training error metric.
     *
     */
    private transient Metrics trainingError;

    /**
     * Reference to validation error metric.
     *
     */
    private transient Metrics validationError;

    /**
     * Validation error metric type. Default REGRESSION.
     *
     */
    private MetricsType validationMetricsType = MetricsType.REGRESSION;

    /**
     * Defines how many entries (matrices) one sample has. Used especially for convolutional layer.
     *
     */
    private int sampleDepth = 1;

    /**
     * Structure containing training input samples with indices.
     *
     */
    private transient LinkedHashMap<Integer, Matrix> trainIns;

    /**
     * Structure containing training output samples with indices.
     *
     */
    private transient LinkedHashMap<Integer, Matrix> trainOuts;

    /**
     * Structure containing test input samples with indices. Used for neural network validation.
     *
     */
    private transient LinkedHashMap<Integer, Matrix> testIns;

    /**
     * Structure containing test output samples with indices. Used for neural network validation.
     *
     */
    private transient LinkedHashMap<Integer, Matrix> testOuts;

    /**
     * Flag is neural network and it's layers are to be reset prior training phase.
     *
     */
    private transient boolean reset;

    /**
     * Count of neural network training iterations per training phase.
     *
     */
    private transient int iterations;

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
     * Current training sample index in training sample set.
     *
     */
    private transient int trainingSampleAt;

    /**
     * Number of training samples per training step i.e. mini batch size.<br>
     * This is also used to define sequence length in sequence training for recurrent layers.<br>
     *
     */
    private int trainingSamplesPerStep = 1;

    /**
     * Flag if training samples are to be taken in order of training sample set or taken randomly from training sample set.
     *
     */
    private boolean trainingSampleInOrder = true;

    /**
     * Flag if samples inside training sample batch are to be kept in order or shuffled.
     *
     */
    private boolean trainingShuffleSamples = false;

    /**
     * Current validation sample index in validation sample set.
     *
     */
    private transient int validationSampleAt;

    /**
     * Number of validation samples per validation step.<br>
     * This is also used to define sequence length in sequence validation for recurrent layers.<br>
     *
     */
    private int validationSamplesPerStep = 1;

    /**
     * Flag if validation samples are to be taken in order of validation sample set or taken randomly from validation sample set.
     *
     */
    private boolean validationSampleInOrder = true;

    /**
     * Flag if samples inside validation sample batch are to be kept in order or shuffled.
     *
     */
    private boolean validationShuffleSamples = false;

    /**
     * Random number generator for neural network class.
     *
     */
    private Random random;

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
     * Cycle lenght as iterators for neural network verbosing.
     *
     */
    private int verboseCycle;

    /**
     * Default constructor for neural network.
     *
     */
    public NeuralNetwork() {
    }

    /**
     * Sets name for neural network instance.
     *
     * @param neuralNetworkName name for neural network instance.
     * @throws NeuralNetworkException throws exception is neural network instance is already started.
     */
    public void setNeuralNetworkName(String neuralNetworkName) throws NeuralNetworkException {
        checkStarted();
        this.neuralNetworkName = neuralNetworkName;
    }

    /**
     * Returns name of neural network instance.
     *
     * @return name of neural network instance.
     */
    public String getNeuralNetworkName() {
        return neuralNetworkName;
    }

    /**
     * Adds regularizer for specific neural network connector (layer).
     * Applies to next layer of connector.
     *
     * @param connectorIndex connector to which regularizer is added to. Index starts from 0 (connector between input layer and next layer).
     * @param regularizationType type of regularizer.
     * @throws NeuralNetworkException throws neural network exception if adding of regularizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addRegularizer(int connectorIndex, RegularizationType regularizationType) throws NeuralNetworkException, DynamicParamException {
        addRegularizer(connectorIndex, regularizationType, null);
    }

    /**
     * Adds regularizer for all neural network connectors (layers).
     *
     * @param regularizationType type of regularizer.
     * @throws NeuralNetworkException throws neural network exception if adding of regularizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addRegularizer(RegularizationType regularizationType) throws NeuralNetworkException, DynamicParamException {
        addRegularizer(regularizationType, null);
    }

    /**
     * Adds regularizer for specific neural network connector (layer).<br>
     * Applies to next layer of connector.<br>
     *
     * @param connectorIndex connector to which regularizer is added to. Index starts from 0 (connector between input layer and next layer).
     * @param regularizationType type of regularizer.
     * @param params parameters for regularizer.
     * @throws NeuralNetworkException throws neural network exception if adding of regularizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addRegularizer(int connectorIndex, RegularizationType regularizationType, String params) throws NeuralNetworkException, DynamicParamException {
        checkStarted();
        if (connectorIndex < 0 || connectorIndex > connectors.size() - 1) throw new NeuralNetworkException("No connector index: " + connectorIndex + " exists.");
        connectors.get(connectorIndex).addRegularization(regularizationType, params);
    }

    /**
     * Adds regularizer for all neural network connectors (layers).
     *
     * @param regularizationType type of regularizer.
     * @param params parameters for regularizer.
     * @throws NeuralNetworkException throws neural network exception if adding of regularizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addRegularizer(RegularizationType regularizationType, String params) throws NeuralNetworkException, DynamicParamException {
        checkStarted();
        for (Connector connector : connectors) connector.addRegularization(regularizationType, params);
    }

    /**
     * Removes regularizer from specific neural network connector (layer).<br>
     * Applies to next layer of connector.<br>
     *
     * @param connectorIndex connector to which regularizer is added to. Index starts from 0 (connector between input layer and next layer).
     * @param regularizationType type of regularizer.
     * @throws NeuralNetworkException throws neural network exception if adding of regularizer fails.
     */
    public void removeRegularizer(int connectorIndex, RegularizationType regularizationType) throws NeuralNetworkException {
        checkStarted();
        if (connectorIndex < 0 || connectorIndex > connectors.size() - 1) throw new NeuralNetworkException("No connector index: " + connectorIndex + " exists.");
        connectors.get(connectorIndex).removeRegularization(regularizationType);
    }

    /**
     * Removes regularizer from all neural network connectors (layer).
     *
     * @param regularizationType type of regularizer.
     * @throws NeuralNetworkException throws neural network exception if removal of regularizer fails.
     */
    public void removeRegularizer(RegularizationType regularizationType) throws NeuralNetworkException {
        checkStarted();
        for (Connector connector : connectors) connector.removeRegularization(regularizationType);
    }

    /**
     * Removes regularizer from specific neural network connector (layer).
     *
     * @param connectorIndex connector to which regularizer is added to. Index starts from 0 (connector between input layer and next layer).
     * @throws NeuralNetworkException throws neural network exception if removal of regularizer fails.
     */
    public void removeRegularization(int connectorIndex) throws NeuralNetworkException {
        checkStarted();
        if (connectorIndex < 0 || connectorIndex > connectors.size() - 1) throw new NeuralNetworkException("No connector index: " + connectorIndex + " exists.");
        connectors.get(connectorIndex).removeRegularization();
    }

    /**
     * Removes all regularizers from neural network (layers).
     *
     * @throws NeuralNetworkException throws neural network exception if removal of regularizers fails.
     */
    public void removeRegularization() throws NeuralNetworkException {
        checkStarted();
        for (Connector connector : connectors) connector.removeRegularization();
    }

    /**
     * Resets regularization for all neural network connectors (layers).
     *
     * @throws NeuralNetworkException throws neural network exception if resetting of regularization fails.
     */
    public void resetRegularization() throws NeuralNetworkException {
        checkStarted();
        for (Connector connector : connectors) connector.resetRegularization();
    }

    /**
     * Resets regularization of specific type for all neural network connectors (layers).
     *
     * @param regularizationType regularization method to be reset.
     * @throws NeuralNetworkException throws neural network exception if resetting of regularization fails.
     */
    public void resetRegularization(RegularizationType regularizationType) throws NeuralNetworkException {
        checkStarted();
        for (Connector connector : connectors) connector.resetRegularization(regularizationType);
    }

    /**
     * Resets regularization for specific neural network connector (layer).
     *
     * @param connectorIndex connector of which optimizer is reset. Index starts from 0 (connector between input layer and next layer).
     * @throws NeuralNetworkException throws neural network exception if resetting of regularization fails.
     */
    public void resetRegularization(int connectorIndex) throws NeuralNetworkException {
        checkStarted();
        if (connectorIndex < 0 || connectorIndex > connectors.size() - 1) throw new NeuralNetworkException("No connector index: " + connectorIndex + " exists.");
        connectors.get(connectorIndex).resetRegularization();
    }

    /**
     * Resets regularization of specific type for specific neural network connector (layer).
     *
     * @param connectorIndex connector of which optimizer is reset. Index starts from 0 (connector between input layer and next layer).
     * @param regularizationType regularization method to be reset.
     * @throws NeuralNetworkException throws neural network exception if resetting of regularization fails.
     */
    public void resetRegularization(int connectorIndex, RegularizationType regularizationType) throws NeuralNetworkException {
        checkStarted();
        if (connectorIndex < 0 || connectorIndex > connectors.size() - 1) throw new NeuralNetworkException("No connector index: " + connectorIndex + " exists.");
        connectors.get(connectorIndex).resetRegularization(regularizationType);
    }

    /**
     * Adds normalizer for specific neural network connector (layer).<br>
     * Applies to next layer of connector.<br>
     *
     * @param connectorIndex connector to which normalizer is added to. Index starts from 0 (connector between input layer and next layer).
     * @param normalizationType type of normalizer.
     * @throws NeuralNetworkException throws neural network exception if adding of normalizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addNormalizer(int connectorIndex, NormalizationType normalizationType) throws NeuralNetworkException, DynamicParamException {
        addNormalizer(connectorIndex, normalizationType, null);
    }

    /**
     * Adds normalizer for all neural network connectors (layers).
     *
     * @param normalizationType type of normalizer.
     * @throws NeuralNetworkException throws neural network exception if adding of normalizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addNormalizer(NormalizationType normalizationType) throws NeuralNetworkException, DynamicParamException {
        addNormalizer(normalizationType, null);
    }

    /**
     * Adds normalizer for specific neural network connector (layer).<br>
     * Applies to next layer of connector.<br>
     *
     * @param connectorIndex connector to which normalizer is added to. Index starts from 0 (connector between input layer and next layer).
     * @param normalizationType type of normalizer.
     * @param params parameters for normalizer.
     * @throws NeuralNetworkException throws neural network exception if adding of normalizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addNormalizer(int connectorIndex, NormalizationType normalizationType, String params) throws NeuralNetworkException, DynamicParamException {
        checkStarted();
        if (connectorIndex < 0 || connectorIndex > connectors.size() - 1) throw new NeuralNetworkException("No connector index: " + connectorIndex + " exists.");
        connectors.get(connectorIndex).addNormalization(normalizationType, params);
    }

    /**
     * Adds normalizer for all neural network connectors (layers).
     *
     * @param normalizationType type of normalizer.
     * @param params parameters for normalizer.
     * @throws NeuralNetworkException throws neural network exception if adding of normalizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addNormalizer(NormalizationType normalizationType, String params) throws NeuralNetworkException, DynamicParamException {
        checkStarted();
        for (Connector connector : connectors) connector.addNormalization(normalizationType, params);
    }

    /**
     * Removes normalizer from specific neural network connector (layer).<br>
     * Applies to next layer of connector.<br>
     *
     * @param connectorIndex connector to which normalizer is added to. Index starts from 0 (connector between input layer and next layer).
     * @param normalizationType type of normalizer.
     * @throws NeuralNetworkException throws neural network exception if adding of normalizer fails.
     */
    public void removeNormalizer(int connectorIndex, NormalizationType normalizationType) throws NeuralNetworkException {
        checkStarted();
        if (connectorIndex < 0 || connectorIndex > connectors.size() - 1) throw new NeuralNetworkException("No connector index: " + connectorIndex + " exists.");
        connectors.get(connectorIndex).removeNormalization(normalizationType);
    }

    /**
     * Removes normalizer from all neural network connectors (layer).
     *
     * @param normalizationType type of normalizer.
     * @throws NeuralNetworkException throws neural network exception if removal of normalizer fails.
     */
    public void removeNormalizer(NormalizationType normalizationType) throws NeuralNetworkException {
        checkStarted();
        for (Connector connector : connectors) connector.removeNormalization(normalizationType);
    }

    /**
     * Removes normalizer from specific neural network connector (layer).
     *
     * @param connectorIndex connector to which normalizer is added to. Index starts from 0 (connector between input layer and next layer).
     * @throws NeuralNetworkException throws neural network exception if removal of normalizer fails.
     */
    public void removeNormalization(int connectorIndex) throws NeuralNetworkException {
        checkStarted();
        if (connectorIndex < 0 || connectorIndex > connectors.size() - 1) throw new NeuralNetworkException("No connector index: " + connectorIndex + " exists.");
        connectors.get(connectorIndex).removeNormalization();
    }

    /**
     * Removes all normalizers from neural network (layers).
     *
     * @throws NeuralNetworkException throws neural network exception if removal of normalizers fails.
     */
    public void removeNormalization() throws NeuralNetworkException {
        checkStarted();
        for (Connector connector : connectors) connector.removeNormalization();
    }

    /**
     * Resets normalization for all neural network connectors (layers).
     *
     * @throws NeuralNetworkException throws neural network exception if resetting of normalization fails.
     */
    public void resetNormalization() throws NeuralNetworkException {
        checkStarted();
        for (Connector connector : connectors) connector.resetNormalization();
    }

    /**
     * Resets normalization of specific type for all neural network connectors (layers).
     *
     * @param normalizationType normalization method to be reset.
     * @throws NeuralNetworkException throws neural network exception if resetting of normalization fails.
     */
    public void resetNormalization(NormalizationType normalizationType) throws NeuralNetworkException {
        checkStarted();
        for (Connector connector : connectors) connector.resetNormalization(normalizationType);
    }

    /**
     * Resets normalization for specific neural network connector (layer).
     *
     * @param connectorIndex connector of which optimizer is reset. Index starts from 0 (connector between input layer and next layer).
     * @throws NeuralNetworkException throws neural network exception if resetting of normalization fails.
     */
    public void resetNormalization(int connectorIndex) throws NeuralNetworkException {
        checkStarted();
        if (connectorIndex < 0 || connectorIndex > connectors.size() - 1) throw new NeuralNetworkException("No connector index: " + connectorIndex + " exists.");
        connectors.get(connectorIndex).resetNormalization();
    }

    /**
     * Resets normalization of specific type for specific neural network connector (layer).
     *
     * @param connectorIndex connector of which optimizer is reset. Index starts from 0 (connector between input layer and next layer).
     * @param normalizationType normalization method to be reset.
     * @throws NeuralNetworkException throws neural network exception if resetting of normalization fails.
     */
    public void resetNormalization(int connectorIndex, NormalizationType normalizationType) throws NeuralNetworkException {
        checkStarted();
        if (connectorIndex < 0 || connectorIndex > connectors.size() - 1) throw new NeuralNetworkException("No connector index: " + connectorIndex + " exists.");
        connectors.get(connectorIndex).resetNormalization(normalizationType);
    }

    /**
     * Sets optimizer for specific neural network connector (layer).<br>
     * Applies to next layer of connector.<br>
     *
     * @param connectorIndex connector to which optimizer is added to. Index starts from 0 (connector between input layer and next layer).
     * @param optimization type of optimizer.
     * @param params parameters for optimizer.
     * @throws NeuralNetworkException throws neural network exception if setting of optimizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setOptimizer(int connectorIndex, OptimizationType optimization, String params) throws NeuralNetworkException, DynamicParamException {
        checkStarted();
        if (connectorIndex < 0 || connectorIndex > connectors.size() - 1) throw new NeuralNetworkException("No connector index: " + connectorIndex + " exists.");
        connectors.get(connectorIndex).setOptimizer(OptimizerFactory.create(optimization, params));
    }

    /**
     * Sets optimizer for all neural network connectors (layers).
     *
     * @param  optimization type of optimizer.
     * @param params parameters for optimizer.
     * @throws NeuralNetworkException throws neural network exception if setting of optimizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setOptimizer(OptimizationType optimization, String params) throws NeuralNetworkException, DynamicParamException {
        checkStarted();
        for (Connector connector : connectors) connector.setOptimizer(OptimizerFactory.create(optimization, params));
    }

    /**
     * Sets optimizer for specific neural network connector (layer).<br>
     * Applies to next layer of connector.<br>
     *
     * @param connectorIndex connector to which optimizer is added to. Index starts from 0 (connector between input layer and next layer).
     * @param optimization type of optimizer.
     * @throws NeuralNetworkException throws neural network exception if setting of optimizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setOptimizer(int connectorIndex, OptimizationType optimization) throws NeuralNetworkException, DynamicParamException {
        setOptimizer(connectorIndex, optimization, null);
    }

    /**
     * Sets optimizer for all neural network connectors (layers).
     *
     * @param optimization type of optimizer.
     * @throws NeuralNetworkException throws neural network exception if setting of optimizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setOptimizer(OptimizationType optimization) throws NeuralNetworkException, DynamicParamException {
        setOptimizer(optimization, null);
    }

    /**
     * Resets optimizer for all neural network connectors (layers).
     *
     * @throws NeuralNetworkException throws neural network exception if resetting of optimizer fails.
     */
    public void resetOptimizer() throws NeuralNetworkException {
        checkStarted();
        for (Connector connector : connectors) connector.resetOptimizer();
    }

    /**
     * Resets optimizer for specific neural network connector (layer).
     *
     * @param connectorIndex connector of which optimizer is reset. Index starts from 0 (connector between input layer and next layer).
     * @throws NeuralNetworkException throws neural network exception if resetting of optimizer fails.
     */
    public void resetOptimizer(int connectorIndex) throws NeuralNetworkException {
        checkStarted();
        if (connectorIndex < 0 || connectorIndex > connectors.size() - 1) throw new NeuralNetworkException("No connector index: " + connectorIndex + " exists.");
        connectors.get(connectorIndex).resetOptimizer();
    }

    /**
     * Sets loss function for neural network (output layer)
     *
     * @param lossFunctionType type of loss function.
     * @throws NeuralNetworkException throws exception if setting of loss function fails.
     */
    public void setLossFunction(BinaryFunctionType lossFunctionType) throws NeuralNetworkException {
        checkStarted();
        if (getOutputLayer() == null) throw new NeuralNetworkException("Output layer is not defined for a neural network.");
        LossFunction lossFunction = new LossFunction(lossFunctionType);
        getOutputLayer().setLossFunction(lossFunction);
    }

    /**
     * Sets loss function for neural network (output layer)
     *
     * @param lossFunctionType type of loss function.
     * @param params parameters for loss function.
     * @throws NeuralNetworkException throws exception if setting of loss function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setLossFunction(BinaryFunctionType lossFunctionType, String params) throws NeuralNetworkException, DynamicParamException {
        checkStarted();
        if (getOutputLayer() == null) throw new NeuralNetworkException("Output layer is not defined for a neural network.");
        LossFunction lossFunction = new LossFunction(lossFunctionType, params);
        getOutputLayer().setLossFunction(lossFunction);
    }

    /**
     * Adds input layer for neural network.
     *
     * @param params parameters for input layer.
     * @throws NeuralNetworkException throws neural network exception if adding of input layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addInputLayer(String params) throws NeuralNetworkException, DynamicParamException {
        checkStarted();
        inputLayer = new InputLayer(0, params);
        layers.add(inputLayer);
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
     * Adds hidden layer for neural network. Hidden layers are executed in order which they are added.
     *
     * @param layerType type of hidden layer.
     * @throws NeuralNetworkException throws neural network exception if adding of hidden layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addHiddenLayer(LayerType layerType) throws NeuralNetworkException, DynamicParamException {
        addHiddenLayer(layerType, null, null, null);
    }

    /**
     * Adds hidden layer for neural network. Hidden layers are executed in order which they are added.
     *
     * @param layerType type of hidden layer.
     * @param params parameters for hidden layer.
     * @throws NeuralNetworkException throws neural network exception if adding of hidden layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addHiddenLayer(LayerType layerType, String params) throws NeuralNetworkException, DynamicParamException {
        addHiddenLayer(layerType, null, null, params);
    }

    /**
     * Adds hidden layer for neural network. Hidden layers are executed in order which they are added.
     *
     * @param layerType type of hidden layer.
     * @param activationFunction activation function for hidden layer.
     * @throws NeuralNetworkException throws neural network exception if adding of hidden layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addHiddenLayer(LayerType layerType, ActivationFunction activationFunction) throws NeuralNetworkException, DynamicParamException {
        addHiddenLayer(layerType, activationFunction, null, null);
    }

    /**
     * Adds hidden layer for neural network. Hidden layers are executed in order which they are added.
     *
     * @param layerType type of hidden layer.
     * @param activationFunction activation function for hidden layer.
     * @param params parameters for hidden layer.
     * @throws NeuralNetworkException throws neural network exception if adding of hidden layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addHiddenLayer(LayerType layerType, ActivationFunction activationFunction, String params) throws NeuralNetworkException, DynamicParamException {
        addHiddenLayer(layerType, activationFunction, null, params);
    }

    /**
     * Adds hidden layer for neural network. Hidden layers are executed in order which they are added.
     *
     * @param layerType type of hidden layer.
     * @param activationFunction activation function for hidden layer.
     * @param initialization layer parameter initialization function for hidden layer.
     * @throws NeuralNetworkException throws neural network exception if adding of hidden layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addHiddenLayer(LayerType layerType, ActivationFunction activationFunction, Init initialization) throws NeuralNetworkException, DynamicParamException {
        addHiddenLayer(layerType, activationFunction, initialization, null);
    }

    /**
     * Adds hidden layer for neural network. Hidden layers are executed in order which they are added.
     *
     * @param layerType type of hidden layer.
     * @param activationFunction activation function for hidden layer.
     * @param initialization layer parameter initialization function for hidden layer.
     * @param params parameters for hidden layer.
     * @throws NeuralNetworkException throws neural network exception if adding of hidden layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addHiddenLayer(LayerType layerType, ActivationFunction activationFunction, Init initialization, String params) throws NeuralNetworkException, DynamicParamException {
        checkStarted();
        numberOfHiddenLayers++;
        layers.add(new HiddenLayer(numberOfHiddenLayers, layerType, activationFunction, initialization, params));
    }

    /**
     * Adds output layer for neural network.
     *
     * @param layerType type of output layer.
     * @throws NeuralNetworkException throws neural network exception if adding of output layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addOutputLayer(LayerType layerType) throws NeuralNetworkException, DynamicParamException {
        addOutputLayer(layerType, null, null, null);
    }

    /**
     * Adds output layer for neural network.
     *
     * @param layerType type of output layer.
     * @param params parameters for output layer.
     * @throws NeuralNetworkException throws neural network exception if adding of output layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addOutputLayer(LayerType layerType, String params) throws NeuralNetworkException, DynamicParamException {
        addOutputLayer(layerType, null, null, params);
    }

    /**
     * Adds output layer for neural network.
     *
     * @param layerType type of output layer.
     * @param activationFunction activation function for output layer.
     * @throws NeuralNetworkException throws neural network exception if adding of output layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addOutputLayer(LayerType layerType, ActivationFunction activationFunction) throws NeuralNetworkException, DynamicParamException {
        addOutputLayer(layerType, activationFunction, null, null);
    }

    /**
     * Adds output layer for neural network.
     *
     * @param layerType type of output layer.
     * @param activationFunction activation function for output layer.
     * @param params parameters for output layer.
     * @throws NeuralNetworkException throws neural network exception if adding of output layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addOutputLayer(LayerType layerType, ActivationFunction activationFunction, String params) throws NeuralNetworkException, DynamicParamException {
        addOutputLayer(layerType, activationFunction, null, params);
    }

    /**
     * Adds output layer for neural network.
     *
     * @param layerType type of output layer.
     * @param activationFunction activation function for output layer.
     * @param initialization layer parameter initialization function for output layer.
     * @throws NeuralNetworkException throws neural network exception if adding of output layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addOutputLayer(LayerType layerType, ActivationFunction activationFunction, Init initialization) throws NeuralNetworkException, DynamicParamException {
        addOutputLayer(layerType, activationFunction, initialization, null);
    }

    /**
     * Adds output layer for neural network.
     *
     * @param layerType type of output layer.
     * @param activationFunction activation function for output layer.
     * @param initialization layer parameter initialization function for output layer.
     * @param params parameters for output layer.
     * @throws NeuralNetworkException throws neural network exception if adding of output layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void addOutputLayer(LayerType layerType, ActivationFunction activationFunction, Init initialization, String params) throws NeuralNetworkException, DynamicParamException {
        checkStarted();
        outputLayer = new OutputLayer(-1, layerType, activationFunction, initialization, params);
        layers.add(outputLayer);
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
     * Returns connectors of neural network.
     *
     * @return connectors of neural network.
     */
    public ArrayList<Connector> getConnectors() {
        return connectors;
    }

    /**
     * Builds neural network.<br>
     * Connects layers to each other with connectors.<br>
     * Initializes layers.<br>
     *
     * @throws MatrixException throws exception is matrix operation fails.
     * @throws NeuralNetworkException throws neural network exception if building of neural network fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void build() throws MatrixException, NeuralNetworkException, DynamicParamException {
        checkStarted();

        connectors.clear();
        for (int layerID = 0; layerID < layers.size() - 1; layerID++) {
            Connector connector = HiddenLayer.Connect(layers.get(layerID), layers.get(layerID + 1));
            connectors.add(connector);
        }
        for (AbstractLayer layer : layers) {
            layer.initialize();
        }
    }

    /**
     * Sets persistence instance for neural network.<br>
     * Persistence is used to serialize neural network instance and store to file or deserialize neural network from file.<br>
     *
     * @param persistence persistence instance.
     */
    public void setPersistence(Persistence persistence) {
        waitToComplete();
        this.persistence = persistence;
    }

    /**
     * Unsets (removes) persistence instance.
     *
     */
    public void unsetPersistence() {
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
     */
    public void start() throws NeuralNetworkException {
        if (neuralNetworkThread != null) return;

        trainingError = new Metrics(MetricsType.REGRESSION);
        if (validationError == null) validationError = new Metrics(validationMetricsType);

        trainIns = new LinkedHashMap<>();
        trainOuts = new LinkedHashMap<>();

        trainingSampleAt = 0;

        testIns = new LinkedHashMap<>();
        testOuts = new LinkedHashMap<>();

        validationSampleAt = 0;

        if (earlyStopping != null) {
            earlyStopping.setTrainingError(trainingError);
            earlyStopping.setValidationError(validationError);
        }

        random = new Random();

        lock = new ReentrantLock();
        execute = lock.newCondition();
        complete = lock.newCondition();
        executionState = ExecutionState.IDLE;

        stopLock = new ReentrantLock();
        stopExecution = false;

        neuralNetworkThread = new Thread(this);
        neuralNetworkThread.setName("NeuralNetwork" + (neuralNetworkName != null ? " (" + neuralNetworkName + ")" : ""));
        neuralNetworkThread.start();

        getInputLayer().start();

    }

    /**
     * Checks if neural network is started (running).
     *
     * @throws NeuralNetworkException throws exception is neural network is started.
     */
    private void checkStarted() throws NeuralNetworkException {
        if (neuralNetworkThread != null) {
            if (neuralNetworkThread.getState() != Thread.State.NEW) throw new NeuralNetworkException("Neural network is started");
        }
    }

    /**
     * Aborts execution of neural network.<br>
     * Execution is aborted after execution of last single operation is completed.<br>
     * Useful when neural network is executing multiple training iterations.<br>
     *
     */
    public void abortExecution() {
        if (neuralNetworkThread == null) return;
        stopLock.lock();
        stopExecution = true;
        stopLock.unlock();
    }

    /**
     * Checks if neural network is not started (running).
     *
     * @throws NeuralNetworkException throws exception is neural network is not started.
     */
    private void checkNotStarted() throws NeuralNetworkException {
        if (neuralNetworkThread == null) throw new NeuralNetworkException("Neural network is not started");
    }

    /**
     * Sets if recurrent inputs of layer are allowed to be reset.<br>
     * Useful when long input sequence is applied over multiple inferences (prediction) requests.<br>
     *
     * @param allowLayerReset if true recurrent inputs are allowed to be reset.
     * @throws NeuralNetworkException throws exception if neural network is not started.
     */
    public void setAllowLayerReset(boolean allowLayerReset) throws NeuralNetworkException {
        checkNotStarted();
        waitToComplete();
        getInputLayer().setAllowLayerReset(allowLayerReset);
    }

    /**
     * Sets sample depth i.e. how many entries (matrices) single sample has. Used especially for convolutional layer. Default value 1.
     *
     * @param sampleDepth sample depth.
     */
    public void setSampleDepth(int sampleDepth) {
        this.sampleDepth = sampleDepth;
    }

    /**
     * Returns sample depth.
     *
     * @return sample depth.
     */
    public int getSampleDepth() {
        return sampleDepth;
    }

    /**
     * Sets training sample sets of neural network.<br>
     * Equal indices of input and output reflect input output pair of neural network.<br>
     *
     * @param inputs training input sample set.
     * @param outputs training output sample set.
     * @throws NeuralNetworkException throws exception if setting of training sample sets fail.
     */
    public void setTrainingData(LinkedHashMap<Integer, Matrix> inputs, LinkedHashMap<Integer, Matrix> outputs) throws NeuralNetworkException {
        checkNotStarted();
        waitToComplete();
        lock.lock();
        if (!inputs.isEmpty() && !outputs.isEmpty()) {
            if (inputs.size() != outputs.size()) {
                lock.unlock();
                throw new NeuralNetworkException("Training input and output sample amount must be equal length");
            }
            trainIns = new LinkedHashMap<>();
            trainIns.putAll(inputs);
            trainOuts = new LinkedHashMap<>();
            trainOuts.putAll(outputs);
            trainingSampleAt = 0;
        }
        lock.unlock();
    }

    /**
     * Sets number of training iterations for execution.
     *
     * @param iterations number of training iterations.
     */
    public void setTrainingIterations(int iterations) {
        waitToComplete();
        this.iterations = iterations;
    }

    /**
     * Sets early stopping condition.
     *
     * @param earlyStopping early stopping instance.
     */
    public void setTrainingEarlyStopping(EarlyStopping earlyStopping) {
        waitToComplete();
        this.earlyStopping = earlyStopping;
        if (earlyStopping != null) {
            earlyStopping.setTrainingError(trainingError);
            earlyStopping.setValidationError(validationError);
        }
    }

    /**
     * Sets auto validation on.
     *
     * @param autoValidationCycle validation cycle in iterations.
     */
    public void setAutoValidate(int autoValidationCycle) {
        waitToComplete();
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
        train(null, null, false, true);
    }

    /**
     * Trains neural network and sets specific training input and output samples.
     *
     * @param inputs training input sample set.
     * @param outputs training output sample set.
     * @throws NeuralNetworkException throws exception if training procedure fails.
     */
    public void train(LinkedHashMap<Integer, Matrix> inputs, LinkedHashMap<Integer, Matrix> outputs) throws NeuralNetworkException {
        train(inputs, outputs, false, true);
    }

    /**
     * Trains neural network with option to reset neural network and it's layers.
     *
     * @param reset if true resets neural network and it's layers.
     * @throws NeuralNetworkException throws exception if training procedure fails.
     */
    public void train(boolean reset) throws NeuralNetworkException {
        train(null, null, reset, true);
    }

    /**
     * Trains neural network with option to reset neural network and it's layers.<br>
     * Sets specific training input and output samples.<br>
     *
     * @param inputs training input sample set.
     * @param outputs training output sample set.
     * @param reset if true resets neural network and it's layers.
     * @throws NeuralNetworkException throws exception if training procedure fails.
     */
    public void train(LinkedHashMap<Integer, Matrix> inputs, LinkedHashMap<Integer, Matrix> outputs, boolean reset) throws NeuralNetworkException {
        train(inputs, outputs, reset, true);
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
        train(null, null, reset, waitToComplete);
    }

    /**
     * Trains neural network with option to reset neural network and it's layers.<br>
     * Sets specific training input and output samples.<br>
     * Optionally waits neural network training procedure to complete.<br>
     *
     * @param inputs training input sample set.
     * @param outputs training output sample set.
     * @param reset if true resets neural network and it's layers.
     * @param waitToComplete if true waits for neural network execution complete otherwise returns function prior training completion.
     * @throws NeuralNetworkException throws exception if training procedure fails.
     */
    public void train(LinkedHashMap<Integer, Matrix> inputs, LinkedHashMap<Integer, Matrix> outputs, boolean reset, boolean waitToComplete) throws NeuralNetworkException {
        checkNotStarted();
        waitToComplete();
        lock.lock();
        if (inputs != null && outputs != null) {
            if (!inputs.isEmpty() && !outputs.isEmpty()) {
                if (inputs.size() != outputs.size()) {
                    lock.unlock();
                    throw new NeuralNetworkException("Training input and output sample amount must be equal length");
                }
                trainIns = new LinkedHashMap<>();
                trainIns.putAll(inputs);
                trainOuts = new LinkedHashMap<>();
                trainOuts.putAll(outputs);
                trainingSampleAt = 0;
            }
        }
        if (trainIns.isEmpty() || trainOuts.isEmpty()) {
            lock.unlock();
            throw new NeuralNetworkException("No training inputs and outputs set");
        }
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
    public LinkedHashMap<Integer, Matrix> predict() throws NeuralNetworkException {
        return predict(null, true);
    }

    /**
     * Predicts values based on given input.
     *
     * @param input input for prediction.
     * @return predicted value (neural network output).
     * @throws NeuralNetworkException throws exception if prediction fails.
     */
    public Matrix predict(Matrix input) throws NeuralNetworkException {
        LinkedHashMap<Integer, Matrix> inputs = new LinkedHashMap<>();
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
    public LinkedHashMap<Integer, Matrix> predict(LinkedHashMap<Integer, Matrix> inputs) throws NeuralNetworkException {
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
    public LinkedHashMap<Integer, Matrix> predict(boolean waitToComplete) throws NeuralNetworkException {
        return predict(null, waitToComplete);
    }

    /**
     * Predicts values based on current test set inputs.<br>
     * Optionally waits neural network prediction procedure to complete.<br>
     * Sets specific inputs for prediction.<br>
     *
     * @param inputs test input set for prediction.
     * @param waitToComplete if true waits for neural network execution complete otherwise returns function prior prediction completion.
     * @return predicted values (neural network output).
     * @throws NeuralNetworkException throws exception if prediction fails.
     */
    public LinkedHashMap<Integer, Matrix> predict(LinkedHashMap<Integer, Matrix> inputs, boolean waitToComplete) throws NeuralNetworkException {
        checkNotStarted();
        waitToComplete();
        lock.lock();
        if (inputs != null) {
            if (!inputs.isEmpty()) {
                testIns = new LinkedHashMap<>();
                testIns.putAll(inputs);
                testOuts = new LinkedHashMap<>();
            }
        }
        if (testIns.isEmpty()) {
            lock.unlock();
            throw new NeuralNetworkException("No prediction inputs set");
        }
        nextState(ExecutionState.PREDICT);
        if (waitToComplete) {
            waitToComplete();
            return new LinkedHashMap<>(getOutput());
        }
        else return null;
    }

    /**
     * Sets verbosing of for validation phase.<br>
     * Follows training verbosing cycle.<br>
     *
     */
    public void verboseValidation() {
        waitToComplete();
        verboseValidation = true;
    }

    /**
     * Unsets verbosing of for validation phase.
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
        validate(null, null, true);
    }

    /**
     * Validates neural network with current test input and output (actual true values) sample set.<br>
     * Optionally waits neural network validation procedure to complete.<br>
     *
     * @param waitToComplete if true waits for neural network execution complete otherwise returns function prior validation completion.
     * @throws NeuralNetworkException throws exception if validation fails.
     */
    public void validate(boolean waitToComplete) throws NeuralNetworkException {
        validate(null, null, waitToComplete);
    }

    /**
     * Validates neural network.<br>
     * Sets specific input and output (actual true values) test samples.<br>
     *
     * @param inputs test input set for validation.
     * @param actual test output (actual true values) set for validation.
     * @throws NeuralNetworkException throws exception if validation fails.
     */
    public void validate(LinkedHashMap<Integer, Matrix> inputs, LinkedHashMap<Integer, Matrix> actual) throws NeuralNetworkException {
        validate(inputs, actual, true);
    }

    /**
     * Sets validation data.
     *
     * @param inputs test input set for validation.
     * @param actual test output (actual true values) set for validation.
     * @throws NeuralNetworkException throws exception if setting of validation data fails.
     */
    public void setValidationData(LinkedHashMap<Integer, Matrix> inputs, LinkedHashMap<Integer, Matrix> actual) throws NeuralNetworkException {
        checkNotStarted();
        waitToComplete();
        lock.lock();
        if (!inputs.isEmpty() && !actual.isEmpty()) {
            if (inputs.size() != actual.size()) {
                lock.unlock();
                throw new NeuralNetworkException("Validation input and actual sample amount must be equal length");
            }
            testIns = new LinkedHashMap<>();
            testIns.putAll(inputs);
            testOuts = new LinkedHashMap<>();
            testOuts.putAll(actual);
            validationSampleAt = 0;
        }
        lock.unlock();
    }

    /**
     * Validates neural network.<br>
     * Sets specific input and output (actual true values) test samples.<br>
     * Optionally waits neural network validation procedure to complete.<br>
     *
     * @param inputs test input set for validation.
     * @param actual test output (actual true values) set for validation.
     * @param waitToComplete if true waits for neural network execution complete otherwise returns function prior validation completion.
     * @throws NeuralNetworkException throws exception if validation fails.
     */
    public void validate(LinkedHashMap<Integer, Matrix> inputs, LinkedHashMap<Integer, Matrix> actual, boolean waitToComplete) throws NeuralNetworkException {
        checkNotStarted();
        waitToComplete();
        lock.lock();
        if (inputs != null && actual != null) {
            if (!inputs.isEmpty() && !actual.isEmpty()) {
                if (inputs.size() != actual.size()) {
                    lock.unlock();
                    throw new NeuralNetworkException("Validation input and actual sample amount must be equal length");
                }
                testIns = new LinkedHashMap<>();
                testIns.putAll(inputs);
                testOuts = new LinkedHashMap<>();
                testOuts.putAll(actual);
                validationSampleAt = 0;
            }
        }
        if (testIns.isEmpty() || testOuts.isEmpty()) {
            lock.unlock();
            throw new NeuralNetworkException("No validation inputs and actual set");
        }
        nextState(ExecutionState.VALIDATE);
        if (waitToComplete) waitToComplete();
    }

    /**
     * Sets next state for neural network.
     *
     * @param executionState next state for neural network.
     */
    private void nextState(ExecutionState executionState) {
        this.executionState = executionState;
        execute.signal();
        lock.unlock();
    }

    /**
     * Stops neural network.
     *
     */
    public void stop() {
        waitToComplete();
        getInputLayer().stop();
        trainIns = null;
        trainOuts = null;
        testIns = null;
        testOuts = null;
        lock.lock();
        executionState = ExecutionState.TERMINATED;
        execute.signal();
        lock.unlock();
    }

    /**
     * Returns output of neural network (output layer).
     *
     * @return output of neural network.
     */
    public LinkedHashMap<Integer, Matrix> getOutput() {
        waitToComplete();
        return getOutputLayer().getOutput().getFlatSequence();
    }

    /**
     * Thread run function.<br>
     * Executes given neural network procedures and synchronizes their execution via neural network thread execution lock.<br>
     *
     */
    public void run() {
        while (true) {
            lock.lock();
            try {
                while (executionState == ExecutionState.IDLE) execute.await();
            }
            catch (InterruptedException exception) {}
            try {
                switch (executionState) {
                    case TRAIN:
                        trainIterations();
                        break;
                    case PREDICT:
                        predictInput();
                        break;
                    case VALIDATE:
                        validateInput(true);
                        break;
                    case TERMINATED:
                        getInputLayer().stop();
                        neuralNetworkThread = null;
                        complete.signal();
                        lock.unlock();
                        return;
                }
            }
            catch (Exception exception) {
                exception.printStackTrace();
                System.exit(-1);
            }
            lock.unlock();
        }
    }

    /**
     * Trains neural network defines number of iterations.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws IOException throws exception if neural network persistence operation fails.
     * @throws NeuralNetworkException throws exception if neural network training fails.
     */
    private void trainIterations() throws MatrixException, IOException, NeuralNetworkException {
        trainingError.resetError();
        for (int iteration = 0; iteration < iterations; iteration++) {
            trainIteration();
            if (stoppedExecution()) break;
            if (earlyStopping != null) {
                if (earlyStopping.stopTraining()) break;
            }
        }
        trainingError.store(totalIterations);
        stateCompleted();
    }

    /**
     * Trains single neural network iteration.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws IOException throws exception if neural network persistence operation fails.
     * @throws NeuralNetworkException throws exception if neural network training fails.
     */
    private void trainIteration() throws MatrixException, IOException, NeuralNetworkException {
        long startTime = System.nanoTime();
        for (Connector connector : connectors) if (reset) connector.reset();
        Sequence inputSequence = new Sequence(sampleDepth);
        Sequence outputSequence = new Sequence(sampleDepth);
        trainingSampleAt = getSamples(trainIns, trainOuts, inputSequence, outputSequence, trainingSampleAt, trainingSamplesPerStep, trainingSampleInOrder, trainingShuffleSamples);
        getOutputLayer().resetError();
        getOutputLayer().setTargets(outputSequence);
        getInputLayer().train(inputSequence);
        getOutputLayer().backward();
        getInputLayer().update();
        long endTime = System.nanoTime();
        trainingTime += endTime - startTime;
        trainingError.report(getOutputLayer().getTotalError());
        trainingError.store(totalIterations, true);
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
        if (verboseTraining) verboneTrainingStatus();
        if (persistence != null) persistence.cycle();
    }

    /**
     * Verboses (prints to consolte) neural network training status.<br>
     * Prints number of iteration, neural network training time and training error.<br>
     *
     */
    private void verboneTrainingStatus() {
        if (totalIterations % verboseCycle == 0) System.out.println((neuralNetworkName != null ? neuralNetworkName + ": " : "") + "Training error (iteration #" + totalIterations +"): " + String.format("%.4f", trainingError.getAbsolute()) + ", Training time in seconds: " + (trainingTime / 1000000000));
    }

    /**
     * Returns sample batch for neural network training and validation purposes.
     *
     * @param inputs training or test sample input set.
     * @param outputs training or test sample output set.
     * @param inputSequence input sequence for training or validation.
     * @param outputSequence output sample batch for training or validation.
     * @param sampleAt starts batch sampling at this index.
     * @param samplesPerStep number of samples per batch.
     * @param sampleInOrder true if sampling is done sequentially from training or test sample set otherwise false.
     * @param shuffleSamples true if sample batch is shuffled otherwise original order is maintained.
     * @return sampleAt after sampling operation.
     */
    private int getSamples(LinkedHashMap<Integer, Matrix> inputs, LinkedHashMap<Integer, Matrix> outputs, Sequence inputSequence, Sequence outputSequence, int sampleAt, int samplesPerStep, boolean sampleInOrder, boolean shuffleSamples) throws MatrixException {
        int sampleEnd = sampleAt + samplesPerStep * sampleDepth;
        ArrayList<Integer> samples = new ArrayList<>();
        if (sampleInOrder) {
            if (samplesPerStep == 0) {
                sampleAt = 0;
                sampleEnd = inputs.size();
            }
            else {
                if (sampleEnd > inputs.size() - 1) {
                    sampleAt = 0;
                    sampleEnd = sampleAt + samplesPerStep * sampleDepth < inputs.size() ? sampleAt + samplesPerStep * sampleDepth : inputs.size() - 1;
                }
            }
            for (int sample = sampleAt; sample < sampleEnd; sample++) samples.add(sample);
            if (shuffleSamples) Collections.shuffle(samples);
        }
        else {
            for (int index = 0; index < samplesPerStep; index++) samples.add(random.nextInt(inputs.size()));
        }
        for (Integer sampleIndex : samples) {
            for (int matrixIndex = 0; matrixIndex < sampleDepth; matrixIndex++) {
                inputSequence.put(sampleIndex, matrixIndex, inputs.get(sampleIndex + matrixIndex));
                outputSequence.put(sampleIndex, matrixIndex, outputs.get(sampleIndex + matrixIndex));
            }
        }
        if (sampleInOrder && samplesPerStep > 0) {
            if (++sampleAt >= inputs.size() - 1 - samplesPerStep * sampleDepth) sampleAt = 0;
        }
        return sampleAt;
    }

    /**
     * Predicts using given test set inputs.
     *
     */
    private void predictInput() throws MatrixException {
        Sequence inputSequence = new Sequence(sampleDepth);
        int sampleIndex = 0;
        int matrixIndex = 0;
        for (Matrix input : testIns.values()) {
            inputSequence.put(sampleIndex, matrixIndex, input);
            if (++matrixIndex == sampleDepth) {
                sampleIndex++;
                matrixIndex = 0;
            }
        }
        getOutputLayer().clearTargets();
        getInputLayer().predict(inputSequence);
        stateCompleted();
    }

    /**
     * Validates with given test set inputs and outputs.
     *
     * @param stateCompleted if flag is sets calls stateCompleted function.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if validation fails.
     */
    private void validateInput(boolean stateCompleted) throws MatrixException, NeuralNetworkException {
        validationError.resetError();
        Sequence inputSequence = new Sequence(sampleDepth);
        Sequence outputSequence = new Sequence(sampleDepth);
        validationSampleAt = getSamples(testIns, testOuts, inputSequence, outputSequence, validationSampleAt, validationSamplesPerStep, validationSampleInOrder, validationShuffleSamples);
        getOutputLayer().clearTargets();
        validationError.report(getInputLayer().predict(inputSequence), outputSequence);
        if (verboseValidation && (totalIterations % verboseCycle == 0)) verboseValidationStatus();
        validationError.store(totalIterations, true);
        if (stateCompleted) stateCompleted();
    }

    /**
     * Sets neural network into completed state and makes state transition to idle state.
     *
     */
    private void stateCompleted() {
        lock.lock();
        executionState = ExecutionState.IDLE;
        complete.signal();
        lock.unlock();
    }

    /**
     * Checks if neural network is executing (processing).
     *
     * @return true if neural network is executing otherwise false.
     */
    public boolean isProcessing() {
        if (neuralNetworkThread == null) return false;
        boolean isProcessing;
        lock.lock();
        isProcessing = !(executionState == ExecutionState.IDLE || executionState == ExecutionState.TERMINATED);
        lock.unlock();
        return isProcessing;
    }

    /**
     * Waits for neural network to finalize it execution (processing).
     *
     */
    public void waitToComplete() {
        if (neuralNetworkThread == null) return;
        lock.lock();
        try {
            while (isProcessing()) complete.await();
        }
        catch (InterruptedException exception) {}
        lock.unlock();
    }

    /**
     * Checks if neural network execution (processing) is stopped.
     *
     * @return true if execution is stopped otherwise false.
     */
    private boolean stoppedExecution() {
        boolean stopped = false;
        stopLock.lock();
        if (stopExecution) {
            stopped = true;
            stopExecution = false;
        }
        stopLock.unlock();
        return stopped;
    }

    /**
     * Sets neural network training batch sampling parameters.
     *
     * @param samplesPerStep amount of samples to be included to training batch.
     * @param sampleInOrder true if sampling is done sequentially from training sample set otherwise false.
     * @param shuffleSamples true if sample batch is shuffled otherwise original order is maintained.
     * @throws NeuralNetworkException throws exception if sampling parameter setting fails.
     */
    public void setTrainingSampling(int samplesPerStep, boolean sampleInOrder, boolean shuffleSamples) throws NeuralNetworkException {
        waitToComplete();
        if (samplesPerStep < 0) throw new NeuralNetworkException("Sample size must be zero (all samples in step) or positive number.");
        trainingSamplesPerStep = samplesPerStep;
        trainingSampleInOrder = sampleInOrder;
        trainingShuffleSamples = shuffleSamples;
        trainingSampleAt = 0;
    }

    /**
     * Returns amount of samples to be included to training batch.
     *
     * @return amount of samples to be included to training batch.
     */
    public int getTrainingSamplesPerStep() {
        return trainingSamplesPerStep;
    }

    /**
     * Returns true if sampling is done sequentially from training sample set otherwise false.
     *
     * @return true if sampling is done sequentially from training sample set otherwise false.
     */
    public boolean getTrainingSampleInOrder() {
        return trainingSampleInOrder;
    }

    /**
     * Returns true if sample batch is shuffled otherwise original order is maintained.
     *
     * @return true if sample batch is shuffled otherwise original order is maintained.
     */
    public boolean getTrainingShuffleSamples() {
        return trainingShuffleSamples;
    }

    /**
     * Sets neural network validation batch sampling parameters.
     *
     * @param samplesPerStep amount of samples to be included to validation batch.
     * @param sampleInOrder true if sampling is done sequentially from validation sample set otherwise false.
     * @param shuffleSamples true if sample batch is shuffled otherwise original order is maintained.
     * @throws NeuralNetworkException throws exception if sampling parameter setting fails.
     */
    public void setValidationSampling(int samplesPerStep, boolean sampleInOrder, boolean shuffleSamples) throws NeuralNetworkException {
        waitToComplete();
        if (samplesPerStep < 0) throw new NeuralNetworkException("Sample size must be zero (all samples in step) or positive number.");
        validationSamplesPerStep = samplesPerStep;
        validationSampleInOrder = sampleInOrder;
        validationShuffleSamples = shuffleSamples;
        validationSampleAt = 0;
    }

    /**
     * Returns amount of samples to be included to validation batch.
     *
     * @return amount of samples to be included to validation batch.
     */
    public int getValidationSamplesPerStep() {
        return validationSamplesPerStep;
    }

    /**
     * Returns true if sampling is done sequentially from validation sample set otherwise false.
     *
     * @return true if sampling is done sequentially from validation sample set otherwise false.
     */
    public boolean getValidationSampleInOrder() {
        return validationSampleInOrder;
    }

    /**
     * Returns true if sample batch is shuffled otherwise original order is maintained.
     *
     * @return true if sample batch is shuffled otherwise original order is maintained.
     */
    public boolean getValidationShuffleSamples() {
        return validationShuffleSamples;
    }

    /**
     * Returns training error instance.
     *
     * @return training error instance.
     */
    public Metrics getTrainingError() {
        waitToComplete();
        return trainingError;
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
     * @return neural network output error.
     */
    public double getOutputError() {
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
        validationError = new Metrics(metricsType, multiClass);
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
        validationError = new Metrics(metricsType);
    }

    /**
     * Returns validation error instance.
     *
     * @return validation error instance.
     */
    public Metrics getValidationError() {
        waitToComplete();
        return validationError;
    }

    /**
     * Verboses validation error.
     *
     * @throws NeuralNetworkException throws exception of printing of validation error fails.
     */
    private void verboseValidationStatus() throws NeuralNetworkException {
        System.out.println((neuralNetworkName != null ? neuralNetworkName + ": " : "") + "Validating...");
        validationError.printReport();
    }

    /**
     * Verboses validation error.
     *
     * @throws NeuralNetworkException throws exception of printing of validation error fails.
     */
    public void printValidationReport() throws NeuralNetworkException {
        waitToComplete();
        verboseValidationStatus();
    }

    /**
     * Makes deep copy of neural network by using object serialization.
     *
     * @return copy of this neural network.
     * @throws IOException throws exception if copying of neural network fails.
     * @throws ClassNotFoundException throws exception if copying of neural network fails.
     */
    public NeuralNetwork copy() throws IOException, ClassNotFoundException {
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
     * Appends other neural network to this neural network by weight tau. Effectively appends each weight matrix of each connector by this weight factor.
     *
     * @param otherNeuralNetwork other neural network that contributes to this neural network.
     * @param tau tau which controls contribution of other connector.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void append(NeuralNetwork otherNeuralNetwork, double tau) throws MatrixException {
        for (int index = 0; index < connectors.size(); index++) {
            connectors.get(index).append(otherNeuralNetwork.getConnectors().get(index), tau);
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
        lock.lock();
        getOutputLayer().setImportanceSamplingWeights(importanceSamplingWeights);
        lock.unlock();
    }

}

