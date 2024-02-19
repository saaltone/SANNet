/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.network;

import core.activation.ActivationFunction;
import core.layer.*;
import core.loss.LossFunction;
import core.loss.LossFunctionType;
import utils.configurable.DynamicParamException;
import utils.matrix.Initialization;
import utils.matrix.MatrixException;

import java.util.TreeMap;

/**
 * Defines configuration (layers and how they are connected to each other) for neural network.<br>
 *
 */
public class NeuralNetworkConfiguration {

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
     * Cumulating neural network layer index count.
     *
     */
    private int neuralNetworkLayerIndexCount = 0;

    /**
     * Default constructor for neural network configuration.
     *
     */
    public NeuralNetworkConfiguration() {
    }

    /**
     * Adds input layer to neural network.
     *
     * @param params parameters for input layer.
     * @return neural network layer index.
     * @throws NeuralNetworkException throws neural network exception if adding of input layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public int addInputLayer(String params) throws NeuralNetworkException, DynamicParamException {
        return addInputLayer(-1 , params);
    }

    /**
     * Adds input layer to neural network.
     *
     * @param inputLayerGroupID input layer group ID.
     * @param params parameters for input layer.
     * @return neural network layer index.
     * @throws NeuralNetworkException throws neural network exception if adding of input layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public int addInputLayer(int inputLayerGroupID, String params) throws NeuralNetworkException, DynamicParamException {
        int neuralNetworkLayerIndex = getNextNeuralNetworkLayerIndex();
        int currentInputLayerGroupId = inputLayerGroupID > -1 ? inputLayerGroupID : 0;

        InputLayer inputLayer = new InputLayer(neuralNetworkLayerIndex, currentInputLayerGroupId, params);
        int inputLayerID = inputLayers.size();
        inputLayers.put(inputLayerID, inputLayer);

        TreeMap<Integer, InputLayer> inputLayerGroup;
        if (inputLayerGroups.containsKey(currentInputLayerGroupId)) inputLayerGroup = inputLayerGroups.get(currentInputLayerGroupId);
        else inputLayerGroups.put(currentInputLayerGroupId, inputLayerGroup = new TreeMap<>());
        inputLayerGroup.put(inputLayerGroup.size(), inputLayer);

        neuralNetworkLayers.put(neuralNetworkLayers.size(), inputLayer);
        return neuralNetworkLayerIndex;
    }

    /**
     * Returns input layers.
     *
     * @return input layers.
     */
    public TreeMap<Integer, InputLayer> getInputLayers() {
        return new TreeMap<>() {{ putAll(inputLayers); }};
    }

    /**
     * Returns input layer groups.
     *
     * @return input layer groups.
     */
    public TreeMap<Integer, TreeMap<Integer, InputLayer>> getInputLayerGroups() {
        return new TreeMap<>() {{ putAll(inputLayerGroups); }};
    }

    /**
     * Adds hidden layer to neural network. Layers are executed in order which they are added.
     *
     * @param layerType type of layer.
     * @return neural network layer index.
     * @throws NeuralNetworkException throws neural network exception if adding of layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public int addHiddenLayer(LayerType layerType) throws NeuralNetworkException, DynamicParamException, MatrixException {
        return addHiddenLayer(layerType, null, null, null);
    }

    /**
     * Adds hidden layer to neural network. Layers are executed in order which they are added.
     *
     * @param layerType type of layer.
     * @param params parameters for layer.
     * @return neural network layer index.
     * @throws NeuralNetworkException throws neural network exception if adding of layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public int addHiddenLayer(LayerType layerType, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        return addHiddenLayer(layerType, null, null, params);
    }

    /**
     * Adds hidden layer to neural network. Layers are executed in order which they are added.
     *
     * @param layerType type of layer.
     * @param activationFunction activation function for layer.
     * @return neural network layer index.
     * @throws NeuralNetworkException throws neural network exception if adding of layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public int addHiddenLayer(LayerType layerType, ActivationFunction activationFunction) throws NeuralNetworkException, DynamicParamException, MatrixException {
        return addHiddenLayer(layerType, activationFunction, null, null);
    }

    /**
     * Adds hidden layer to neural network. Layers are executed in order which they are added.
     *
     * @param layerType type of layer.
     * @param activationFunction activation function for layer.
     * @param params parameters for layer.
     * @return neural network layer index.
     * @throws NeuralNetworkException throws neural network exception if adding of layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public int addHiddenLayer(LayerType layerType, ActivationFunction activationFunction, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        return addHiddenLayer(layerType, activationFunction, null, params);
    }

    /**
     * Adds hidden layer to neural network. Layers are executed in order which they are added.
     *
     * @param layerType type of layer.
     * @param activationFunction activation function for layer.
     * @param initialization layer parameter initialization function for layer.
     * @return neural network layer index.
     * @throws NeuralNetworkException throws neural network exception if adding of layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public int addHiddenLayer(LayerType layerType, ActivationFunction activationFunction, Initialization initialization) throws NeuralNetworkException, DynamicParamException, MatrixException {
        return addHiddenLayer(layerType, activationFunction, initialization, null);
    }

    /**
     * Adds hidden layer to neural network. Layers are executed in order which they are added.
     *
     * @param layerType type of layer.
     * @param initialization layer parameter initialization function for layer.
     * @return neural network layer index.
     * @throws NeuralNetworkException throws neural network exception if adding of layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public int addHiddenLayer(LayerType layerType, Initialization initialization) throws NeuralNetworkException, DynamicParamException, MatrixException {
        return addHiddenLayer(layerType, initialization, null);
    }

    /**
     * Adds hidden layer to neural network. Layers are executed in order which they are added.
     *
     * @param layerType type of layer.
     * @param initialization layer parameter initialization function for layer.
     * @param params parameters for layer.
     * @return neural network layer index.
     * @throws NeuralNetworkException throws neural network exception if adding of layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public int addHiddenLayer(LayerType layerType, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        return addHiddenLayer(layerType, null, initialization, params);
    }

    /**
     * Adds hidden layer to neural network. Layers are executed in order which they are added.
     *
     * @param layerType type of layer.
     * @param activationFunction activation function for layer.
     * @param initialization layer parameter initialization function for layer.
     * @param params parameters for layer.
     * @return neural network layer index.
     * @throws NeuralNetworkException throws neural network exception if adding of layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public int addHiddenLayer(LayerType layerType, ActivationFunction activationFunction, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        int neuralNetworkLayerIndex = getNextNeuralNetworkLayerIndex();
        AbstractLayer hiddenLayer = LayerFactory.create(neuralNetworkLayerIndex, layerType, activationFunction, initialization, params);
        hiddenLayers.put(hiddenLayers.size(), hiddenLayer);
        neuralNetworkLayers.put(neuralNetworkLayers.size(), hiddenLayer);
        return neuralNetworkLayerIndex;
    }

    /**
     * Returns hidden layers.
     *
     * @return hidden layers.
     */
    public TreeMap<Integer, AbstractLayer> getHiddenLayers() {
        return new TreeMap<>() {{ putAll(hiddenLayers); }};
    }

    /**
     * Adds output layer to neural network.
     *
     * @param lossFunctionType loss function type for output layer.
     * @return neural network layer index.
     * @throws NeuralNetworkException throws neural network exception if adding of output layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public int addOutputLayer(LossFunctionType lossFunctionType) throws NeuralNetworkException, DynamicParamException, MatrixException {
        return addOutputLayer(lossFunctionType, null);
    }

    /**
     * Adds output layer to neural network.
     *
     * @param lossFunctionType loss function type for output layer.
     * @param params parameters for loss function.
     * @return neural network layer index.
     * @throws NeuralNetworkException throws neural network exception if adding of output layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public int addOutputLayer(LossFunctionType lossFunctionType, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        return addOutputLayer(-1, lossFunctionType, params);
    }

    /**
     * Adds output layer to neural network.
     *
     * @param outputLayerGroupID output layer group ID.
     * @param lossFunctionType loss function type for output layer.
     * @return neural network layer index.
     * @throws NeuralNetworkException throws neural network exception if adding of output layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public int addOutputLayer(int outputLayerGroupID, LossFunctionType lossFunctionType) throws NeuralNetworkException, DynamicParamException, MatrixException {
        return addOutputLayer(outputLayerGroupID, lossFunctionType, null);
    }

    /**
     * Adds output layer to neural network.
     *
     * @param outputLayerGroupID output layer group ID.
     * @param lossFunctionType loss function type for output layer.
     * @param params parameters for loss function.
     * @return neural network layer index.
     * @throws NeuralNetworkException throws neural network exception if adding of output layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if custom function is attempted to be created with this constructor.
     */
    public int addOutputLayer(int outputLayerGroupID, LossFunctionType lossFunctionType, String params) throws NeuralNetworkException, DynamicParamException, MatrixException {
        return addOutputLayer(outputLayerGroupID, new LossFunction(lossFunctionType, params), params);
    }

    /**
     * Adds output layer to neural network.
     *
     * @param lossFunction loss function for output layer.
     * @return neural network layer index.
     * @throws NeuralNetworkException throws neural network exception if adding of output layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public int addOutputLayer(LossFunction lossFunction) throws NeuralNetworkException, DynamicParamException {
        return addOutputLayer(lossFunction, null);
    }

    /**
     * Adds output layer to neural network.
     *
     * @param lossFunction loss function for output layer.
     * @param params parameters for layer.
     * @return neural network layer index.
     * @throws NeuralNetworkException throws neural network exception if adding of output layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public int addOutputLayer(LossFunction lossFunction, String params) throws NeuralNetworkException, DynamicParamException {
        return addOutputLayer(-1, lossFunction, params);
    }

    /**
     * Adds output layer to neural network.
     *
     * @param outputLayerGroupID output layer group ID.
     * @param lossFunction loss function for output layer.
     * @return neural network layer index.
     * @throws NeuralNetworkException throws neural network exception if adding of output layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public int addOutputLayer(int outputLayerGroupID, LossFunction lossFunction) throws NeuralNetworkException, DynamicParamException {
        return addOutputLayer(outputLayerGroupID, lossFunction, null);
    }

    /**
     * Adds output layer to neural network.
     *
     * @param outputLayerGroupID output layer group ID.
     * @param lossFunction loss function for output layer.
     * @param params parameters for layer.
     * @return neural network layer index.
     * @throws NeuralNetworkException throws neural network exception if adding of output layer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public int addOutputLayer(int outputLayerGroupID, LossFunction lossFunction, String params) throws NeuralNetworkException, DynamicParamException {
        int currentOutputLayerGroupId = outputLayerGroupID > -1 ? outputLayerGroupID : 0;
        int neuralNetworkLayerIndex = getNextNeuralNetworkLayerIndex();

        OutputLayer outputLayer = new OutputLayer(neuralNetworkLayerIndex, currentOutputLayerGroupId, lossFunction, params);

        int outputLayerID = outputLayers.size();
        outputLayers.put(outputLayerID, outputLayer);

        TreeMap<Integer, OutputLayer> outputLayerGroup;
        if (outputLayerGroups.containsKey(currentOutputLayerGroupId)) outputLayerGroup = outputLayerGroups.get(currentOutputLayerGroupId);
        else outputLayerGroups.put(currentOutputLayerGroupId, outputLayerGroup = new TreeMap<>());
        outputLayerGroup.put(outputLayerID, outputLayer);

        neuralNetworkLayers.put(neuralNetworkLayers.size(), outputLayer);
        return neuralNetworkLayerIndex;
    }

    /**
     * Returns next neural network layer index.
     *
     * @return next neural network layer index.
     */
    private int getNextNeuralNetworkLayerIndex() {
        return neuralNetworkLayerIndexCount++;
    }

    /**
     * Returns output layers.
     *
     * @return output layers.
     */
    public TreeMap<Integer, OutputLayer> getOutputLayers() {
        return new TreeMap<>() {{ putAll(outputLayers); }};
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
        return new TreeMap<>() {{ putAll(neuralNetworkLayers); }};
    }

    /**
     * Connects previous and next layers.
     *
     * @param previousLayer previous layer.
     * @param nextLayer next layer.
     * @throws NeuralNetworkException throws exception if next layer has already previous layer and cannot have multiple previous layers.
     */
    public void connectLayers(NeuralNetworkLayer previousLayer, NeuralNetworkLayer nextLayer) throws NeuralNetworkException {
        if (nextLayer.hasPreviousLayers() && !nextLayer.canHaveMultiplePreviousLayers()) throw new NeuralNetworkException("Next layer has already previous layer and cannot have multiple previous layers.");
        previousLayer.addNextLayer(nextLayer);
        nextLayer.addPreviousLayer(previousLayer);
    }

    /**
     * Connects previous and next layers.
     *
     * @param previousLayerIndex previous layer index.
     * @param nextLayerIndex next layer index.
     * @throws NeuralNetworkException throws exception if next layer has already previous layer and cannot have multiple previous layers.
     */
    public void connectLayers(int previousLayerIndex, int nextLayerIndex) throws NeuralNetworkException {
        connectLayers(neuralNetworkLayers.get(previousLayerIndex), neuralNetworkLayers.get(nextLayerIndex));
    }

    /**
     * Connects neural network layers serially in order.
     *
     * @throws NeuralNetworkException throws exception if next layer has already previous layer and cannot have multiple previous layers.
     */
    public void connectLayersSerially() throws NeuralNetworkException {
        NeuralNetworkLayer previousNeuralNetworkLayer = null;
        for (NeuralNetworkLayer nextNeuralNetworkLayer : getNeuralNetworkLayers().values()) {
            if (previousNeuralNetworkLayer != null) connectLayers(previousNeuralNetworkLayer, nextNeuralNetworkLayer);
            previousNeuralNetworkLayer = nextNeuralNetworkLayer;
        }
    }

    /**
     * Validates neural network configuration.<br>
     * Checks that all layer are connected properly.<br>
     *
     * @throws NeuralNetworkException thrown if validation of neural network configuration fails.
     */
    public void validate() throws NeuralNetworkException {
        for (InputLayer inputLayer : inputLayers.values()) if (!inputLayer.hasNextLayers()) throw new NeuralNetworkException("Input layer #" + inputLayer.getLayerIndex() + " does not have next layer.");
        for (AbstractLayer hiddenLayer : hiddenLayers.values()) {
            if (!hiddenLayer.hasPreviousLayers()) throw new NeuralNetworkException("Hidden layer #" + hiddenLayer.getLayerIndex() + " does not have previous layer.");
            if (!hiddenLayer.hasNextLayers()) throw new NeuralNetworkException("Hidden layer #" + hiddenLayer.getLayerIndex() + " does not have next layer.");
        }
        for (OutputLayer outputLayer : outputLayers.values()) if (!outputLayer.hasPreviousLayers()) throw new NeuralNetworkException("Input layer #" + outputLayer.getLayerIndex() + " does not have previous layer.");

        checkLayerCompatibility(neuralNetworkLayers);

        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers.values()) neuralNetworkLayer.initializeDimensions();
    }

    /**
     * Checks that neural network layers are compatible with each other.
     *
     * @param neuralNetworkLayers neural network layers
     * @throws NeuralNetworkException throws exception if neural network layers are not compatible with each other
     */
    private void checkLayerCompatibility(TreeMap<Integer, NeuralNetworkLayer> neuralNetworkLayers) throws NeuralNetworkException {
        boolean hasRecurrentLayers = false;
        for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers.values()) {
            if (neuralNetworkLayer.isRecurrentLayer()) {
                hasRecurrentLayers = true;
                break;
            }
        }
        if (hasRecurrentLayers) {
            for (NeuralNetworkLayer neuralNetworkLayer : neuralNetworkLayers.values()) {
                if (!neuralNetworkLayer.worksWithRecurrentLayer()) {
                    throw new NeuralNetworkException(LayerFactory.getLayerTypeByName(neuralNetworkLayer) + " layer does not work with recurrent layers.");
                }
            }
        }
    }

}
