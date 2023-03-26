/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer;

import core.network.NeuralNetworkException;
import core.optimization.Optimizer;
import utils.configurable.DynamicParamException;
import utils.sampling.Sequence;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeMap;

/**
 * Interface for neural network layer.<br>
 *
 */
public interface NeuralNetworkLayer {

    /**
     * Returns index of a layer.
     *
     * @return index of a layer.
     */
    int getLayerIndex();

    /**
     * Checks if layer can have multiple previous layers.
     *
     * @return  if true layer can have multiple previous layers otherwise false.
     */
    boolean canHaveMultiplePreviousLayers();

    /**
     * Adds reference to next neural network layer.
     *
     * @param nextLayer reference to next neural network layer.
     * @throws NeuralNetworkException throws exception if next layer is attempted to be added to output layer.
     */
    void addNextLayer(NeuralNetworkLayer nextLayer) throws NeuralNetworkException;

    /**
     * Returns references to next layers.
     *
     * @return references to next layers.
     */
    TreeMap<Integer, NeuralNetworkLayer> getNextLayers();

    /**
     * Returns if layer has next layers.
     *
     * @return true if layer has next layers otherwise false.
     */
    boolean hasNextLayers();

    /**
     * Removes next neural network layer
     *
     * @param neuralNetworkLayer neural network layer.
     * @throws NeuralNetworkException throws exception if next neural network layer is not found.
     */
    void removeNextLayer(NeuralNetworkLayer neuralNetworkLayer) throws NeuralNetworkException;

    /**
     * Adds reference to previous neural network layer.
     *
     * @param previousLayer reference to previous neural network layer.
     * @throws NeuralNetworkException throws exception if previous layer is attempted to be added to input layer.
     */
    void addPreviousLayer(NeuralNetworkLayer previousLayer) throws NeuralNetworkException;

    /**
     * Returns references to previous neural network layers.
     *
     * @return references to previous neural network layers.
     */
    TreeMap<Integer, NeuralNetworkLayer> getPreviousLayers();

    /**
     * Returns if layer has previous layers.
     *
     * @return true if layer has previous layer otherwise false.
     */
    boolean hasPreviousLayers();

    /**
     * Removes previous neural network layer
     *
     * @param neuralNetworkLayer neural network layer.
     * @throws NeuralNetworkException throws exception if previous neural network layer is not found.
     */
    void removePreviousLayer(NeuralNetworkLayer neuralNetworkLayer) throws NeuralNetworkException;

    /**
     * Returns width of neural network layer.
     *
     * @return width of neural network layer.
     */
    int getLayerWidth();

    /**
     * Returns height of neural network layer.
     *
     * @return height of neural network layer.
     */
    int getLayerHeight();

    /**
     * Returns depth of neural network layer. Relevant for convolutional layers.
     *
     * @return depth of neural network layer.
     */
    int getLayerDepth();

    /**
     * Checks if layer is recurrent layer type.
     *
     * @return true if layer is recurrent layer otherwise false.
     */
    boolean isRecurrentLayer();

    /**
     * Checks if layer works with recurrent layers.
     *
     * @return if true layer works with recurrent layers otherwise false.
     */
    boolean worksWithRecurrentLayer();

    /**
     * Initializes neural network layer dimensions.
     *
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     */
    void initializeDimensions() throws NeuralNetworkException;

    /**
     * Sets reset flag for procedure expression dependencies.
     *
     * @param resetDependencies if true procedure expression dependencies are reset otherwise false.
     */
    void resetDependencies(boolean resetDependencies);

    /**
     * Resets layer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void reset() throws MatrixException;

    /**
     * Reinitializes neural network layer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void reinitialize() throws MatrixException;

    /**
     * Returns outputs of neural network layer.
     *
     * @return outputs of neural network layer.
     */
    Sequence getLayerOutputs();

    /**
     * Returns neural network layer input gradients.
     *
     * @return neural network layer input gradients.
     */
    Sequence getLayerOutputGradients();

    /**
     * Returns weights for normalization.
     *
     * @return weights for normalization.
     */
    HashSet<Matrix> getNormalizedWeights();

    /**
     * Returns weights for regularization.
     *
     * @return weights for regularization.
     */
    HashSet<Matrix> getRegularizedWeights();

    /**
     * Returns neural network weight gradients.
     *
     * @return neural network weight gradients.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    HashMap<Matrix, Matrix> getLayerWeightGradients() throws MatrixException;

    /**
     * Starts neural network layer and it's execution thread.
     *
     * @throws NeuralNetworkException throws exception if neural network layer name cannot be returned.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void start() throws NeuralNetworkException, MatrixException, DynamicParamException;

    /**
     * Stops neural network layer and terminates neural network layer execution thread.<br>
     * Sets layer state to TERMINATED.<br>
     *
     */
    void stop();

    /**
     * Executes training step for neural network layer and propagates procedure to next layer.
     *
     * @param inputs training inputs for layer.
     */
    void train(Sequence inputs);

    /**
     * Executes training step for neural network layer and propagates procedure to next layer.<br>
     * Uses existing training inputs and outputs.<br>
     *
     */
    void train();

    /**
     * Executes predict step for neural network layer and propagates procedure to next layer.
     *
     * @param inputs predict inputs for layer.
     */
    void predict(Sequence inputs);

    /**
     * Executes predict step for neural network layer and propagates procedure to next layer.<br>
     * Uses existing testing inputs.<br>
     *
     */
    void predict();

    /**
     * Executes backward (gradient) propagation phase for training step of neural network layer.
     *
     * @throws NeuralNetworkException throws exception if backward operation fails.
     */
    void backward() throws NeuralNetworkException;

    /**
     * Executes parameter (weight) update for training step of neural network layer.
     *
     */
    void update();

    /**
     * Waits for layer to complete execution.
     *
     */
    void waitToComplete();

    /**
     * Cumulates error from regularization. Mainly from L1 / L2 / Lp regularization.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return cumulated error from regularization.
     */
    double error() throws MatrixException, DynamicParamException;

    /**
     * Executes forward processing step of layer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void forwardProcess() throws MatrixException, DynamicParamException;

    /**
     * Executes backward processing step of layer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void backwardProcess() throws MatrixException, DynamicParamException;

    /**
     * Sets optimizer for layer.<br>
     * Optimizer optimizes weight parameters iteratively towards optimal solution.<br>
     *
     * @param optimizer optimizer to be added.
     */
    void setOptimizer(Optimizer optimizer);

    /**
     * Resets optimizer of layer.
     *
     */
    void resetOptimizer();

    /**
     * Returns map of weights.
     *
     * @return map of weights.
     */
    HashMap<Integer, Matrix> getWeightsMap();

    /**
     * Appends other neural network layer with equal weights to this layer by weighting factor tau.
     *
     * @param otherNeuralNetworkLayer other neural network layer.
     * @param tau tau which controls contribution of other layer.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void append(NeuralNetworkLayer otherNeuralNetworkLayer, double tau) throws MatrixException;

    /**
     * Prints structure and metadata of neural network layer.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    void print() throws NeuralNetworkException;

    /**
     * Prints expression chains of neural network.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    void printExpressions() throws NeuralNetworkException;

    /**
     * Prints gradient chains of neural network.
     *
     * @throws NeuralNetworkException throws exception if printing of neural network fails.
     */
    void printGradients() throws NeuralNetworkException;

}
