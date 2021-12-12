/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
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

/**
 * Interface for neural network layer.<br>
 *
 */
public interface NeuralNetworkLayer {

    /**
     * Sets reference to next neural network layer.
     *
     * @param nextLayer reference to next neural network layer.
     */
    void setNextLayer(NeuralNetworkLayer nextLayer);

    /**
     * Sets reference to previous neural network layer.
     *
     * @param previousLayer reference to previous neural network layer.
     */
    void setPreviousLayer(NeuralNetworkLayer previousLayer);

    /**
     * Returns reference to previous neural network layer.
     *
     * @return reference to previous neural network layer.
     */
    NeuralNetworkLayer getPreviousLayer();

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
     * Check if layer is bidirectional.
     *
     * @return true if layer is bidirectional otherwise returns false.
     */
    boolean isBidirectional();

    /**
     * Checks if layer is convolutional layer type.
     *
     * @return true if layer if convolutional layer otherwise false.
     */
    boolean isConvolutionalLayer();

    /**
     * Initializes neural network layer dimensions.
     *
     * @throws NeuralNetworkException thrown if initialization of layer fails.
     */
    void initializeDimensions() throws NeuralNetworkException;

    /**
     * Reinitializes neural network layer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void reinitialize() throws MatrixException;

    /**
     * Returns output of neural network.
     *
     * @return output of neural network.
     */
    Sequence getOutput();

    /**
     * Returns outputs of neural network layer.
     *
     * @return outputs of neural network layer.
     */
    Sequence getLayerOutputs();

    /**
     * Returns neural network layer gradients.
     *
     * @return neural network layer gradients.
     */
    Sequence getLayerGradients();

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
     * @return output of next layer or this layer if next layer does not exist.
     */
    Sequence predict(Sequence inputs);

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
     * Cumulates error from regularization. Mainly from L1 / L2 / Lp regularization.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return cumulated error from regularization.
     */
    double error() throws MatrixException, DynamicParamException;

    /**
     * Marks state completed and propagates information to forward or backward direction depending on given flag.
     *
     * @param forwardDirection if true propagates state completion signal to forward direction otherwise propagates to backward direction.
     */
    void stateCompleted(boolean forwardDirection);

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
