/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.layer;

import core.network.NeuralNetworkException;
import core.normalization.NormalizationType;
import core.optimization.Optimizer;
import core.regularization.RegularizationType;
import utils.DynamicParamException;
import utils.Sequence;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashMap;

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
     * Returns reference to next layer.
     *
     * @return reference to next layer.
     */
    NeuralNetworkLayer getNextLayer();

    /**
     * Returns if layer has next layer.
     *
     * @return true if layer has next layer otherwise false.
     */
    boolean hasNextLayer();

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
     * Returns if layer has previous layer.
     *
     * @return true if layer has previous layer otherwise false.
     */
    boolean hasPreviousLayer();

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
     * @return true if layer if recurrent layer otherwise false.
     */
    boolean isRecurrentLayer();

    /**
     * Checks if layer is convolutional layer type.
     *
     * @return true if layer if convolutional layer otherwise false.
     */
    boolean isConvolutionalLayer();

    /**
     * Sets if recurrent inputs of layer are allowed to be reset during training.
     *
     * @param resetStateTraining if true allows reset.
     */
    void setResetStateTraining(boolean resetStateTraining);

    /**
     * Sets if recurrent inputs of layer are allowed to be reset during testing.
     *
     * @param resetStateTesting if true allows reset.
     */
    void setResetStateTesting(boolean resetStateTesting);

    /**
     * Sets if recurrent inputs of layer are allowed to be restored during training.
     *
     * @param restoreStateTraining if true allows restore.
     */
    void setRestoreStateTraining(boolean restoreStateTraining);

    /**
     * Sets if recurrent inputs of layer are allowed to be restored during testing.
     *
     * @param restoreStateTesting if true allows restore.
     */
    void setRestoreStateTesting(boolean restoreStateTesting);

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
     * Initializes neural network layer.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    void initialize() throws NeuralNetworkException;

    /**
     * Reinitializes neural network layer.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void reinitialize() throws NeuralNetworkException, MatrixException;

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
     * Resets normalizers and optimizer of layer.
     *
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    void reset() throws MatrixException;

    /**
     * Adds regularization method for layer.
     *
     * @param regularizationType regularization method.
     * @throws NeuralNetworkException throws exception if adding of regularizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void addRegularization(RegularizationType regularizationType) throws NeuralNetworkException, DynamicParamException;

    /**
     * Adds regularization method for layer.
     *
     * @param regularizationType regularization method.
     * @param params parameters for regularizer.
     * @throws NeuralNetworkException throws exception if adding of regularizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void addRegularization(RegularizationType regularizationType, String params) throws NeuralNetworkException, DynamicParamException;

    /**
     * Removes any regularization from layer.
     *
     */
    void removeRegularization();

    /**
     * Removes specific regularization from layer.
     *
     * @param regularizationType regularization method to be removed.
     * @throws NeuralNetworkException throws exception if removal of regularizer fails.
     */
    void removeRegularization(RegularizationType regularizationType) throws NeuralNetworkException;

    /**
     * Adds normalization method for layer.
     *
     * @param normalizationType normalization method.
     * @throws NeuralNetworkException throws exception if adding of normalizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void addNormalization(NormalizationType normalizationType) throws NeuralNetworkException, DynamicParamException;

    /**
     * Adds normalization method for layer.
     *
     * @param normalizationType normalization method.
     * @param params parameters for normalizer.
     * @throws NeuralNetworkException throws exception if adding of normalizer fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void addNormalization(NormalizationType normalizationType, String params) throws NeuralNetworkException, DynamicParamException;

    /**
     * Removes any normalization from layer.
     *
     */
    void removeNormalization();

    /**
     * Removes specific normalization from layer.
     *
     * @param normalizationType normalization method to be removed.
     * @throws NeuralNetworkException throws exception if removal of normalizer fails.
     */
    void removeNormalization(NormalizationType normalizationType) throws NeuralNetworkException;

    /**
     * Resets specific normalization for layer.
     *
     * @param normalizationType normalization method to be reset.
     * @throws NeuralNetworkException throws exception if reset of normalizer fails.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    void resetNormalization(NormalizationType normalizationType) throws NeuralNetworkException, MatrixException;

    /**
     * Resets all normalization for layer.
     *
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    void resetNormalization() throws MatrixException;

    /**
     * Reinitializes specific normalization for layer.
     *
     * @param normalizationType normalization method to be reinitialized.
     * @throws NeuralNetworkException throws exception if reinitialization of normalizer fails.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    void reinitializeNormalization(NormalizationType normalizationType) throws NeuralNetworkException, MatrixException;

    /**
     * Resets all normalization for layer.
     *
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    void reinitializeNormalization() throws MatrixException;

    /**
     * Sets optimizer for layer.<br>
     * Optimizer optimizes weight parameters iteratively towards optimal solution.<br>
     *
     * @param optimizer optimizer to be added.
     */
    void setOptimizer(Optimizer optimizer);

    /**
     * Resets optimizer for layer.
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
