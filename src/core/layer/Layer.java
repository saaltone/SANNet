/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.layer;

import core.NeuralNetworkException;
import utils.matrix.Init;
import utils.matrix.MatrixException;

/**
 * Interface for neural network layer.
 *
 */
public interface Layer {

    /**
     * Returns used initialization function.
     *
     * @return used initialization function.
     */
    Init getInitialization();

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
     * Sets if recurrent inputs of layer are allowed to be reset.
     *
     * @param allowLayerReset if true allows reset.
     */
    void setAllowLayerReset(boolean allowLayerReset);

    /**
     * Initializes neural network layer.<br>
     * Initializes weight and bias and their gradients.<br>
     * Initializes needed execution parameters for neural network layer.<br>
     *
     * @throws MatrixException throws exception is matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    void initialize() throws MatrixException, NeuralNetworkException;

    /**
     * Executes forward processing step of layer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    void forwardProcess() throws MatrixException, NeuralNetworkException;

    /**
     * Executes backward processing step of layer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void backwardProcess() throws MatrixException;

    /**
     * Returns layer type by name
     *
     * @return layer type by name
     * @throws NeuralNetworkException throws exception if layer is of an unknown type.
     */
    String getTypeByName() throws NeuralNetworkException;

}
