/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.layer;

import core.NeuralNetworkException;
import utils.Init;
import utils.Matrix;
import utils.MatrixException;

import java.util.TreeMap;

/**
 * Interface for neural network layer.
 *
 */
public interface Layer {

    /**
     * Sets forward connector with link to next neural network layer.
     *
     * @param forward reference to forward connector.
     */
    void setForward(Connector forward);

    /**
     * Sets backward connector with link to previous neural network layer.
     *
     * @param backward reference to backward connector.
     */
    void setBackward(Connector backward);

    /**
     * Sets width of the neural network layer.
     *
     * @param width width of neural network layer.
     */
    void setWidth(int width);

    /**
     * Gets width of neural network layer.
     *
     * @return width of neural network layer.
     */
    int getWidth();

    /**
     * Sets height of the neural network layer. Relevant for convolutional layers.
     *
     * @param height height of neural network layer.
     */
    void setHeight(int height);

    /**
     * Gets height of neural network layer. Relevant for convolutional layers.
     *
     * @return height of neural network layer.
     */
    int getHeight();

    /**
     * Sets depth of the neural network layer. Relevant for convolutional layers.
     *
     * @param depth depth of neural network layer.
     */
    void setDepth(int depth);

    /**
     * Gets depth of neural network layer. Relevant for convolutional layers.
     *
     * @return depth of neural network layer.
     */
    int getDepth();

    /**
     * Gets used initialization function.
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
     * Gets outputs of neural network layer.
     *
     * @param outs outputs of neural network layer.
     * @return outputs of neural network layer.
     */
    TreeMap<Integer, Matrix> getOuts(TreeMap<Integer, Matrix> outs);

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
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    void initialize() throws NeuralNetworkException;

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
