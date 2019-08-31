/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.layer;

import core.NeuralNetworkException;
import core.activation.ActivationFunction;
import core.activation.ActivationFunctionType;
import utils.*;

import java.io.Serializable;
import java.util.HashMap;
import java.util.TreeMap;

/**
 * Abstract class that implements execution layer for actual neural network layers (feed forward layer, recurrent layer etc.)<br>
 * Provides supportive functions for actual neural network layers.<br>
 *
 */
public abstract class AbstractExecutionLayer implements Layer, Serializable {

    private static final long serialVersionUID = -2696526850302490503L;

    /**
     * Reference to connector between this and previous layer.
     *
     */
    protected Connector backward;

    /**
     * Reference to connector between this and previous layer.
     *
     */
    protected Connector forward;

    /**
     * Reference to parent layer that handles neural network layer state handling and initiates primary functions (train, predict, validate etc.)
     *
     */
    protected final AbstractLayer parent;

    /**
     * Width of neural network layer. Also known as number of neural network layer nodes.
     *
     */
    private int width;

    /**
     * Height of neural network layer. Relevant for convolutional layers.
     *
     */
    private int height = 1;

    /**
     * Depth of neural network layer. Relevant for convolutional layers.
     *
     */
    private int depth = 1;

    /**
     * Activation function for neural network layer.
     *
     */
    protected ActivationFunction activation = new ActivationFunction(ActivationFunctionType.ELU);

    /**
     * Initialization function for neural network layer.
     *
     */
    protected Init initialization = Init.UNIFORM_XAVIER;

    /**
     * Supportive ones matrix for layer processing.
     *
     */
    private Matrix ones;

    /**
     * Supportive identity matrix for layer processing.
     *
     */
    private Matrix I;

    /**
     * Constructor for abstract execution layer.
     *
     * @param parent reference to parent abstract layer.
     * @param activation activation function.
     * @param initialization initialization function.
     * @param params parameters for neural network layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected AbstractExecutionLayer(AbstractLayer parent, ActivationFunction activation, Init initialization, String params) throws DynamicParamException {
        this.parent = parent;
        if (activation != null) this.activation = activation;
        if (initialization != null) this.initialization = initialization;

        if (params != null) setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Gets parameters used for layer.<br>
     * Implemented by actual neural network layer with parameters specific to that layer.<br>
     *
     * @return parameters used for recurrent layer.
     */
    protected abstract HashMap<String, DynamicParam.ParamType> getParamDefs();

    /**
     * Sets parameters used for layer.<br>
     * Implemented by actual neural network layer with parameters specific to that layer.<br>
     *
     * @param params parameters used for recurrent layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    protected abstract void setParams(DynamicParam params) throws DynamicParamException;

    public String getTypeByName() throws NeuralNetworkException  {
        return LayerFactory.getLayerTypeByName(this);
    }

    /**
     * Sets forward connector with link to next neural network layer.
     *
     * @param forward reference to forward connector.
     */
    public void setForward(Connector forward) {
        this.forward = forward;
    }

    /**
     * Sets backward connector with link to previous neural network layer.
     *
     * @param backward reference to backward connector.
     */
    public void setBackward(Connector backward) {
        this.backward = backward;
    }

    /**
     * Sets width of the neural network layer.
     *
     * @param width width of neural network layer.
     */
    public void setWidth(int width) { this.width = width; }

    /**
     * Gets width of neural network layer.
     *
     * @return width of neural network layer.
     */
    public int getWidth() {
        return !flattenedOutput() ? width : width * height * depth;
    }

    /**
     * Sets height of the neural network layer. Relevant for convolutional layers.
     *
     * @param height height of neural network layer.
     */
    public void setHeight(int height) { this.height = height; }

    /**
     * Gets height of neural network layer. Relevant for convolutional layers.
     *
     * @return height of neural network layer.
     */
    public int getHeight() {
        return height;
    }

    /**
     * Sets depth of the neural network layer. Relevant for convolutional layers.
     *
     * @param depth depth of neural network layer.
     */
    public void setDepth(int depth) { this.depth = depth; }

    /**
     * Gets depth of neural network layer. Relevant for convolutional layers.
     *
     * @return depth of neural network layer.
     */
    public int getDepth() {
        return depth;
    }

    /**
     * Gets used initialization function.
     *
     * @return used initialization function.
     */
    public Init getInitialization() {
        return initialization;
    }

    /**
     * Gets outputs of previous layer.
     *
     * @return outputs of previous layer.
     */
    protected TreeMap<Integer, Matrix> getOutsP() {
        return backward.getPLayer().getOuts();
    }

    /**
     * Gets outputs of neural network layer.
     *
     * @return outputs of neural network layer.
     */
    public TreeMap<Integer, Matrix> getOuts(TreeMap<Integer, Matrix> outs) {
        return outs;
    }

    /**
     * Applies activation function to the outputs of neural network layer.
     *
     * @param inputs inputs for activation function.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void applyActivationFunction(TreeMap<Integer, Matrix> inputs) throws MatrixException {
        for (Integer index : inputs.keySet()) applyActivationFunction(inputs.get(index), true);
    }

    /**
     * Applies activation function to the output of neural network layer.
     *
     * @param input input for activation function.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void applyActivationFunction(Matrix input) throws MatrixException {
        applyActivationFunction(input, true);
    }

    /**
     * Applies activation function to the outputs of neural network layer.
     *
     * @param inputs inputs for activation function.
     * @param inplace applies activation function directly to input otherwise returns copy of input.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return returns inputs after applying of activation function.
     */
    protected TreeMap<Integer, Matrix> applyActivationFunction(TreeMap<Integer, Matrix> inputs, boolean inplace) throws MatrixException {
        TreeMap<Integer, Matrix> result = new TreeMap<>();
        for (Integer index : inputs.keySet()) result.put(index, applyActivationFunction(inputs.get(index), false));
        return result;
    }

    /**
     * Applies activation function to the output of neural network layer.
     *
     * @param input input for activation function.
     * @param inplace applies activation function directly to input otherwise returns copy of input.
     * @return if inplace returns input matrix otherwise new result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix applyActivationFunction(Matrix input, boolean inplace) throws MatrixException {
        Matrix result = inplace ? input : new DMatrix(input.getRows(), input.getCols());
        if (activation.getType() != ActivationFunctionType.SOFTMAX) {
            input.apply(result, activation.getFunction());
        }
        else {
            Matrix outputTemp = input.subtract(input.max()); // stable softmax X - max(X)
            outputTemp.apply(outputTemp, activation.getFunction());
            outputTemp.divide(outputTemp.sum(), result); // e^X / sum(e^X)
        }
        return result;
    }

    /**
     * Gets and calculates inner gradient of neural network layer.
     *
     * @param out output of neural network layer.
     * @param dEo output gradient of neural network layer.
     * @return inner gradient of neural network layer
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix getdEi(Matrix out, Matrix dEo) throws MatrixException {
        if (activation.getType() != ActivationFunctionType.SOFTMAX) {
            Matrix dAct = out.apply(activation.getDerivative());
            return dEo.multiply(dAct);
        }
        else {
            ones = (ones == null) ? new DMatrix(out.getRows(), 1, Init.ONE) : ones;
            I = (I == null) ? new DMatrix(out.getRows(), out.getRows(), Init.IDENTITY) : I;
            // dAct has diagonal entries of 1 - out and other entries -out i.e. I - out
            Matrix dAct = I.subtract(out.dot(ones.T()));
            // Finally dAct (network layer error) is dotted by output error resulting into input error
            return dAct.dot(dEo);
        }
    }

    /**
     * Checks if neural network layer output is flattened.
     *
     * @return true if neural network layer output is flattened otherwise false.
     */
    private boolean flattenedOutput() {
        return (isConvolutionalLayer() && forward.getNLayer() != null) && forward.getNLayer().isConvolutionalLayer();
    }

    /**
     * Flattens neural network output. Relevant for convolutional layers.<br>
     * Uses numbers of output samples and filter amount as parameters for flattening.<br>
     *
     * @param inputs output to be flattened.
     * @return flattened neural network output.
     */
    protected TreeMap<Integer, Matrix> flattenOutput(TreeMap<Integer, Matrix> inputs) {
        TreeMap<Integer, Matrix> outputs = new TreeMap<>();
        int sampleIndex = 0;
        int filterIndex = 0;
        int size = width * height * depth;
        Matrix output = null;
        for (Integer index : inputs.keySet()) {
            Matrix input = inputs.get(index);
            if (filterIndex == 0) outputs.put(sampleIndex, output = new DMatrix(size, 1));
            for (int row = 0; row < width; row++) {
                for (int col = 0; col < height; col++) {
                    output.setValue(getPos(row, col, filterIndex), 0 , input.getValue(row, col));
                }
            }
            if (++filterIndex == depth) {
                filterIndex = 0;
                sampleIndex++;
            }
        }
        return outputs;
    }

    /**
     * Unflattens neural network output. Relevant for convolutional layers.<br>
     * Uses numbers of output samples and channel amounts as parameters for unflattening.<br>
     *
     * @param inputs output to be unflattened.
     * @return unflattened neural network output.
     */
    protected TreeMap<Integer, Matrix> unflattenOutput(TreeMap<Integer, Matrix> inputs) {
        TreeMap<Integer, Matrix> outputs = new TreeMap<>();
        Matrix output;
        for (Integer sampleIndex : inputs.keySet()) {
            Matrix input = inputs.get(sampleIndex);
            for (int filterIndex = 0; filterIndex < depth; filterIndex++) {
                outputs.put(getOutIndex(filterIndex, sampleIndex, depth), output = new DMatrix(width, height));
                for (int row = 0; row < width; row++) {
                    for (int col = 0; col < height; col++) {
                        output.setValue(row, col, input.getValue(getPos(row, col, filterIndex), 0));
                    }
                }
            }
        }
        return outputs;
    }

    /**
     * Gets one dimensional index calculated based on width, height and depth.
     *
     * @param w weight as input
     * @param h heigth as input
     * @param d depth as input
     * @return one dimensional index
     */
    private int getPos(int w, int h, int d) {
        return w + width * h + width * height * d;
    }

    /**
     * Gets (calculates) flat filter index by filterIndex, depthIndex and number of channels for a convolutional layer.
     *
     * @param filterIndex index for filter.
     * @param channelIndex index for input channel.
     * @param channels number of input channels.
     * @return flat filter index.
     */
    protected int getFilterIndex(int filterIndex, int channelIndex, int channels) {
        return channelIndex + filterIndex * channels;
    }

    /**
     * Gets (calculates) output index by filterIndex, sampleIndex and number of filters for a convolutional layer.
     *
     * @param filterIndex index for filter.
     * @param sampleIndex index for current sample.
     * @param filters number of filters.
     * @return output index.
     */
    protected int getOutIndex(int filterIndex, int sampleIndex, int filters) {
        return filterIndex + sampleIndex * filters;
    }

    /**
     * Gets (calculates) input index by filterIndex, sampleIndex and number of channels for a convolutional layer.
     *
     * @param channelIndex index for channel.
     * @param sampleIndex index for current sample.
     * @param channels number of channels.
     * @return input index.
     */
    protected int getInIndex(int channelIndex, int sampleIndex, int channels) {
        return channelIndex + sampleIndex * channels;
    }

}
