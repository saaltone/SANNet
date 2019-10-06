/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.layer;

import core.NeuralNetworkException;
import core.activation.ActivationFunction;
import utils.*;

import java.io.Serializable;
import java.util.HashMap;
import java.util.TreeMap;

/**
 * Abstract class that implements execution layer for actual neural network layers (feed forward layer, recurrent layer etc.)<br>
 * Provides supportive functions for actual neural network layers.<br>
 * Support automatic gradient i.e. backward gradient calculation for layers supporting it.<br>
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
    protected ActivationFunction activation;

    /**
     * Initialization function for neural network layer.
     *
     */
    protected Init initialization = Init.UNIFORM_XAVIER;

    /**
     * Procedure for the layer. Procedure contains chain of forward and backward expressions.
     *
     */
    private Procedure procedure = null;

    /**
     * Flag if state is reset prior start of next training sequence.
     *
     */
    protected boolean resetStateTraining = true;

    /**
     * Flag if state is reset prior start of next test (validate, predict) sequence.
     *
     */
    protected boolean resetStateTesting = true;

    /**
     * Previous state;
     *
     */
    private transient boolean previousStateTraining;

    /**
     * If true allows layer recurrent input to be reset between test (validate, predict) sequence.
     *
     */
    private boolean allowLayerReset = true;

    /**
     * Limits number of backward propagation sequence steps.
     *
     */
    protected int truncateSteps = -1;

    /**
     * Constructor for abstract execution layer.
     *
     * @param parent reference to parent abstract layer.
     * @param activation activation function.
     * @param initialization initialization function.
     * @param params parameters for neural network layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception setting of activation function fails.
     */
    protected AbstractExecutionLayer(AbstractLayer parent, ActivationFunction activation, Init initialization, String params) throws DynamicParamException, NeuralNetworkException {
        this.parent = parent;

        if (activation != null) this.activation = activation;
        else this.activation = new ActivationFunction(UniFunctionType.ELU);

        if (initialization != null) this.initialization = initialization;

        if (params != null) setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Gets parameters used for layer.<br>
     * Implemented by actual neural network layer with parameters specific to that layer.<br>
     *
     * @return parameters used for recurrent layer.
     */
    public abstract HashMap<String, DynamicParam.ParamType> getParamDefs();

    /**
     * Sets parameters used for layer.<br>
     * Implemented by actual neural network layer with parameters specific to that layer.<br>
     *
     * @param params parameters used for recurrent layer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public abstract void setParams(DynamicParam params) throws DynamicParamException;

    /**
     * Return layer type by name.
     *
     * @return layer type by name.
     * @throws NeuralNetworkException throws exception if operation fails.
     */
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
     * Sets if recurrent inputs of layer are allowed to be reset.
     *
     * @param allowLayerReset if true allows reset.
     */
    public void setAllowLayerReset(boolean allowLayerReset) {
        this.allowLayerReset = allowLayerReset;
    }

    /**
     * Takes single forward processing step process layer input(s).<br>
     * Applies automated forward procedure when relevant to layer.<br>
     * Additionally applies any regularization defined for the layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    public void forwardProcess() throws MatrixException, NeuralNetworkException {
        if (procedure == null) defineProcedure();

        backward.regulateForwardPre(getOutsP(), -1);
        backward.normalizeForwardPre(getOutsP(), 1);

        parent.resetOuts();

        boolean hasDependencies = procedure.hasDependencies();

        if (allowLayerReset && hasDependencies) {
            if (previousStateTraining != backward.isTraining()) procedure.resetDependencies();
            else if(resetStateTraining || resetStateTesting) procedure.resetDependencies();
        }
        previousStateTraining = backward.isTraining();

        for (Integer index : getOutsP().keySet()) {
            backward.regulateForwardPre(getOutsP(), index);
            parent.getOuts().put(index, procedure.calculateExpression(index, getOutsP().get(index)).getMatrix(index));
            if(!backward.isTraining() && !hasDependencies) procedure.reset(index);
        }
        if(!backward.isTraining()) procedure.reset();

        backward.regulateForwardPost(parent.getOuts());
        backward.normalizeForwardPost(parent.getOuts());

        if (forward == null) parent.updateOutputError();
    }

    /**
     * Defines layer procedure for forward and backward calculation (automatic gradient) by applying procedure factory.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private void defineProcedure() throws MatrixException {
        ProcedureFactory procedureFactory = new ProcedureFactory();

        procedureFactory.registerMatrix(backward.getWs(), true);

        boolean reset = true;
        while (procedure == null) {
            Matrix input = new DMatrix(backward.getPLayer().getWidth(), 1, Init.ONE);
            procedureFactory.newProcedure(input);
            Matrix output = getForwardProcedure(input, reset);
            procedure = procedureFactory.endProcedure(output);
            reset = false;
        }
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @param input input of forward procedure.
     * @param reset reset recurring inputs of procedure.
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected abstract Matrix getForwardProcedure(Matrix input, boolean reset) throws MatrixException;

    /**
     * Takes single backward processing step to process layer output gradient(s).<br>
     * Applies automated backward (automatic gradient) procedure when relevant to layer.<br>
     * Additionally applies any regularization defined for the layer.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backwardProcess() throws MatrixException {
        parent.resetOutGrads();

        backward.resetGrad();

        backward.regulateBackward(-1);

        int errorStep = 0;
        for (Integer index : parent.getOuts().descendingKeySet()) {
            backward.regulateBackward(index);
            parent.getdEos().put(index, procedure.calculateGradient(index, parent.getdEosN().get(index)).getGradient(index));
            for (Matrix W : backward.getdWs().keySet()) backward.getdWs(W).put(index, procedure.getNode(W).getGradient(index));
            if (truncateSteps > 0 && ++errorStep >= truncateSteps) break;
        }

        procedure.reset();

        backward.normalizeBackward();

        backward.sumGrad();
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
