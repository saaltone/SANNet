/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer.regularization;

import core.layer.AbstractExecutionLayer;
import core.layer.WeightSet;
import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.Initialization;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashSet;
import java.util.TreeMap;

/**
 * Implements drop out regularization method for layer weights (parameters).<br>
 * Drop out is based on stochastic selection of layer nodes that are removed from training process at each training step.<br>
 * This forces other nodes to take over learning process reducing neural network's tendency to overfit.<br>
 * <br>
 * Reference: <a href="https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf">...</a><br>
 *
 */
public class Dropout extends AbstractExecutionLayer {

    /**
     * Parameter name types for drop out.
     *     - probability: probability of masking out a layer node. Default value 0.5.<br>
     *     - monte_carlo: if true applies Monte Carlo dropout (applies dropout also during inference). Default value false.<br>
     *
     */
    private final static String paramNameTypes = "(probability:DOUBLE), " +
            "(monte_carlo:BOOLEAN)";

    /**
     * Drop out probability of node.
     *
     */
    private double probability;

    /**
     * If true applies Monte Carlo drop out (drop out during inference) otherwise applies normal drop out.
     *
     */
    private boolean monte_carlo;

    /**
     * Input matrices for procedure construction.
     *
     */
    private TreeMap<Integer, Matrix> inputs;

    /**
     * Constructor for drop out layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for feedforward layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Dropout(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException {
        super (layerIndex, initialization, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        probability = 0.5;
        monte_carlo = false;
    }

    /**
     * Returns parameters used for drop out layer.
     *
     * @return parameters used for drop out layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + Dropout.paramNameTypes;
    }

    /**
     * Sets parameters used for drop out.<br>
     * <br>
     * Supported parameters are:<br>
     *     - probability: probability of masking out a layer node. Default value 0.5.<br>
     *     - monte_carlo: if true applies Monte Carlo dropout (applies dropout also during inference). Default value false.<br>
     *
     * @param params parameters used for drop out.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("probability")) probability = 1 - params.getValueAsDouble("probability");
        if (params.hasParam("monte_carlo")) monte_carlo = params.getValueAsBoolean("monte_carlo");
    }

    /**
     * Checks if layer is recurrent layer type.
     *
     * @return always false.
     */
    public boolean isRecurrentLayer() { return false; }

    /**
     * Checks if layer works with recurrent layers.
     *
     * @return if true layer works with recurrent layers otherwise false.
     */
    public boolean worksWithRecurrentLayer() {
        return true;
    }

    /**
     * Returns weight set.
     *
     * @return weight set.
     */
    protected WeightSet getWeightSet() {
        return null;
    }

    /**
     * Initializes neural network layer weights.
     *
     */
    public void initializeWeights() {
    }

    /**
     * Returns input matrices for procedure construction.
     *
     * @param resetPreviousInput if true resets also previous input.
     * @return input matrix for procedure construction.
     */
    public TreeMap<Integer, Matrix> getInputMatrices(boolean resetPreviousInput) {
        inputs = new TreeMap<>();
        Matrix input = new DMatrix(getLayerWidth(), getLayerHeight(), getLayerDepth(), Initialization.ONE);
        input.setName("Input" + getDefaultPreviousLayer().getLayerIndex());
        inputs.put(0, input);
        return inputs;
    }

    /**
     * Builds forward procedure and implicitly builds backward procedure.
     *
     * @return output of forward procedure.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix getForwardProcedure() throws MatrixException {
        Matrix output = inputs.get(0);
        output = output.dropout(probability, monte_carlo, false);
        output.setName("Output");

        return output;
    }

    /**
     * Returns matrices for which gradient is not calculated.
     *
     * @return matrices for which gradient is not calculated.
     */
    public HashSet<Matrix> getStopGradients() {
        return new HashSet<>();
    }

    /**
     * Returns constant matrices.
     *
     * @return constant matrices.
     */
    public HashSet<Matrix> getConstantMatrices() {
        return new HashSet<>();
    }

    /**
     * Returns number of truncated steps for gradient calculation. -1 means no truncation.
     *
     * @return number of truncated steps.
     */
    protected int getTruncateSteps() {
        return -1;
    }

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        return "Probability: " + (1 - probability) + ", Monte Carlo: " + monte_carlo;
    }

}
