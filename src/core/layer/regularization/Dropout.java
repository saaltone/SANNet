/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.layer.regularization;

import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Initialization;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.sampling.Sequence;

/**
 * Implements drop out regularization method for layer weights (parameters).<br>
 * Drop out is based on stochastic selection of layer nodes that are removed from training process at each training step.<br>
 * This forces other nodes to take over learning process reducing neural network's tendency to overfit.<br>
 * <br>
 * Reference: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf<br>
 *
 */
public class Dropout extends AbstractRegularizationLayer {

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
     * Takes single forward processing step to process layer input(s).<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forwardProcess() throws MatrixException {
        if (isTraining() || monte_carlo) {
            this.reset();
            Sequence inputSequence = getDefaultLayerInput();
            for (MMatrix sample : inputSequence.values()) {
                int matrixDepth = sample.getDepth();
                for (int inputDepth = 0; inputDepth < matrixDepth; inputDepth++) {
                    // Implements forward step for inverted drop out.<br>
                    // Function selectively masks out certain percentage of node governed by parameter probability during training phase.<br>
                    // During training phase it also compensates all remaining inputs by dividing by probability.<br>
                    Matrix matrix = sample.get(inputDepth);
                    matrix.multiply(1 / probability, matrix);
                    matrix.setMask();
                    matrix.getMask().setProbability(probability);
                    matrix.getMask().maskRowByProbability();
                }
            }
            setLayerOutputs(inputSequence);
        }
        else passLayerOutputs();
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
