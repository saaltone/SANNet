/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.regularization;

import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.sampling.Sequence;

/**
 * Implements gradient clipping class.<br>
 * Gradient clipping cuts gradient when certain threshold is reached to prevent then from growing too big i.e. exploding.<br>
 * <br>
 * Reference: https://hackernoon.com/gradient-clipping-57f04f0adae<br>
 *
 */
public class GradientClipping extends AbstractRegularization {

    /**
     * Parameter name types for GradientClipping.
     *     - threshold: threshold for clipping gradients. Default value 0.1.<br>
     *
     */
    private final static String paramNameTypes = "(threshold:DOUBLE)";

    /**
     * Threshold for gradient clipping.
     *
     */
    private double threshold;

    /**
     * Constructor for gradient clipping class.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public GradientClipping() throws DynamicParamException {
        super(RegularizationType.GRADIENT_CLIPPING, GradientClipping.paramNameTypes);
    }

    /**
     * Constructor for gradient clipping class.
     *
     * @param params parameters for gradient clipping.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public GradientClipping(String params) throws DynamicParamException {
        super(RegularizationType.GRADIENT_CLIPPING, GradientClipping.paramNameTypes, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        threshold = 0.1;
    }

    /**
     * Sets parameters used for gradient clipping.<br>
     * <br>
     * Supported parameters are:<br>
     *     - threshold: threshold for clipping gradients. Default value 0.1.<br>
     *
     * @param params parameters used for gradient clipping.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("threshold")) threshold = params.getValueAsDouble("threshold");
    }

    /**
     * Not used.
     *
     * @param sequence input sequence.
     */
    public void forward(Sequence sequence) {}

    /**
     * Not used.
     *
     * @param inputs inputs.
     */
    public void forward(MMatrix inputs) {}

    /**
     * Not used.
     *
     * @param weight weight matrix.
     * @return not used.
     */
    public double error(Matrix weight) {
        return 0;
    }

    /**
     * Executes gradient clipping prior weight update step for neural network.<br>
     * Compares gradient sum against specific threshold and cuts gradients proportionally is threshold is exceeded.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backward(Matrix weight, Matrix weightGradientSum) throws MatrixException {
        double weightGradientSumL2norm = Math.sqrt(weightGradientSum.norm(2));
        if (weightGradientSumL2norm > threshold) weightGradientSum.multiply(threshold / weightGradientSumL2norm, weightGradientSum);
    }

}
