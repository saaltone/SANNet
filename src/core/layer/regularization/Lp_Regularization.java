/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.layer.regularization;

import core.network.NeuralNetworkException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Initialization;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Implements Lp regularization (experimental). P here is any norm higher or equal to 1.<br>
 * Setting p = 1 this becomes L1 regularization and setting p = 2 this becomes L2 regularization.<br>
 * <br>
 * This is experimental regularization method.<br>
 *
 */
public class Lp_Regularization extends AbstractLx_Regularization {

    /**
     * Parameter name types for Lp regularization.
     *     - p: p norm of normalizer. Default 3.<br>
     *
     */
    private final static String paramNameTypes = "(p:INT)";

    /**
     * Order of norm.
     *
     */
    private int p;

    /**
     * Constructor for Lp regularization layer.
     *
     * @param layerIndex layer index
     * @param initialization initialization function for weight.
     * @param params parameters for feedforward layer.
     * @throws NeuralNetworkException throws exception if setting of activation function fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Lp_Regularization(int layerIndex, Initialization initialization, String params) throws NeuralNetworkException, DynamicParamException {
        super (layerIndex, initialization, params);
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        super.initializeDefaultParams();
        p = 3;
    }

    /**
     * Returns parameters used for Lp regularization layer.
     *
     * @return parameters used for Lp regularization layer.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + Lp_Regularization.paramNameTypes;
    }

    /**
     * Sets parameters used for Lp regularization.<br>
     * <br>
     * Supported parameters are:<br>
     *     - p: p norm of normalizer. Default 3.<br>
     *
     * @param params parameters used for Lp regularization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if minimum layer dimensions are not met.
     */
    public void setParams(DynamicParam params) throws DynamicParamException, NeuralNetworkException {
        super.setParams(params);
        if (params.hasParam("p")) p = params.getValueAsInteger("p");
    }

    /**
     * Applies regularization.
     *
     * @param weight weight
     * @param lambda lambda value
     * @return regularization result
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected Matrix applyRegularization(Matrix weight, double lambda) throws MatrixException {
        return weight.apply((value) -> value != 0 ? p * lambda * Math.pow(Math.abs(value), p - 1) / value : 0);
    }

    /**
     * Returns layer details as string.
     *
     * @return layer details as string.
     */
    protected String getLayerDetailsByName() {
        return super.getLayerDetailsByName() + ", p: " + p;
    }

}
