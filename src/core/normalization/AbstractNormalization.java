/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.normalization;

import core.optimization.Optimizer;
import utils.configurable.Configurable;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.procedure.ForwardProcedure;
import utils.procedure.Procedure;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashMap;

/**
 * Defines AbstractNormalization class that defined common functions for normalization.
 *
 */
public abstract class AbstractNormalization implements Configurable, Normalization, ForwardProcedure, Serializable {

    @Serial
    private static final long serialVersionUID = -1568307305919278917L;

    /**
     * Parameter name types for normalization.
     *
     */
    private final String paramNameTypes;

    /**
     * Type of normalization.
     *
     */
    private final NormalizationType normalizationType;

    /**
     * If true neural network is in state otherwise false.
     *
     */
    private transient boolean isTraining;

    /**
     * Optimizer for batch normalization;
     *
     */
    private Optimizer optimizer;

    /**
     * Procedure for normalization.
     *
     */
    private Procedure procedure;

    /**
     * Default constructor for batch normalization class.
     *
     * @param normalizationType normalization type.
     * @param paramNameTypes parameter name types.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractNormalization(NormalizationType normalizationType, String paramNameTypes) throws DynamicParamException {
        this.normalizationType = normalizationType;
        this.paramNameTypes = paramNameTypes;
        initializeDefaultParams();
    }

    /**
     * Constructor for batch normalization class.
     *
     * @param normalizationType normalization type.
     * @param paramNameTypes parameter name types.
     * @param params parameters for batch normalization.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractNormalization(NormalizationType normalizationType, String paramNameTypes, String params) throws DynamicParamException {
        this(normalizationType, paramNameTypes);
        if (paramNameTypes != null && params != null) setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for batch normalization.
     *
     * @return parameters used for batch normalization.
     */
    public String getParamDefs() {
        return paramNameTypes;
    }

    /**
     * Sets flag for batch normalization if neural network is in training state.
     *
     * @param isTraining if true neural network is in state otherwise false.
     */
    public void setTraining(boolean isTraining) {
        this.isTraining = isTraining;
    }

    /**
     * Returns true if neural network is in training mode otherwise returns false.
     *
     * @return true if neural network is in training mode otherwise returns false.
     */
    protected boolean isTraining() {
        return isTraining;
    }

    /**
     * Sets optimizer for normalizer.
     *
     * @param optimizer optimizer
     */
    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
    }

    /**
     * Returns optimizer
     *
     * @return optimizer
     */
    protected Optimizer getOptimizer() {
        return optimizer;
    }

    /**
     * Sets procedure.
     *
     * @param procedure procedure.
     */
    protected void setProcedure(Procedure procedure) {
        this.procedure = procedure;
    }

    /**
     * Returns procedure.
     *
     * @return procedure.
     */
    protected Procedure getProcedure() {
        return procedure;
    }

    /**
     * Returns name of normalization.
     *
     * @return name of normalization.
     */
    public String getName() {
        return normalizationType.toString();
    }

    /**
     * Updates weight gradient
     *
     * @param weight weight
     * @param weightGradients weight gradients
     * @throws MatrixException throws exception if matrix operation fails.
     */
    protected void updateWeightGradient(Matrix weight, HashMap<Matrix, Matrix> weightGradients) throws MatrixException {
        Matrix weightGradient = getProcedure().getGradient(weight);
        if (!weightGradients.containsKey(weight)) weightGradients.put(weight, weightGradient);
        else weightGradients.get(weight).add(weightGradient, weightGradients.get(weight));
    }

    /**
     * Executes optimizer step for normalizer.
     *
     * @param weightGradients weight gradients.
     * @return empty weight gradients.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public HashMap<Matrix, Matrix> optimize(HashMap<Matrix, Matrix> weightGradients) throws MatrixException, DynamicParamException {
        for (Matrix weight : weightGradients.keySet()) getOptimizer().optimize(weight, weightGradients.get(weight));
        return new HashMap<>();
    }

    /**
     * Prints expression chains of normalization.
     *
     */
    public void printExpressions() {
        if (getProcedure() == null) return;
        System.out.println("Normalization: " + getName() + ":");
        getProcedure().printExpressionChain();
        System.out.println();
    }

    /**
     * Prints gradient chains of normalization.
     *
     */
    public void printGradients() {
        if (getProcedure() == null) return;
        System.out.println("Normalization: " + getName() + ":");
        getProcedure().printGradientChain();
        System.out.println();
    }

}
