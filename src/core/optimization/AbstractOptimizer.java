/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.optimization;

import utils.configurable.Configurable;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashMap;

/**
 * Implements abstract optimizer containing common functions for optimizers.
 *
 */
public abstract class AbstractOptimizer implements Configurable, Optimizer, Serializable {

    @Serial
    private static final long serialVersionUID = 6617359131344171545L;

    /**
     * Optimization type.
     *
     */
    private final OptimizationType optimizationType;

    /**
     * Parameter name types for optimizer.
     *
     */
    private final String paramNameTypes;

    /**
     * Default constructor for AbstractOptimizer.
     *
     * @param optimizationType optimization type.
     * @param paramNameTypes parameter name types for optimizer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public AbstractOptimizer(OptimizationType optimizationType, String paramNameTypes) throws DynamicParamException, MatrixException {
        this.optimizationType = optimizationType;
        this.paramNameTypes = paramNameTypes;
        initializeDefaultParams();
    }

    /**
     * Constructor for AbstractOptimizer.
     *
     * @param optimizationType optimization type.
     * @param paramNameTypes parameter name types.
     * @param params parameters for Adadelta.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public AbstractOptimizer(OptimizationType optimizationType, String paramNameTypes, String params) throws DynamicParamException, MatrixException {
        this.optimizationType = optimizationType;
        this.paramNameTypes = paramNameTypes;
        initializeDefaultParams();
        if (paramNameTypes != null && params != null) setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for optimizer.
     *
     * @return parameters used for optimizer.
     */
    public String getParamDefs() {
        return paramNameTypes;
    }

    /**
     * Returns name of optimizer.
     *
     * @return name of optimizer.
     */
    public String getName() {
        return optimizationType.toString();
    }

    /**
     * Returns existing of new parameter matrix based on given matrix.
     *
     * @param parameterMatrices parameter matrices.
     * @param matrix matrix.
     * @return parameter matrix.
     */
    protected Matrix getParameterMatrix(HashMap<Matrix, Matrix> parameterMatrices, Matrix matrix) {
        Matrix parameterMatrix = parameterMatrices.get(matrix);
        if (parameterMatrix == null)  parameterMatrices.put(matrix, parameterMatrix = new DMatrix(matrix.getRows(), matrix.getColumns(), matrix.getDepth()));
        return parameterMatrix;
    }

    /**
     * Set value to specific matrix.
     *
     * @param parameterMatrices parameter matrices.
     * @param matrix matrix
     * @param value value.
     */
    protected void setParameterMatrix(HashMap<Matrix, Matrix> parameterMatrices, Matrix matrix, Matrix value) {
        parameterMatrices.put(matrix, value);
    }

}
