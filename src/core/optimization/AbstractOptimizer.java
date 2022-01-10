/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.optimization;

import utils.configurable.Configurable;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;

import java.io.Serial;
import java.io.Serializable;

/**
 * Class that defines AbstractOptimizer containing common functions for optimizers.
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
     * Parameters of optimizer.
     *
     */
    private final String params;

    /**
     * Default constructor for AbstractOptimizer.
     *
     * @param optimizationType optimization type.
     * @param paramNameTypes parameter name types for optimizer.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractOptimizer(OptimizationType optimizationType, String paramNameTypes) throws DynamicParamException {
        this.optimizationType = optimizationType;
        this.paramNameTypes = paramNameTypes;
        this.params = null;
        initializeDefaultParams();
    }

    /**
     * Constructor for AbstractOptimizer.
     *
     * @param optimizationType optimization type.
     * @param paramNameTypes parameter name types.
     * @param params parameters for Adadelta.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractOptimizer(OptimizationType optimizationType, String paramNameTypes, String params) throws DynamicParamException {
        this.optimizationType = optimizationType;
        this.paramNameTypes = paramNameTypes;
        this.params = params;
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

}
