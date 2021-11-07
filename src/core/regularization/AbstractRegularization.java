/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.regularization;

import utils.configurable.Configurable;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;

import java.io.Serial;
import java.io.Serializable;

/**
 * Class that defined AbstractRegularization and defined common functions for regularization.
 *
 */
public abstract class AbstractRegularization implements Configurable, Regularization, Serializable {

    @Serial
    private static final long serialVersionUID = 591019870341792870L;

    /**
     * Parameter name types for regularization.
     *
     */
    private final String paramNameTypes;

    /**
     * Type of regularization.
     *
     */
    private final RegularizationType regularizationType;

    /**
     * If true neural network is in state otherwise false.
     *
     */
    private transient boolean isTraining;

    /**
     * Constructor for AbstractRegularization.
     *
     * @param regularizationType regularization type
     * @param paramNameTypes parameter name types.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractRegularization(RegularizationType regularizationType, String paramNameTypes) throws DynamicParamException {
        this.regularizationType = regularizationType;
        this.paramNameTypes = paramNameTypes;
        initializeDefaultParams();
    }

    /**
     * Constructor for AbstractRegularization.
     *
     * @param regularizationType regularization type.
     * @param paramNameTypes parameter name types.
     * @param params parameters for drop out.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public AbstractRegularization(RegularizationType regularizationType, String paramNameTypes, String params) throws DynamicParamException {
        this(regularizationType, paramNameTypes);
        if (paramNameTypes != null && params != null) setParams(new DynamicParam(params, getParamDefs()));
    }

    /**
     * Returns parameters used for drop out.
     *
     * @return parameters used for drop out.
     */
    public String getParamDefs() {
        return paramNameTypes;
    }

    /**
     * Sets flag for drop out if neural network is in training state.
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
     * Returns name of regularization.
     *
     * @return name of regularization.
     */
    public String getName() {
        return regularizationType.toString();
    }

}
