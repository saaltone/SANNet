/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package utils;

/**
 * Interface that defines configurable entity.<br>
 *
 */
public interface Configurable {

    /**
     * Initializes default params.
     *
     */
    void initializeDefaultParams();

    /**
     * Returns parameters used for configurable entity.
     *
     * @return parameters used for configurable entity.
     */
    String getParamDefs();

    /**
     * Sets parameters used for configurable entity.<br>
     *
     * @param params parameters
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void setParams(DynamicParam params) throws DynamicParamException;

}
