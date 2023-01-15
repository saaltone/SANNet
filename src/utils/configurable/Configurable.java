/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.configurable;

/**
 * Interface that defines configurable entity.<br>
 *
 */
public interface Configurable {

    /**
     * Initializes default params.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void initializeDefaultParams() throws DynamicParamException;

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
