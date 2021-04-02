package utils;

import java.util.HashMap;

/**
 * Interface that defines Configurable entity.
 *
 */
public interface Configurable {

    /**
     * Returns parameters used for Configurable entity.
     *
     * @return parameters used for Configurable entity.
     */
    HashMap<String, DynamicParam.ParamType> getParamDefs();

    /**
     * Sets parameters used for Configurable entity.<br>
     *
     * @param params parameters
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void setParams(DynamicParam params) throws DynamicParamException;

}
