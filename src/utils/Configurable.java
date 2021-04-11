package utils;

import java.util.HashMap;

/**
 * Interface that defines configurable entity.<br>
 *
 */
public interface Configurable {

    /**
     * Returns parameters used for configurable entity.
     *
     * @return parameters used for configurable entity.
     */
    HashMap<String, DynamicParam.ParamType> getParamDefs();

    /**
     * Sets parameters used for configurable entity.<br>
     *
     * @param params parameters
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    void setParams(DynamicParam params) throws DynamicParamException;

}
