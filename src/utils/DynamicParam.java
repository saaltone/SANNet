/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Class for dynamic parameter handling.<br>
 * Let's user to define parameters and their types.<br>
 * Stores parameter as name, type and value triplet.<br>
 *
 */
public class DynamicParam implements Serializable {

    private static final long serialVersionUID = -5783783860424243849L;

    /**
     * Enum for defining parameter type<br>
     * Supports integer (INT), long (LONG), float (FLOAT), double (DOUBLE), char (CHAR), string (String) and boolean (BOOLEAN) types.<br>
     *
     */
    public enum ParamType {
        INT,
        LONG,
        FLOAT,
        DOUBLE,
        CHAR,
        STRING,
        BOOLEAN
    }

    /**
     * Class that handles and stores parameter type value pair.
     *
     */
    private static class TypeValue {

        /**
         * Defines parameter type.
         *
         */
        private ParamType type;

        /**
         * Defines value as object.
         *
         */
        private Object value;

        /**
         * Construction to construct parameter from name, type and value (as string).
         *
         * @param param parameter name.
         * @param type parameter type.
         * @param newValue parameter value.
         * @throws DynamicParamException throws exception if parameter creation fails.
         */
        TypeValue(String param, ParamType type, String newValue) throws DynamicParamException {
            if (type == ParamType.INT) {
                try {
                    int val = Integer.parseInt(newValue);
                    this.type = type;
                    this.value = val;
                }
                catch (NumberFormatException exception) {
                    throw new IllegalArgumentException("Parameter: " + param + ": Cannot convert value to Integer");
                }
            }
            if (type == ParamType.LONG) {
                try {
                    long val = Long.parseLong(newValue);
                    this.type = type;
                    this.value = val;
                }
                catch (NumberFormatException exception) {
                    throw new DynamicParamException("Parameter: " + param + ": Cannot convert value to Long");
                }
            }
            if (type == ParamType.FLOAT) {
                try {
                    float val = Float.parseFloat(newValue);
                    this.type = type;
                    this.value = val;
                }
                catch (NumberFormatException exception) {
                    throw new DynamicParamException("Parameter: " + param + ": Cannot convert value to Float");
                }
            }
            if (type == ParamType.DOUBLE) {
                try {
                    double val = Double.parseDouble(newValue);
                    this.type = type;
                    this.value = val;
                }
                catch (NumberFormatException exception) {
                    throw new DynamicParamException("Parameter: " + param + ": Cannot convert value to Double");
                }
            }
            if (type == ParamType.CHAR) {
                if (newValue.length() != 1) throw new DynamicParamException("Parameter: " + param + ": Cannot convert value to Character");
                char val = newValue.charAt(0);
                this.type = type;
                this.value = val;
            }
            if (type == ParamType.STRING) {
                this.type = type;
                this.value = newValue;
            }
            if (type == ParamType.BOOLEAN) {
                if (newValue.equalsIgnoreCase("TRUE")) this.value = true;
                else if (newValue.equalsIgnoreCase("FALSE")) this.value = false;
                else throw new DynamicParamException("Parameter: " + param + ": Cannot convert value to Boolean");
                this.type = type;
            }
        }

        /**
         * Construction to construct parameter from name, type and value (as object).
         *
         * @param param parameter name.
         * @param type parameter type.
         * @param value parameter value.
         * @throws DynamicParamException throws exception if parameter creation fails.
         */
        TypeValue(String param, ParamType type, Object value) throws DynamicParamException {
            if (type == ParamType.INT && !(value instanceof Integer)) throw new DynamicParamException("Parameter: " + param + " value not type of Integer.");
            if (type == ParamType.LONG && !(value instanceof Long)) throw new DynamicParamException("Parameter: " + param + " value not type of Long.");
            if (type == ParamType.FLOAT && !(value instanceof Float)) throw new DynamicParamException("Parameter: " + param + " value not type of Float.");
            if (type == ParamType.DOUBLE && !(value instanceof Double)) throw new DynamicParamException("Parameter: " + param + " value not type of Double.");
            if (type == ParamType.CHAR && !(value instanceof Character)) throw new DynamicParamException("Parameter: " + param + " value not type of Character.");
            if (type == ParamType.STRING && !(value instanceof String)) throw new DynamicParamException("Parameter: " + param + " value not type of String.");
            if (type == ParamType.BOOLEAN && !(value instanceof Boolean)) throw new DynamicParamException("Parameter: " + param + " value not type of Boolean.");
            this.type = type;
            this.value = value;
        }

        /**
         * Returns parameter type
         *
         * @return type of parameter.
         */
        public ParamType getType() {
            return type;
        }

        /**
         * Returns value of parameter.
         *
         * @return value of parameter as object.
         */
        public Object getValue() {
            return value;
        }

        /**
         * Returns value of parameter as integer.
         *
         * @param param name of parameter.
         * @return value of parameter.
         * @throws DynamicParamException throws exception if parameter type is not integer.
         */
        public int getValueAsInteger(String param) throws DynamicParamException {
            if (type != ParamType.INT) throw new DynamicParamException("Parameter: " + param + " value not type of Integer");
            return (int)value;
        }

        /**
         * Returns value of parameter as long.
         *
         * @param param name of parameter.
         * @return value of parameter.
         * @throws DynamicParamException throws exception if parameter type is not long.
         */
        public long getValueAsLong(String param) throws DynamicParamException {
            if (type != ParamType.LONG) throw new DynamicParamException("Parameter: " + param + " value not type of Long");
            return (long)value;
        }

        /**
         * Returns value of parameter as float.
         *
         * @param param name of parameter.
         * @return value of parameter.
         * @throws DynamicParamException throws exception if parameter type is not float.
         */
        public float getValueAsFloat(String param) throws DynamicParamException {
            if (type != ParamType.FLOAT) throw new DynamicParamException("Parameter: " + param + " value not type of Float");
            return (float)value;
        }

        /**
         * Returns value of parameter as double.
         *
         * @param param name of parameter.
         * @return value of parameter.
         * @throws DynamicParamException throws exception if parameter type is not double.
         */
        public double getValueAsDouble(String param) throws DynamicParamException {
            if (type != ParamType.DOUBLE) throw new DynamicParamException("Parameter: " + param + " value not type of Double");
            return (double)value;
        }

        /**
         * Returns value of parameter as character.
         *
         * @param param name of parameter.
         * @return value of parameter.
         * @throws DynamicParamException throws exception if parameter type is not character.
         */
        public char getValueAsChar(String param) throws DynamicParamException {
            if (type != ParamType.CHAR) throw new DynamicParamException("Parameter: " + param + " value not type of Char");
            return (char)value;
        }

        /**
         * Returns value of parameter as string.
         *
         * @param param name of parameter.
         * @return value of parameter.
         * @throws DynamicParamException throws exception if parameter type is not string.
         */
        public String getValueAsString(String param) throws DynamicParamException {
            if (type != ParamType.STRING) throw new DynamicParamException("Parameter: " + param + " value not type of String");
            return (String)value;
        }

        /**
         * Returns value of parameter as boolean.
         *
         * @param param name of parameter.
         * @return value of parameter.
         * @throws DynamicParamException throws exception if parameter type is not boolean.
         */
        public boolean getValueAsBoolean(String param) throws DynamicParamException {
            if (type != ParamType.BOOLEAN) throw new DynamicParamException("Parameter: " + param + " value not type of Boolean");
            return (boolean)value;
        }

    }

    /**
     * Hashmap used to store parameter list.
     *
     */
    private final HashMap<String, TypeValue> paramList = new HashMap<>();

    /**
     * Constructor to build dynamic parameter list.
     *
     * @param params parameter name value pairs separated by comma.
     * @param nameTypes name type pairs of parameters as hashmap.
     * @throws DynamicParamException throws exception if parameters are not provided or parameter is properly defined.
     */
    public DynamicParam(String params, HashMap<String, ParamType> nameTypes) throws DynamicParamException {
        if (params.isEmpty()) return;
        String[] paramsList = params.split(",");
        if (paramsList.length == 0) throw new DynamicParamException("No parameters found.");
        for (String s : paramsList) setParamByVal(s, nameTypes);
    }

    /**
     * Defines parameter based on given name value and name type pairs.
     *
     * @param param parameter to be set as name value pair separated by equal sign.
     * @param nameTypes name type list of parameters.
     * @throws DynamicParamException throws exception if parameter is not properly defined or parameter is cannot be casted into defined parameter type.
     */
    public void setParamByVal(String param, HashMap<String, ParamType> nameTypes) throws DynamicParamException {
        String[] typeValue = param.split("=");
        if (typeValue.length != 2) throw new DynamicParamException("Invalid parameter definition: " + param);
        typeValue[0] = typeValue[0].trim();
        if (!nameTypes.containsKey(typeValue[0])) throw new DynamicParamException("Invalid parameter: " + typeValue[0]);
        typeValue[1] = typeValue[1].trim();
        paramList.put(typeValue[0], new TypeValue(typeValue[0], nameTypes.get(typeValue[0]), typeValue[1]));
    }

    /**
     * Returns value of parameter.
     *
     * @param param name of parameter.
     * @return value of parameter as object.
     * @throws DynamicParamException throws exception if parameter by name is not found.
     */
    public Object getValue(String param) throws DynamicParamException {
        if (!paramList.containsKey(param.trim())) throw new DynamicParamException("No parameter: " + param + " found");
        return paramList.get(param).getValue();
    }

    /**
     * Returns value of parameter as integer.
     *
     * @param param name of parameter.
     * @return value of parameter as integer.
     * @throws DynamicParamException throws exception if parameter by name is not found.
     */
    public int getValueAsInteger(String param) throws DynamicParamException {
        if (!paramList.containsKey(param.trim())) throw new DynamicParamException("No parameter: " + param + " found");
        return paramList.get(param).getValueAsInteger(param);
    }

    /**
     * Returns value of parameter as long.
     *
     * @param param name of parameter.
     * @return value of parameter as long.
     * @throws DynamicParamException throws exception if parameter by name is not found.
     */
    public long getValueAsLong(String param) throws DynamicParamException {
        if (!paramList.containsKey(param.trim())) throw new DynamicParamException("No parameter: " + param + " found");
        return paramList.get(param).getValueAsLong(param);
    }

    /**
     * Returns value of parameter as float.
     *
     * @param param name of parameter.
     * @return value of parameter as float.
     * @throws DynamicParamException throws exception if parameter by name is not found.
     */
    public float getValueAsFloat(String param) throws DynamicParamException {
        if (!paramList.containsKey(param.trim())) throw new DynamicParamException("No parameter: " + param + " found");
        return paramList.get(param).getValueAsFloat(param);
    }

    /**
     * Returns value of parameter as double.
     *
     * @param param name of parameter.
     * @return value of parameter as double.
     * @throws DynamicParamException throws exception if parameter by name is not found.
     */
    public double getValueAsDouble(String param) throws DynamicParamException {
        if (!paramList.containsKey(param.trim())) throw new DynamicParamException("No parameter: " + param + " found");
        return paramList.get(param).getValueAsDouble(param);
    }

    /**
     * Returns value of parameter as character.
     *
     * @param param name of parameter.
     * @return value of parameter as character.
     * @throws DynamicParamException throws exception if parameter by name is not found.
     */
    public char getValueAsChar(String param) throws DynamicParamException {
        if (!paramList.containsKey(param.trim())) throw new DynamicParamException("No parameter: " + param + " found");
        return paramList.get(param).getValueAsChar(param);
    }

    /**
     * Returns value of parameter as string.
     *
     * @param param name of parameter.
     * @return value of parameter as string.
     * @throws DynamicParamException throws exception if parameter by name is not found.
     */
    public String getValueAsString(String param) throws DynamicParamException {
        if (!paramList.containsKey(param.trim())) throw new DynamicParamException("No parameter: " + param + " found");
        return paramList.get(param).getValueAsString(param);
    }

    /**
     * Returns value of parameter as boolean.
     *
     * @param param name of parameter.
     * @return value of parameter as boolean.
     * @throws DynamicParamException throws exception if parameter by name is not found.
     */
    public boolean getValueAsBoolean(String param) throws DynamicParamException {
        if (!paramList.containsKey(param.trim())) throw new DynamicParamException("No parameter: " + param + " found");
        return paramList.get(param).getValueAsBoolean(param);
    }

    /**
     * Checks if specific parameter exists by name.
     *
     * @param param name of parameter.
     * @return true if parameter is found otherwise false.
     */
    public boolean hasParam(String param) {
        return paramList.containsKey(param.trim());
    }

}

