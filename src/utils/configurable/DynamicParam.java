/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.configurable;

import java.io.Serial;
import java.io.Serializable;
import java.util.HashMap;

/**
 * Implements dynamic parameter handling.<br>
 * Allows user to define parameters and their types.<br>
 * Stores parameter as name, type and value triplet.<br>
 *
 */
public class DynamicParam implements Serializable {

    @Serial
    private static final long serialVersionUID = -5783783860424243849L;

    /**
     * Enum for defining parameter type.
     *
     */
    public enum ParamType {

        /**
         * Integer
         *
         */
        INT,

        /**
         * Long
         *
         */
        LONG,

        /**
         * Float
         *
         */
        FLOAT,

        /**
         * Double
         *
         */
        DOUBLE,

        /**
         * Character
         *
         */
        CHAR,

        /**
         * String
         *
         */
        STRING,

        /**
         * Boolean
         *
         */
        BOOLEAN,

        /**
         * List
         *
         */
        LIST

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
            switch (type) {
                case INT:
                    try {
                        int val = Integer.parseInt(newValue);
                        this.type = type;
                        this.value = val;
                    }
                    catch (NumberFormatException exception) {
                        throw new IllegalArgumentException("Parameter: " + param + ": Cannot convert value to Integer");
                    }
                    break;
                case LONG:
                    try {
                        long val = Long.parseLong(newValue);
                        this.type = type;
                        this.value = val;
                    }
                    catch (NumberFormatException exception) {
                        throw new DynamicParamException("Parameter: " + param + ": Cannot convert value to Long");
                    }
                    break;
                case FLOAT:
                    try {
                        float val = Float.parseFloat(newValue);
                        this.type = type;
                        this.value = val;
                    }
                    catch (NumberFormatException exception) {
                        throw new DynamicParamException("Parameter: " + param + ": Cannot convert value to Float");
                    }
                    break;
                case DOUBLE:
                    try {
                        double val = Double.parseDouble(newValue);
                        this.type = type;
                        this.value = val;
                    }
                    catch (NumberFormatException exception) {
                        throw new DynamicParamException("Parameter: " + param + ": Cannot convert value to Double");
                    }
                    break;
                case CHAR:
                    if (newValue.length() != 1) throw new DynamicParamException("Parameter: " + param + ": Cannot convert value to Character");
                    char val = newValue.charAt(0);
                    this.type = type;
                    this.value = val;
                    break;
                case STRING:
                    this.type = type;
                    this.value = newValue;
                    break;
                case BOOLEAN:
                    if (newValue.equalsIgnoreCase("TRUE")) this.value = true;
                    else if (newValue.equalsIgnoreCase("FALSE")) this.value = false;
                    else throw new DynamicParamException("Parameter: " + param + ": Cannot convert value to Boolean");
                    this.type = type;
                    break;
                case LIST:
                    this.type = type;
                    if (!newValue.startsWith("[") && !newValue.startsWith("]")) throw new DynamicParamException("Parameter: " + param + ": Cannot convert value to List");
                    this.value = newValue.replace("[", "").replace("]", "").replace(" ", "").split(";");
                    break;
            }
        }

        /**
         * Constructor to construct parameter from name, type and value (as object).
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
            if (type == ParamType.LIST && !(value instanceof String)) throw new DynamicParamException("Parameter: " + param + " value not type of List.");
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

        /**
         * Returns value of parameter as list.
         *
         * @param param name of parameter.
         * @return value of parameter.
         * @throws DynamicParamException throws exception if parameter type is not boolean.
         */
        public String[] getValueAsList(String param) throws DynamicParamException {
            if (type != ParamType.LIST) throw new DynamicParamException("Parameter: " + param + " value not type of List");
            return (String[])value;
        }

    }

    /**
     * Hashmap used to store parameter list.
     *
     */
    private final HashMap<String, TypeValue> paramList = new HashMap<>();

    /**
     * Name type pairs of parameters as hashmap.
     *
     */
    private final HashMap<String, ParamType> nameTypes = new HashMap<>();


    /**
     * Constructor to build dynamic parameter list.
     *
     * @param params parameter name value pairs separated by comma.
     * @param nameTypes name type pairs of parameters as string representation.
     * @throws DynamicParamException throws exception if parameters are not provided or parameter is properly defined.
     */
    public DynamicParam(String params, String nameTypes) throws DynamicParamException {
        this(nameTypes);
        setParamsByVal(params);
    }

    /**
     * Constructor for dynamic parameter.
     * Name type value pairs are described in format (name:type), (name:type)...
     *
     * @param nameTypes name type pairs of parameters as string representation.
     * @throws DynamicParamException throws exception if name types are not defined.
     */
    public DynamicParam(String nameTypes) throws DynamicParamException {
        parseNameTypes(nameTypes);
    }

    /**
     * Parses name types from string representation.
     * Name type value pairs are described in format (name:type), (name:type)...
     *
     * @param nameTypes name type string representation.
     * @throws DynamicParamException throws exception if parsing fails.
     */
    private void parseNameTypes(String nameTypes) throws DynamicParamException {
        if (nameTypes == null) throw new DynamicParamException("No name types defined.");
        String[] nameTypesList = nameTypes.split(",");
        if (nameTypesList.length == 0) throw new DynamicParamException("No name types found.");
        for (String nameTypeString : nameTypesList) {
            String nameTypeStringEntry = nameTypeString.trim();
            if (nameTypeStringEntry.isEmpty()) continue;
            if (!nameTypeStringEntry.startsWith("(")) throw new DynamicParamException("Invalid entry for type pair: " + nameTypeStringEntry);
            if (!nameTypeStringEntry.endsWith(")")) throw new DynamicParamException("Invalid entry for type pair: " + nameTypeStringEntry);
            if (!nameTypeStringEntry.contains(":")) throw new DynamicParamException("Invalid entry for type pair: " + nameTypeStringEntry);
            nameTypeStringEntry = nameTypeStringEntry.replace("(","");
            nameTypeStringEntry = nameTypeStringEntry.replace(")","");
            String[] nameTypePair = nameTypeStringEntry.split(":");
            ParamType paramType = switch (nameTypePair[1].trim().toUpperCase()) {
                case "INT" -> ParamType.INT;
                case "LONG" -> ParamType.LONG;
                case "FLOAT" -> ParamType.FLOAT;
                case "DOUBLE" -> ParamType.DOUBLE;
                case "CHAR" -> ParamType.CHAR;
                case "STRING" -> ParamType.STRING;
                case "BOOLEAN" -> ParamType.BOOLEAN;
                case "LIST" -> ParamType.LIST;
                default -> throw new DynamicParamException("Illegal type for name type pair: " + nameTypeString);
            };
            this.nameTypes.put(nameTypePair[0].trim(), paramType);
        }
    }

    /**
     * Constructor for dynamic parameter.
     *
     * @param nameTypes name type pairs of parameters as hashmap.
     * @throws DynamicParamException throws exception if name types are not defined.
     */
    public DynamicParam(HashMap<String, ParamType> nameTypes) throws DynamicParamException {
        if (nameTypes == null || nameTypes.isEmpty()) throw new DynamicParamException("No name types defined.");
        this.nameTypes.putAll(nameTypes);
    }

    /**
     * Constructor for dynamic parameter.
     *
     * @param nameTypes name type pairs of parameters as hashmap.
     * @throws DynamicParamException throws exception if name types are not defined.
     */
    public DynamicParam(HashMap<String, ParamType>[] nameTypes) throws DynamicParamException {
        if (nameTypes == null) throw new DynamicParamException("No name types defined.");
        for (HashMap<String, ParamType> nameTypeEntry : nameTypes) this.nameTypes.putAll(nameTypeEntry);
        if (this.nameTypes.isEmpty()) throw new DynamicParamException("No name types defined.");
    }

    /**
     * Constructor to build dynamic parameter list.
     *
     * @param params parameter name value pairs separated by comma.
     * @param nameTypes name type pairs of parameters as hashmap.
     * @throws DynamicParamException throws exception if parameters are not provided or parameter is properly defined.
     */
    public DynamicParam(String params, HashMap<String, ParamType> nameTypes) throws DynamicParamException {
        this(nameTypes);
        setParamsByVal(params);
    }

    /**
     * Defines parameter based on given name value and name type pairs.
     *
     * @param params parameters to be set as name value pair separated by equal sign.
     * @throws DynamicParamException throws exception if parameter is not properly defined or parameter cannot be cast into defined parameter type.
     */
    public void setParamsByVal(String params) throws DynamicParamException {
        if (params.trim().isEmpty()) return;
        String[] paramsList = params.trim().split(",");
        if (paramsList.length == 0) throw new DynamicParamException("No parameters found.");
        for (String param : paramsList) setParamByVal(param);
    }

    /**
     * Defines parameter based on given name value and name type pairs.
     *
     * @param param parameter to be set as name value pair separated by equal sign.
     * @throws DynamicParamException throws exception if parameter is not properly defined or parameter cannot be cast into defined parameter type.
     */
    public void setParamByVal(String param) throws DynamicParamException {
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
     * Returns value of parameter as list.
     *
     * @param param name of parameter.
     * @return value of parameter as list.
     * @throws DynamicParamException throws exception if parameter by name is not found.
     */
    public String[] getValueAsList(String param) throws DynamicParamException {
        if (!paramList.containsKey(param.trim())) throw new DynamicParamException("No parameter: " + param + " found");
        return paramList.get(param).getValueAsList(param);
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

