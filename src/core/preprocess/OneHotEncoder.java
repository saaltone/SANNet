/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.preprocess;

import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.SMatrix;

import java.util.HashMap;
import java.util.LinkedHashMap;

/**
 * Class for one hot encoding data.<br>
 *
 */
public class OneHotEncoder {

    /**
     * Mapping to store value index pairs for one hot encoding.
     *
     */
    private HashMap<Integer, LinkedHashMap<Double, Integer>> mapping = new HashMap<>();

    /**
     * Default constructor for one hot encoder.
     *
     */
    public OneHotEncoder() {
    }

    /**
     * One hot encodes given sample set.
     *
     * @param input sample set to be encoded.
     * @param keepMapping if true keeps earlier value index mapping in encoding phase.
     * @return one hot encoded sample set.
     * @throws MatrixException throws matrix exception is creation of sample fails.
     */
    public LinkedHashMap<Integer, MMatrix> encode(LinkedHashMap<Integer, MMatrix> input, boolean keepMapping) throws MatrixException {
        if (input.size() == 0) return new LinkedHashMap<>();
        LinkedHashMap<Integer, LinkedHashMap<Integer, LinkedHashMap<Integer, Integer>>> itemsMap = new LinkedHashMap<>();
        if (!keepMapping) mapping = new LinkedHashMap<>();
        int rows = -1;
        for (MMatrix sample : input.values()) {
            for (Matrix entry : sample.values()) {
                if (rows == -1) rows = entry.getColumns();
                else if(rows != entry.getColumns()) throw new MatrixException("Inconsistent number of columns in input.");
            }
        }
        int prevMaxKey = 0;
        int curMaxKey;
        for (int row = 0; row < rows; row++) {
            curMaxKey = prevMaxKey;
            LinkedHashMap<Double, Integer> rowMapping;
            if (mapping.containsKey(row)) rowMapping = mapping.get(row);
            else mapping.put(row, rowMapping = new LinkedHashMap<>());
            for (Integer entry : input.keySet()) {
                MMatrix sample = input.get(entry);
                LinkedHashMap<Integer, LinkedHashMap<Integer, Integer>> sampleMapping;
                if (itemsMap.containsKey(entry)) sampleMapping = itemsMap.get(entry);
                else itemsMap.put(entry, sampleMapping = new LinkedHashMap<>());
                for (Integer depth : sample.keySet()) {
                    Matrix item = sample.get(depth);
                    LinkedHashMap<Integer, Integer> entryMapping;
                    if (sampleMapping.containsKey(depth)) entryMapping = sampleMapping.get(depth);
                    else sampleMapping.put(depth, entryMapping = new LinkedHashMap<>());
                    int mappingKey = getMappingKey(rowMapping, item.getValue(row, 0));
                    entryMapping.put(row, curMaxKey + mappingKey);
                    prevMaxKey = Math.max(prevMaxKey, mappingKey + 1);
                }
            }
        }
        LinkedHashMap<Integer, MMatrix> output = new LinkedHashMap<>();
        for (Integer entry : itemsMap.keySet()) {
            MMatrix sample = new MMatrix(itemsMap.get(entry).size());
            for (Integer depth : itemsMap.get(entry).keySet()) {
                Matrix item = new SMatrix(prevMaxKey, 1);
                sample.put(depth, item);
                for (Integer row : itemsMap.get(entry).get(depth).values()) {
                    item.setValue(row, 0, 1);
                }
                output.put(entry, new MMatrix(item));
            }
        }
        return output;
    }

    /**
     * Return mapping key corresponding to a specific value.
     *
     * @param mapping current mapping.
     * @param value value.
     * @return mapping key corresponding to a specific value.
     */
    private int getMappingKey(HashMap<Double, Integer> mapping, double value) {
        if (mapping.containsKey(value)) return mapping.get(value);
        else {
            int newMappingKey = mapping.size();
            mapping.put(value, newMappingKey);
            return newMappingKey;
        }
    }

}
