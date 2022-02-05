/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.preprocess;

import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.SMatrix;

import java.util.HashMap;

/**
 * Implements functionality for one hot encoding data.<br>
 *
 */
public class OneHotEncoder {

    /**
     * Mapping to store value index pairs for one hot encoding.
     *
     */
    private HashMap<Integer, HashMap<Double, Integer>> mapping = new HashMap<>();

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
    public HashMap<Integer, MMatrix> encode(HashMap<Integer, MMatrix> input, boolean keepMapping) throws MatrixException {
        if (input.size() == 0) return new HashMap<>();
        HashMap<Integer, HashMap<Integer, HashMap<Integer, Integer>>> itemsMap = new HashMap<>();
        if (!keepMapping) mapping = new HashMap<>();
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
            HashMap<Double, Integer> rowMapping;
            if (mapping.containsKey(row)) rowMapping = mapping.get(row);
            else mapping.put(row, rowMapping = new HashMap<>());
            for (Integer entry : input.keySet()) {
                MMatrix sample = input.get(entry);
                HashMap<Integer, HashMap<Integer, Integer>> sampleMapping;
                if (itemsMap.containsKey(entry)) sampleMapping = itemsMap.get(entry);
                else itemsMap.put(entry, sampleMapping = new HashMap<>());
                for (Integer depth : sample.keySet()) {
                    Matrix item = sample.get(depth);
                    HashMap<Integer, Integer> entryMapping;
                    if (sampleMapping.containsKey(depth)) entryMapping = sampleMapping.get(depth);
                    else sampleMapping.put(depth, entryMapping = new HashMap<>());
                    int mappingKey = getMappingKey(rowMapping, item.getValue(row, 0));
                    entryMapping.put(row, curMaxKey + mappingKey);
                    prevMaxKey = Math.max(prevMaxKey, mappingKey + 1);
                }
            }
        }
        HashMap<Integer, MMatrix> output = new HashMap<>();
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
