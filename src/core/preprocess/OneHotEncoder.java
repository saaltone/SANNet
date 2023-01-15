/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.preprocess;

import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.SMatrix;

import java.util.HashMap;
import java.util.Map;

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
            int matrixDepth = sample.getDepth();
            for (int inputDepth = 0; inputDepth < matrixDepth; inputDepth++) {
                Matrix entry = sample.get(inputDepth);
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
            for (Map.Entry<Integer, MMatrix> entry : input.entrySet()) {
                int index = entry.getKey();
                MMatrix sample = entry.getValue();
                HashMap<Integer, HashMap<Integer, Integer>> sampleMapping;
                if (itemsMap.containsKey(index)) sampleMapping = itemsMap.get(index);
                else itemsMap.put(index, sampleMapping = new HashMap<>());
                int depth = sample.getDepth();
                for (int depthIndex = 0; depthIndex < depth; depthIndex++) {
                    Matrix item = sample.get(depthIndex);
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
        for (Map.Entry<Integer, HashMap<Integer, HashMap<Integer, Integer>>> entry : itemsMap.entrySet()) {
            int index = entry.getKey();
            HashMap<Integer, HashMap<Integer, Integer>> itemEntries = entry.getValue();
            for (Map.Entry<Integer, HashMap<Integer, Integer>> entry1 : itemEntries.entrySet()) {
                HashMap<Integer, Integer> itemEntry = entry1.getValue();
                Matrix item = new SMatrix(prevMaxKey, 1);
                for (Integer row : itemEntry.values()) {
                    item.setValue(row, 0, 1);
                }
                output.put(index, new MMatrix(item));
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
