/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.preprocess;

import utils.Matrix;
import utils.SMatrix;

import java.util.HashMap;
import java.util.LinkedHashMap;

/**
 * Class for one hot encoding data.
 *
 */
public class OneHotEncoder {

    /**
     * Mapping to store value index pairs for one hot encoding.
     *
     */
    private HashMap<Integer, LinkedHashMap<Double, Integer>> mapping = new HashMap<>();

    /**
     * Stores current maximum index for one hot encoder.
     *
     */
    private int maxIndex = 0;

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
     */
    public LinkedHashMap<Integer, Matrix> encode(LinkedHashMap<Integer, Matrix> input, boolean keepMapping) {
        if (input.size() == 0) return new LinkedHashMap<>();
        LinkedHashMap<Integer, LinkedHashMap<Integer, Integer>> itemsMap = new LinkedHashMap<>();
        if (!keepMapping) {
            mapping = new LinkedHashMap<>();
            maxIndex = 0;
        }
        for (int itemRow = 0; itemRow < input.get(0).getRows(); itemRow++) {
            LinkedHashMap<Double, Integer> itemMapping;
            if (mapping.containsKey(itemRow)) itemMapping = mapping.get(itemRow);
            else mapping.put(itemRow, itemMapping = new LinkedHashMap<>());
            for (int row = 0; row < input.size(); row++) {
                Double item = input.get(row).getValue(itemRow, 0);
                int index = -1;
                if (itemMapping.containsKey(item)) index = itemMapping.get(item);
                else if (!keepMapping) itemMapping.put(item, index = maxIndex++);
                if (index != -1) {
                    LinkedHashMap<Integer, Integer> itemMap;
                    if (itemsMap.containsKey(row)) itemMap = itemsMap.get(row);
                    else itemsMap.put(row, itemMap = new LinkedHashMap<>());
                    itemMap.put(itemRow, index);
                }
            }
        }
        LinkedHashMap<Integer, Matrix> output = new LinkedHashMap<>();
        if (maxIndex == 0) return output;
        for (Integer row : itemsMap.keySet()) {
            Matrix item = new SMatrix(maxIndex, 1);
            for (Integer itemRow : itemsMap.get(row).keySet()) {
                item.setValue(itemsMap.get(row).get(itemRow), 0, 1);
            }
            output.put(row, item);
        }
        return output;
    }
}
