/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.preprocess;

import core.NeuralNetworkException;
import utils.Sample;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;

/**
 * Class used to split sample set into training and test sample sets by given share.<br>
 * Does split directly in initial sample order or first by randomizing set.<br>
 *
 */
public class DataSplitter {

    /**
     * Splits data into training and test sample sets.
     *
     * @param data initial sample set to be split.
     * @param testDataShare share 0% - 100% of test data (0 - 1) from initial sample set.
     * @param randomize true if initial sample set order is randomized before split otherwise false.
     * @return split training and test sample tests.
     * @throws NeuralNetworkException throws exception if invalid inputs are given.
     */
    public static HashMap<Integer, LinkedHashMap<Integer, Sample>> split(HashMap<Integer, LinkedHashMap<Integer, Sample>> data, double testDataShare, boolean randomize) throws NeuralNetworkException {
        if (testDataShare < 0 || testDataShare > 1) throw new NeuralNetworkException("Invalid test data share: " + testDataShare + ". demo.Test data share must be between 0 and 1");
        if (data.size() != 2) throw new NeuralNetworkException("Split must have input and output data.");
        if (data.get(0).size() != data.get(1).size()) {
            throw new NeuralNetworkException("Input data size: " + data.get(0).size() + " and output data size: " + data.get(1).size() + " are not matching: ");
        }
        int trainDataAmount = (int)((double)data.get(0).size() * (1 - testDataShare));
        HashMap<Integer, LinkedHashMap<Integer, Sample>> result = new HashMap<>();
        result.put(0, new LinkedHashMap<>());
        result.put(1, new LinkedHashMap<>());
        result.put(2, new LinkedHashMap<>());
        result.put(3, new LinkedHashMap<>());
        ArrayList<Integer> itemList = new ArrayList<>();
        int rowAmount = data.get(0).size();
        for (int index = 0; index < rowAmount; index++) itemList.add(index);
        if (randomize) Collections.shuffle(itemList);
        int trainIndex = 0;
        int testIndex = 0;
        for (int index = 0; index < rowAmount; index++) {
            if (index < trainDataAmount) {
                result.get(0).put(trainIndex, data.get(0).get(itemList.get(index)));
                result.get(1).put(trainIndex++, data.get(1).get(itemList.get(index)));
            }
            else {
                result.get(2).put(testIndex, data.get(0).get(itemList.get(index)));
                result.get(3).put(testIndex++, data.get(1).get(itemList.get(index)));
            }
        }
        return result;
    }

}

