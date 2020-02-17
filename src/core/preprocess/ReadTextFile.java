/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package core.preprocess;

import utils.matrix.Matrix;
import utils.matrix.SMatrix;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Scanner;

/**
 * Defines class for reading text file.<br>
 * Reads text file and maps each character as separate input or output columns.<br>
 * Number of characters in input and output can be specified.<br>
 *
 */
public class ReadTextFile {

    /**
     * Reads text file and maps each character as separate input and output columns.<br>
     * Returns input matrix with hash map index 0 and output matrix with hash map index 1.<br>
     *
     * @param fileName name of file to be read.
     * @param numOfCharsIn number of characters per input column.
     * @param numOfCharsOut number of characters per output column.
     * @param deltaInOut delta in number of characters between start of input and output.
     * @param skipRowsFromStart skips specified number of rows from start.
     * @return structure containing input and output matrices.
     * @throws FileNotFoundException throws exception if file is not found.
     */
    public static HashMap<Integer, LinkedHashMap<Integer, Matrix>> readFile(String fileName, int numOfCharsIn, int numOfCharsOut, int deltaInOut, int skipRowsFromStart) throws FileNotFoundException {
        File file = new File(fileName);

        Scanner scanner = new Scanner(file);

        LinkedHashMap<Integer, LinkedHashMap<Integer, Integer>> inputData = new LinkedHashMap<>();
        LinkedHashMap<Integer, LinkedHashMap<Integer, Integer>> outputData = new LinkedHashMap<>();
        int countSkipRows = 0;
        StringBuilder text = new StringBuilder();
        while (scanner.hasNextLine()) {
            while (countSkipRows < skipRowsFromStart) {
                scanner.nextLine();
                countSkipRows++;
            }
            String line = scanner.nextLine();
            text.append(line);
        }
        text = new StringBuilder(text.toString().toLowerCase());

        int length = Math.min(text.length() - numOfCharsIn, text.length() - numOfCharsOut - deltaInOut) + 1;
        for (int pos = 0; pos < length; pos++) {
            LinkedHashMap<Integer, Integer> inVals = new LinkedHashMap<>();
            for (int inCount = 0; inCount < numOfCharsIn; inCount++) {
                int charAt = charToInt(text.charAt(pos + inCount));
                inVals.put(inCount, charAt);
            }
            inputData.put(pos, inVals);

            LinkedHashMap<Integer, Integer> outVals = new LinkedHashMap<>();
            for (int outCount = 0; outCount < numOfCharsOut; outCount++) {
                int charAt = charToInt(text.charAt(pos + outCount + deltaInOut));
                outVals.put(outCount, charAt);
            }
            outputData.put(pos, outVals);
        }
        LinkedHashMap<Integer, Matrix> inputs = new LinkedHashMap<>();
        LinkedHashMap<Integer, Matrix> outputs = new LinkedHashMap<>();
        HashMap<Integer, LinkedHashMap<Integer, Matrix>> result = new HashMap<>();
        result.put(0, inputs);
        result.put(1, outputs);

        int charSize = charSize();
        for (Integer pos : inputData.keySet()) {
            Matrix input = new SMatrix(numOfCharsIn * charSize, 1);
            for (Integer index : inputData.get(pos).keySet()) {
                int charAt = inputData.get(pos).get(index);
                input.setValue(charAt + index * charSize, 0, 1);
            }
            inputs.put(pos, input);
        }

        for (Integer pos : inputData.keySet()) {
            Matrix output = new SMatrix(numOfCharsOut * charSize, 1);
            for (Integer index : outputData.get(pos).keySet()) {
                int charAt = outputData.get(pos).get(index);
                int col = charAt + index * charSize;
                output.setValue(col, 0,1);
            }
            outputs.put(pos, output);
        }

        return result;
    }

    /**
     * Remaps specific character values to integer values.
     *
     * @param charAt character to be mapped.
     * @return mapped character value.
     */
    public static int charToInt(int charAt) {
        int mappedChar = 0;
        if (charAt >= 48 && charAt <= 57) mappedChar = charAt - 47;
        if (charAt >= 97 && charAt <= 122) mappedChar = charAt - 86;
        if (charAt == 33) mappedChar = 37;
        if (charAt == 39) mappedChar = 38;
        if (charAt == 44) mappedChar = 39;
        if (charAt == 46) mappedChar = 40;
        if (charAt == 63) mappedChar = 41;
        if (charAt == 228) mappedChar = 42;
        if (charAt == 246) mappedChar = 43;
        return mappedChar;
    }

    /**
     * Remaps specific integer value to character value.
     *
     * @param intAt integer to be mapped.
     * @return mapped character value.
     */
    public static int intTochar(int intAt) {
        int mappedChar = 32;
        if (intAt >= 1 && intAt <= 10) mappedChar = intAt + 47;
        if (intAt >= 11 && intAt <= 36) mappedChar = intAt + 86;
        if (intAt == 37) mappedChar = 33;
        if (intAt == 38) mappedChar = 39;
        if (intAt == 39) mappedChar = 44;
        if (intAt == 40) mappedChar = 46;
        if (intAt == 41) mappedChar = 63;
        if (intAt == 42) mappedChar = 228;
        if (intAt == 43) mappedChar = 246;
        return mappedChar;
    }

    /**
     * Size of characters as one hot encoded items.
     *
     * @return size of character.
     */
    public static int charSize() {
        return 44;
    }

}
