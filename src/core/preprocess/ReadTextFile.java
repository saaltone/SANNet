/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.preprocess;

import utils.matrix.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;

/**
 * Implements functionality for reading text files.<br>
 * Reads text file and maps each character as separate input or output columns (one-hot encodes characters).<br>
 * Number of characters in input and output can be specified.<br>
 *
 */
public class ReadTextFile {

    /**
     * Reads text file and maps each character as separate input and output columns.<br>
     * Returns input matrix with hash map index 0 and output matrix with hash map index 1.<br>
     *
     * @param fileName name of file to be read.
     * @param numberOfInputCharacters number of characters per input column.
     * @param numberOfOutputCharacters number of characters per output column.
     * @param inputOutputDelta delta in number of characters between start of input and output.
     * @param skipRowsFromStart skips specified number of rows from start.
     * @return structure containing input and output matrices.
     * @throws FileNotFoundException throws exception if file is not found.
     */
    public static HashMap<Integer, HashMap<Integer, MMatrix>> readFile(String fileName, int numberOfInputCharacters, int numberOfOutputCharacters, int inputOutputDelta, int skipRowsFromStart) throws FileNotFoundException {
        StringBuilder text = readText(fileName, skipRowsFromStart);

        HashMap<Integer, HashMap<Integer, Integer>> inputData = new HashMap<>();
        HashMap<Integer, HashMap<Integer, Integer>> outputData = new HashMap<>();
        int length = Math.min(text.length() - numberOfInputCharacters, text.length() - numberOfOutputCharacters - inputOutputDelta) + 1;
        for (int pos = 0; pos < length; pos++) {
            HashMap<Integer, Integer> inValues = new HashMap<>();
            for (int inCount = 0; inCount < numberOfInputCharacters; inCount++) {
                int charAt = charToInt(text.charAt(pos + inCount));
                inValues.put(inCount, charAt);
            }
            inputData.put(pos, inValues);

            HashMap<Integer, Integer> outValues = new HashMap<>();
            for (int outCount = 0; outCount < numberOfOutputCharacters; outCount++) {
                int charAt = charToInt(text.charAt(pos + outCount + inputOutputDelta));
                outValues.put(outCount, charAt);
            }
            outputData.put(pos, outValues);
        }
        HashMap<Integer, MMatrix> inputs = new HashMap<>();
        HashMap<Integer, MMatrix> outputs = new HashMap<>();
        HashMap<Integer, HashMap<Integer, MMatrix>> result = new HashMap<>();
        result.put(0, inputs);
        result.put(1, outputs);

        int charSize = charSize();
        for (Integer pos : inputData.keySet()) {
            Matrix input = new SMatrix(numberOfInputCharacters * charSize, 1);
            for (Integer index : inputData.get(pos).keySet()) {
                int charAt = inputData.get(pos).get(index);
                input.setValue(charAt + index * charSize, 0, 1);
            }
            inputs.put(pos, new MMatrix(input));
        }

        for (Integer pos : inputData.keySet()) {
            Matrix output = new SMatrix(numberOfOutputCharacters * charSize, 1);
            for (Integer index : outputData.get(pos).keySet()) {
                int charAt = outputData.get(pos).get(index);
                int col = charAt + index * charSize;
                output.setValue(col, 0,1);
            }
            outputs.put(pos, new MMatrix(output));
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
    public static char intToChar(int intAt) {
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
        return (char)mappedChar;
    }

    /**
     * Size of characters as one hot encoded items.
     *
     * @return size of character.
     */
    public static int charSize() {
        return 44;
    }

    /**
     * Reads text file and maps each word as separate binary encoded input and one-hot encoded output.<br>
     *
     * @param fileName name of file to be read.
     * @param numberOfInputWords number of words per input column.
     * @param skipRowsFromStart skips specified number of rows from start.
     * @param dictionaryIndexMapping dictionary index mapping.
     * @return structure containing input and output matrices.
     * @throws FileNotFoundException throws exception if file is not found.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public static HashMap<Integer, HashMap<Integer, MMatrix>> readFileAsBinaryEncoded(String fileName, int numberOfInputWords, int skipRowsFromStart, HashMap<Integer, String> dictionaryIndexMapping) throws FileNotFoundException, MatrixException {
        StringBuilder readText = readText(fileName, skipRowsFromStart);
        String[] words = readText.toString().split(" ");
        Arrays.setAll(words, index -> words[index].trim());
        Arrays.setAll(words, index -> words[index].replaceAll("[^a-zäöåA-ZÄÖÅ]", "").toLowerCase());
        TreeSet<String> dictionary = new TreeSet<>();
        Collections.addAll(dictionary, words);
        int maxBits = ComputableMatrix.numberOfBits(dictionary.size());
        HashMap<Matrix, Integer> dictionaryBinaryIndexMapping = new HashMap<>();
        HashMap<String, Matrix> dictionaryStringBinaryIndexMapping = new HashMap<>();
        int index = 0;
        int dictionarySize = dictionary.size();
        for (String word : dictionary) {
            dictionaryIndexMapping.put(index, word);
            Matrix binaryMatrix = ComputableMatrix.encodeToBitColumnVector(index, maxBits);
            dictionaryBinaryIndexMapping.put(binaryMatrix, index);
            dictionaryStringBinaryIndexMapping.put(word, binaryMatrix);
            index++;
        }

        ArrayList<Matrix> encodedWords = new ArrayList<>();
        for (String word : words) {
            if (!word.equals("")) {
                encodedWords.add(dictionaryStringBinaryIndexMapping.get(word));
            }
        }

        HashMap<Integer, MMatrix> inputs = new HashMap<>();
        HashMap<Integer, MMatrix> outputs = new HashMap<>();
        HashMap<Integer, HashMap<Integer, MMatrix>> result = new HashMap<>();
        result.put(0, inputs);
        result.put(1, outputs);

        ArrayDeque<Matrix> encodedWordQueue = new ArrayDeque<>();
        LinkedList<Matrix> encodedWordList = new LinkedList<>();
        int pos = 0;
        for (Matrix encodedWord : encodedWords) {
            encodedWordQueue.addLast(encodedWord);
            encodedWordList.addLast(encodedWord);

            if (encodedWordQueue.size() >= numberOfInputWords + 1) {
                ArrayList<Matrix> inputMatrices = new ArrayList<>();
                ArrayList<Matrix> outputMatrices = new ArrayList<>();
                int wordIndex = 0;
                for (Matrix matrix : encodedWordList) {
                    if (wordIndex++ < encodedWordList.size() - 1) inputMatrices.add(matrix);
                    else {
                        Matrix outputMatrix = new DMatrix(dictionarySize, 1);
                        outputMatrix.setValue(dictionaryBinaryIndexMapping.get(matrix), 0, 1);
                        outputMatrices.add(outputMatrix);
                    }
                }

                JMatrix joinedInputMatrix = new JMatrix(inputMatrices, true);
                JMatrix joinedOutputMatrix = new JMatrix(outputMatrices, true);
                inputs.put(pos, new MMatrix(joinedInputMatrix));
                outputs.put(pos, new MMatrix(joinedOutputMatrix));
                pos++;

                encodedWordQueue.removeFirst();
                encodedWordList.removeFirst();
            }
        }

        return result;
    }

    /**
     * Read and returns text from file. Converts text to lower case.
     *
     * @param fileName file name
     * @param skipRowsFromStart number of rows skipped from start.
     * @return text from file.
     * @throws FileNotFoundException throws exception if file is not found.
     */
    public static StringBuilder readText(String fileName, int skipRowsFromStart) throws FileNotFoundException {
        File file = new File(fileName);
        Scanner scanner = new Scanner(file);
        int countSkipRows = 0;
        StringBuilder text = new StringBuilder();
        while (scanner.hasNextLine()) {
            while (countSkipRows < skipRowsFromStart) {
                scanner.nextLine();
                countSkipRows++;
            }
            String line = scanner.nextLine();
            text.append(line).append(" ");
        }
        text = new StringBuilder(text.toString().toLowerCase());
        return text;
    }

}
