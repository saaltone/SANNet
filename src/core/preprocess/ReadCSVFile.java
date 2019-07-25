/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.preprocess;

import utils.DMatrix;
import utils.Matrix;
import utils.SMatrix;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Scanner;

/**
 * Defines class for reading CVS file.
 *
 */
public class ReadCSVFile {

    /**
     * Reads CVS file and transforms comma separated data as input and output sample set.<br>
     * Separates each value separated by separator to own column.<br>
     * Returns input matrix with hash map index 0 and output matrix with hash map index 1.<br>
     *
     * @param fileName name of file to be read.
     * @param separator separator that separates values.
     * @param inputCols columns to be used for input samples.
     * @param outputCols columns to be used for output samples.
     * @param skipRowsFromStart skips this number of rows from start.
     * @param asSparseMatrix returns sample set as sparse matrix (SMatrix).
     * @param inAs2D assumes two dimensional input such as image.
     * @param inRows number of input rows.
     * @param inCols number of input cols (relevant for 2D input).
     * @param outAs2D assumes two dimensional output such as image.
     * @param outRows number of output rows.
     * @param outCols number of output cols (relevant for 2D output).
     * @return structure containing input and output matrices.
     * @throws FileNotFoundException throws exception if file is not found.
     */
    public static HashMap<Integer, LinkedHashMap<Integer, Matrix>> readFile(String fileName, String separator, HashSet<Integer> inputCols, HashSet<Integer> outputCols, int skipRowsFromStart, boolean asSparseMatrix, boolean inAs2D, int inRows, int inCols, boolean outAs2D, int outRows, int outCols) throws FileNotFoundException {
        LinkedHashMap<Integer, Integer> inMap = new LinkedHashMap<>();
        LinkedHashMap<Integer, Integer> outMap = new LinkedHashMap<>();
        int index;
        index = 0;
        for (Integer pos : inputCols) inMap.put(pos, index++);
        index = 0;
        for (Integer pos : outputCols) outMap.put(pos, index++);
        inRows = inAs2D ? inRows : inMap.size();
        inCols = inAs2D ? inCols : 1;
        outRows = outAs2D ? outRows : outMap.size();
        outCols = outAs2D ? outCols : 1;

        File file = new File(fileName);
        Scanner scanner = new Scanner(file);

        LinkedHashMap<Integer, Matrix> inputData = new LinkedHashMap<>();
        LinkedHashMap<Integer, Matrix> outputData = new LinkedHashMap<>();
        int countSkipRows = 0;
        int row = 0;
        while (scanner.hasNextLine()) {
            while (countSkipRows < skipRowsFromStart && scanner.hasNextLine()) {
                scanner.nextLine();
                countSkipRows++;
            }
            if (!scanner.hasNextLine()) break;
            String[] items = scanner.nextLine().split(separator);
            Matrix inItem = !asSparseMatrix ? new DMatrix(inRows, inCols) : new SMatrix(inRows, inCols);
            for (Integer pos : inMap.keySet()) {
                if (items[pos].compareTo("0") != 0) {
                    inItem.setValue(getRow(inAs2D, inMap.get(pos), inCols), getCol(inAs2D, inMap.get(pos), inCols), convertToDouble(items[pos]));
                }
            }
            inputData.put(row, inItem);
            Matrix outItem = !asSparseMatrix ? new DMatrix(outRows, outCols) : new SMatrix(outRows, outCols);
            for (Integer pos : outMap.keySet()) {
                if (items[pos].compareTo("0") != 0) {
                    outItem.setValue(getRow(outAs2D, outMap.get(pos), outCols), getCol(outAs2D, outMap.get(pos), outCols), convertToDouble(items[pos]));
                }
            }
            outputData.put(row, outItem);
            row++;
        }
        HashMap<Integer, LinkedHashMap<Integer, Matrix>> result = new HashMap<>();
        result.put(0, inputData);
        result.put(1, outputData);
        return result;
    }

    /**
     * Returns input row either directly as position or row in 2d input structure.
     *
     * @param as2D if 2D structure is assumed.
     * @param pos raw input position.
     * @param cols number of columns in 2D structure.
     * @return row either directly as position or as row in 2D structure.
     */
    private static int getRow(boolean as2D, int pos, int cols) {
        return !as2D ? pos : pos / cols;
    }

    /**
     * Returns input column either as 0 or column in 2d input structure.
     *
     * @param as2D if 2D structure is assumed.
     * @param pos raw input position.
     * @param cols number of columns in 2D structure.
     * @return column either directly as position or as 0 in 2D structure.
     */
    private static int getCol(boolean as2D, int pos, int cols) {
        return !as2D ? 0 : pos % cols;
    }

    /**
     * Reads CVS file and transforms comma separated data as input and output sample set.<br>
     * Separates each value separated by separator to own column.<br>
     * Returns input matrix with hash map index 0 and output matrix with hash map index 1.<br>
     *
     * @param fileName name of file to be read.
     * @param separator separator that separates values.
     * @param inputCols columns to be used for input samples.
     * @param outputCols columns to be used for output samples.
     * @param skipRowsFromStart skips this number of rows from start.
     * @param asSparseMatrix returns sample set as sparse matrix (SMatrix).
     * @return structure containing input and output matrices.
     * @throws FileNotFoundException throws exception if file is not found.
     */
    public static HashMap<Integer, LinkedHashMap<Integer, Matrix>> readFile(String fileName, String separator, HashSet<Integer> inputCols, HashSet<Integer> outputCols, int skipRowsFromStart, boolean asSparseMatrix) throws FileNotFoundException {
        return readFile(fileName, separator, inputCols, outputCols, skipRowsFromStart, asSparseMatrix, false, 0, 0, false, 0, 0);
    }

    /**
     * Reads CVS file and transforms comma separated data as input and output sample set.<br>
     * Separates each value separated by separator to own column. Assumes ";" as separator.<br>
     * Returns input matrix with hash map index 0 and output matrix with hash map index 1.<br>
     *
     * @param fileName name of file to be read.
     * @param inputCols columns to be used for input samples.
     * @param outputCols columns to be used for input samples.
     * @param asSparseMatrix returns sample set as sparse matrix (SMatrix).
     * @return structure containing input and output matrices.
     * @throws FileNotFoundException throws exception if file is not found.
     */
    public static HashMap<Integer, LinkedHashMap<Integer, Matrix>> readFile(String fileName, HashSet<Integer> inputCols, HashSet<Integer> outputCols, boolean asSparseMatrix) throws FileNotFoundException {
        return readFile(fileName, ";", inputCols, outputCols, 0, asSparseMatrix, false, 0, 0, false, 0, 0);
    }

    private static Double convertToDouble(String item) {
        double value = 0;
        try {
            value = Double.parseDouble(item);
        }
        catch(NumberFormatException e) {}
        return value;
    }

}
