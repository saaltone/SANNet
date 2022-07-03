/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.preprocess;

import utils.matrix.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Scanner;

/**
 * Implements functionality for reading CVS file.<br>
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
     * @param inputColumns columns to be used for input samples.
     * @param outputColumns columns to be used for output samples.
     * @param skipRowsFromStart skips this number of rows from start.
     * @param asSparseMatrix returns sample set as sparse matrix (SMatrix).
     * @param inAs2D assumes 2-dimensional input such as image.
     * @param inRows number of input rows.
     * @param inCols number of input cols (relevant for 2D input).
     * @param outAs2D assumes 2-dimensional output such as image.
     * @param outRows number of output rows.
     * @param outCols number of output cols (relevant for 2D output).
     * @return structure containing input and output matrices.
     * @throws FileNotFoundException throws exception if file is not found.
     * @throws MatrixException throws exception if matrix is exceeding its depth or matrix is not defined.
     */
    public static HashMap<Integer, HashMap<Integer, MMatrix>> readFile(String fileName, String separator, HashSet<Integer> inputColumns, HashSet<Integer> outputColumns, int skipRowsFromStart, boolean asSparseMatrix, boolean inAs2D, int inRows, int inCols, boolean outAs2D, int outRows, int outCols) throws FileNotFoundException, MatrixException {
        HashMap<Integer, Integer> inputColumnMap = new HashMap<>();
        HashMap<Integer, Integer> outputColumnMap = new HashMap<>();
        int index;
        index = 0;

        for (Integer pos : inputColumns) inputColumnMap.put(pos, index++);

        index = 0;
        for (Integer pos : outputColumns) outputColumnMap.put(pos, index++);

        inRows = inAs2D ? inRows : inputColumnMap.size();
        inCols = inAs2D ? inCols : 1;

        outRows = outAs2D ? outRows : outputColumnMap.size();
        outCols = outAs2D ? outCols : 1;

        File file = new File(fileName);
        Scanner scanner = new Scanner(file);

        int countSkipRows = 0;
        while (countSkipRows++ < skipRowsFromStart && scanner.hasNextLine()) scanner.nextLine();

        HashMap<Integer, MMatrix> inputData = new HashMap<>();
        HashMap<Integer, MMatrix> outputData = new HashMap<>();

        int row = 0;
        while (scanner.hasNextLine()) {
            String[] items = scanner.nextLine().split(separator);

            addItem(items, inputColumnMap, row, inAs2D, inRows, inCols, asSparseMatrix, inputData);

            addItem(items, outputColumnMap, row, outAs2D, outRows, outCols, asSparseMatrix, outputData);

            row++;
        }
        HashMap<Integer, HashMap<Integer, MMatrix>> result = new HashMap<>();
        result.put(0, inputData);
        result.put(1, outputData);
        return result;
    }

    /**
     * Adds item to data
     *
     * @param items items
     * @param columnMap column map
     * @param row row
     * @param as2D if true 2D structure is assumed
     * @param rows rows
     * @param columns columns
     * @param asSparseMatrix if true sparse matrix usage is assumed
     * @param data data
     * @throws MatrixException throws exception if matrix is exceeding its depth or matrix is not defined.
     */
    private static void addItem(String[] items, HashMap<Integer, Integer> columnMap, int row, boolean as2D, int rows, int columns, boolean asSparseMatrix, HashMap<Integer, MMatrix> data) throws MatrixException {
        Matrix inItem = !asSparseMatrix ? new DMatrix(rows, columns) : new SMatrix(rows, columns);
        for (Map.Entry<Integer, Integer> entry : columnMap.entrySet()) {
            int pos = entry.getKey();
            int value = entry.getValue();
            if (items[pos].compareTo("0") != 0) {
                inItem.setValue(getRow(as2D, value, columns), getCol(as2D, value, columns), convertToDouble(items[pos]));
            }
        }
        data.put(row, new MMatrix(inItem));
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
     * @throws MatrixException throws exception if matrix is exceeding its depth or matrix is not defined.
     */
    public static HashMap<Integer, HashMap<Integer, MMatrix>> readFile(String fileName, String separator, HashSet<Integer> inputCols, HashSet<Integer> outputCols, int skipRowsFromStart, boolean asSparseMatrix) throws FileNotFoundException, MatrixException {
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
     * @throws MatrixException throws exception if matrix is exceeding its depth or matrix is not defined.
     */
    public static HashMap<Integer, HashMap<Integer, MMatrix>> readFile(String fileName, HashSet<Integer> inputCols, HashSet<Integer> outputCols, boolean asSparseMatrix) throws FileNotFoundException, MatrixException {
        return readFile(fileName, ";", inputCols, outputCols, 0, asSparseMatrix, false, 0, 0, false, 0, 0);
    }

    /**
     * Converts string to double value.
     *
     * @param item string to be converted.
     * @return converted double value.
     */
    private static Double convertToDouble(String item) {
        double value = 0;
        try {
            value = Double.parseDouble(item);
        }
        catch (NumberFormatException numberFormatException) {
        }
        return value;
    }

}
