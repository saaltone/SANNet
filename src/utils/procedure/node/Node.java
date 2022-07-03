/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.procedure.node;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * Defines interface for node.
 *
 */
public interface Node {

    /**
     * If true node is of type multi index.
     *
     * @return true node is of type multi index.
     */
    boolean isMultiIndex();

    /**
     * Returns true if given matrix is reference matrix of this node.
     *
     * @param matrix given matrix
     * @return true if given matrix is reference matrix of this node.
     */
    boolean isReferenceOf(Matrix matrix);

    /**
     * Returns true is reference matrix is of scalar type otherwise false.
     *
     * @return true is reference matrix is of scalar type otherwise false.
     */
    boolean isScalar();

    /**
     * Returns number of rows in reference matrix.
     *
     * @return number of rows in reference matrix.
     */
    int getRows();

    /**
     * Returns number of columns in reference matrix.
     *
     * @return number of columns in reference matrix.
     */
    int getColumns();

    /**
     * Set dependency node backward.
     *
     * @param fromNode from node.
     */
    void setFromNode(Node fromNode);

    /**
     * Set dependency node forward.
     *
     * @param toNode to node.
     */
    void setToNode(Node toNode);

    /**
     * Updates matrix dependency to forward direction.
     *
     * @param index index
     * @throws MatrixException throws exception if scalar type of node and matrix are not matching or node is of type multi-index.
     */
    void updateMatrixDependency(int index) throws MatrixException;

    /**
     * Updates gradient dependency to backward direction.
     *
     * @param index index
     * @throws MatrixException throws exception if scalar type of node and matrix are not matching or node is of type multi-index.
     */
    void updateGradientDependency(int index) throws MatrixException;

    /**
     * Stores matrix dependency
     *
     * @param backupIndex backup index
     */
    void storeMatrixDependency(int backupIndex);

    /**
     * Restores matrix dependency.
     *
     * @param backupIndex backup index.
     */
    void restoreMatrixDependency(int backupIndex);

    /**
     * Return name of node
     *
     * @return name of node
     */
    String getName();

    /**
     * Sets if gradient is updated for this node. If true gradient is not updated otherwise it is updated.
     *
     * @param stopGradient if true gradient is not updated otherwise it is updated.
     */
    void setStopGradient(boolean stopGradient);

    /**
     * Returns if gradient is updated for this node. If true gradient is not updated otherwise it is updated.
     *
     * @return if true gradient is not updated otherwise it is updated.
     */
    boolean isStopGradient();

    /**
     * Returns size of node.
     *
     * @return size of node.
     */
    int size();

    /**
     * Returns key set of node.
     *
     * @return key set of node.
     */
    Set<Integer> keySet();

    /**
     * Returns entry set of node.
     *
     * @return entry set of node.
     */
    Set<Map.Entry<Integer, Matrix>> entrySet();

    /**
     * Checks if node contains specific matrix.
     *
     * @param matrix specific matrix.
     * @return returns true if node contains specific matrix.
     */
    boolean contains(Matrix matrix);

    /**
     * Resets node and removes other data than constant data.
     *
     * @param resetDependentNodes if true resets also dependent nodes.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    void reset(boolean resetDependentNodes) throws MatrixException;

    /**
     * Returns new matrix with dimensions of reference matrix.
     *
     * @return new matrix with dimensions of reference matrix.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    Matrix getNewMatrix() throws MatrixException;

    /**
     * Returns new matrix of node.
     *
     * @param index data index for matrix.
     * @return matrix of node.
     * @throws MatrixException throws exception if scalar type of node and matrix are not matching.
     */
    Matrix getNewMatrix(int index) throws MatrixException;

    /**
     * Sets matrix of this node.
     *
     * @param matrix new matrix.
     * @throws MatrixException throws exception if scalar type of node and matrix are not matching or node is of type multi-index.
     */
    void setMatrix(Matrix matrix) throws MatrixException;

    /**
     * Sets matrix of this node.
     *
     * @param index data index for matrix.
     * @param matrix new matrix.
     * @throws MatrixException throws exception if scalar type of node and matrix are not matching.
     */
    void setMatrix(int index, Matrix matrix) throws MatrixException;

    /**
     * Returns matrix of node.
     *
     * @return matrix of node.
     */
    Matrix getMatrix();

    /**
     * Returns matrix of node.
     *
     * @param index data index for matrix.
     * @return matrix of node.
     */
    Matrix getMatrix(int index);

    /**
     * Returns matrices of node.
     *
     * @return matrices of node.
     */
    TreeMap<Integer, Matrix> getMatrices();

    /**
     * Sets gradient matrix of node.
     *
     * @param index data index for gradient.
     * @param gradient gradient matrix of node.
     */
    void setGradient(int index, Matrix gradient);

    /**
     * Returns gradient matrix of node.
     *
     * @return gradient matrix of node.
     */
    Matrix getGradient();

    /**
     * Returns gradient matrix of node.
     *
     * @param index data index of gradient.
     * @return gradient matrix of node.
     */
    Matrix getGradient(int index);

    /**
     * Returns gradient mean (average).
     *
     * @return gradient mean (average).
     * @throws MatrixException throws exception if matrix operation fails.
     */
    Matrix getGradientMean() throws MatrixException;

    /**
     * Cumulates gradient.
     *
     * @param index data index.
     * @param outputGradient output gradient.
     * @param negateGradient if true output gradient contribution is negated prior being cumulated.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void cumulateGradient(int index, Matrix outputGradient, boolean negateGradient) throws MatrixException;

}
