/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package utils.procedure.node;

import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.Set;

/**
 * Defines interface for Node.
 *
 */
public interface Node {

    /**
     * Returns reference matrix.
     *
     * @return reference matrix.
     */
    Matrix getReferenceMatrix();

    /**
     * Returns id of node.
     *
     * @return id of node.
     */
    int getId();

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
     * Returns from node.
     *
     * @return from node.
     */
    Node getFromNode();

    /**
     * Set dependency node forward.
     *
     * @param toNode to node.
     */
    void setToNode(Node toNode);

    /**
     * Returns to node.
     *
     * @return to node.
     */
    Node getToNode();

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
     * @throws MatrixException throws exception if storing dependency fails.
     */
    void storeMatrixDependency(int backupIndex) throws MatrixException;

    /**
     * Restores matrix dependency.
     *
     * @param backupIndex backup index.
     * @throws MatrixException throws exception if restoring of backup fails.
     */
    void restoreMatrixDependency(int backupIndex) throws MatrixException;

    /**
     * Creates copy of node.
     *
     * @param copyGradients if true copies also gradient information.
     * @throws MatrixException throws exception is matrix is not defined.
     * @return copy of node.
     */
    Node copy(boolean copyGradients) throws MatrixException;

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
     * Returns first key of node.
     *
     * @return first key of node.
     */
    int firstKey();

    /**
     * Returns last key of node.
     *
     * @return last key of node.
     */
    int lastKey();

    /**
     * Checks if node contains specific matrix.
     *
     * @param matrix specific matrix.
     * @return returns true if node contains specific matrix.
     */
    boolean contains(Matrix matrix);

    /**
     * Returns empty matrix with dimensions of reference matrix.
     *
     * @return empty matrix with dimensions of reference matrix.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    Matrix getEmptyMatrix() throws MatrixException;

    /**
     * Resets node and removes other data than constant data.
     *
     * @param resetDependentNodes if true resets also dependent nodes.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    void resetNode(boolean resetDependentNodes) throws MatrixException;

    /**
     * Sets matrices for node.
     *
     * @param matrices matrices of node.
     */
    void setMatrices(MMatrix matrices);

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
     * Returns new matrix of node.
     *
     * @param index data index for matrix.
     * @return matrix of node.
     * @throws MatrixException throws exception if scalar type of node and matrix are not matching.
     */
    Matrix getNewMatrix(int index) throws MatrixException;

    /**
     * Returns matrices of node.
     *
     * @return matrices of node.
     */
    MMatrix getMatrices();

    /**
     * Sets gradient matrix of node.
     *
     * @param gradient gradient matrix of node.
     */
    void setGradient(Matrix gradient);

    /**
     * Sets gradient matrix of node.
     *
     * @param index data index for gradient.
     * @param gradient gradient matrix of node.
     * @throws MatrixException throws exception if putting of matrix fails.
     */
    void setGradient(int index, Matrix gradient) throws MatrixException;

    /**
     * Sets gradients for node.
     *
     * @param gradients gradients of node.
     */
    void setGradients(MMatrix gradients);

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
     * Returns gradients of node.
     *
     * @return gradients of node.
     */
    MMatrix getGradients();

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
