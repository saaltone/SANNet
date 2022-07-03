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
 * Defines node with single matrix inside.
 *
 */
public class SingleNode extends AbstractNode {

    /**
     * Constant matrix if node is treated as constant node.
     *
     */
    private Matrix matrix;

    /**
     * Constant gradient node is treated as constant node.
     *
     */
    private transient Matrix gradient;

    /**
     * Constructor for single node.
     *
     * @param id id.
     * @param referenceMatrix reference matrix.
     * @throws MatrixException throws exception is matrix is not defined.
     */
    public SingleNode(int id, Matrix referenceMatrix) throws MatrixException {
        super(id, referenceMatrix);
        matrix = referenceMatrix;
    }

    /**
     * If true node is of type multi index.
     *
     * @return true node is of type multi index.
     */
    public boolean isMultiIndex() {
        return false;
    }

    /**
     * Stores matrix dependency
     *
     * @param backupIndex backup index
     */
    public void storeMatrixDependency(int backupIndex) {
    }

    /**
     * Restores matrix dependency.
     *
     * @param backupIndex backup index.
     */
    public void restoreMatrixDependency(int backupIndex) {
    }

    /**
     * Returns size of node.
     *
     * @return size of node.
     */
    public int size() {
        return 1;
    }

    /**
     * Returns key set of node.
     *
     * @return key set of node.
     */
    public Set<Integer> keySet() {
        return null;
    }

    /**
     * Returns entry set of node.
     *
     * @return entry set of node.
     */
    public Set<Map.Entry<Integer, Matrix>> entrySet() {
        return null;
    }

    /**
     * Checks if node contains specific matrix.
     *
     * @param matrix specific matrix.
     * @return returns true if node contains specific matrix.
     */
    public boolean contains(Matrix matrix) {
        return matrix == this.matrix;
    }

    /**
     * Resets node and removes other data than constant data.
     *
     * @param resetDependentNodes if true resets also dependent nodes.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public void reset(boolean resetDependentNodes) throws MatrixException {
        gradient = null;
        super.reset(resetDependentNodes);
    }

    /**
     * Sets matrix of this node.
     *
     * @param matrix new matrix.
     * @throws MatrixException throws exception if scalar type of node and matrix are not matching or node is of type multi-index.
     */
    public void setMatrix(Matrix matrix) throws MatrixException {
        super.setMatrix(matrix);
        this.matrix = matrix;
    }

    /**
     * Sets matrix of this node.
     *
     * @param index data index for matrix.
     * @param matrix new matrix.
     */
    public void setMatrix(int index, Matrix matrix) {
        this.matrix = matrix;
    }

    /**
     * Returns matrix of node.
     *
     * @return matrix of node.
     */
    public Matrix getMatrix() {
        return matrix;
    }

    /**
     * Returns matrix of node.
     *
     * @param index data index for matrix.
     * @return matrix of node.
     */
    public Matrix getMatrix(int index) {
        return matrix;
    }

    /**
     * Returns matrices of node.
     *
     * @return matrices of node.
     */
    public TreeMap<Integer, Matrix> getMatrices() {
        return null;
    }

    /**
     * Sets gradient matrix of node.
     *
     * @param index data index for gradient.
     * @param gradient gradient matrix of node.
     */
    public void setGradient(int index, Matrix gradient) {
        this.gradient = gradient;
    }

    /**
     * Returns gradient matrix of node.
     *
     * @return gradient matrix of node.
     */
    public Matrix getGradient() {
        return gradient;
    }

    /**
     * Returns gradient matrix of node.
     *
     * @param index data index of gradient.
     * @return gradient matrix of node.
     */
    public Matrix getGradient(int index) {
        return gradient;
    }

}
