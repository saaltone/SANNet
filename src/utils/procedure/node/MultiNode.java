/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.procedure.node;

import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashMap;
import java.util.Set;

/**
 * Defines node with multiple matrices inside.
 *
 */
public class MultiNode extends AbstractNode {

    /**
     * Matrices for node.
     *
     */
    private transient MMatrix matrices;

    /**
     * Gradients for node.
     *
     */
    private transient MMatrix gradients;

    /**
     * Matrix backup for forward dependencies.
     *
     */
    private transient HashMap<Integer, MMatrix> matrixBackup = new HashMap<>();

    /**
     * Constructor for node.
     *
     * @param id id.
     * @param referenceMatrix reference matrix.
     * @throws MatrixException throws exception is matrix is not defined.
     */
    public MultiNode(int id, Matrix referenceMatrix) throws MatrixException {
        super(id, referenceMatrix);
        matrices = new MMatrix();
        matrices.put(0, referenceMatrix);
        gradients = new MMatrix();
    }

    /**
     * Constructor for node.
     *
     * @param id id.
     * @param referenceMatrix reference matrix.
     * @throws MatrixException throws exception is matrix is not defined.
     */
    public MultiNode(int id, MMatrix referenceMatrix) throws MatrixException {
        this(id, referenceMatrix.getReferenceMatrix());
        for (Integer index : referenceMatrix.keySet()) matrices.put(index, referenceMatrix.get(index));
    }

    /**
     * If true node is of type multi index.
     *
     * @return true node is of type multi index.
     */
    public boolean isMultiIndex() {
        return true;
    }

    /**
     * Stores matrix dependency
     *
     * @param backupIndex backup index
     * @throws MatrixException throws exception if storing dependency fails.
     */
    public void storeMatrixDependency(int backupIndex) throws MatrixException {
        if (getToNode() == null) return;
        MMatrix matricesBackup = new MMatrix();
        for (Integer index : keySet()) matricesBackup.put(index, getMatrix(index));
        matrixBackup.put(backupIndex, matricesBackup);
    }

    /**
     * Restores matrix dependency.
     *
     * @param backupIndex backup index.
     * @throws MatrixException throws exception if restoring of backup fails.
     */
    public void restoreMatrixDependency(int backupIndex) throws MatrixException {
        if (getToNode() == null || matrixBackup == null) return;
        if (matrixBackup.containsKey(backupIndex)) {
            MMatrix matricesBackup = matrixBackup.get(backupIndex);
            for (Integer index : matricesBackup.keySet()) getMatrices().put(index, matricesBackup.get(index));
        }
    }

    /**
     * Returns size of node.
     *
     * @return size of node.
     */
    public int size() {
        return matrices.size();
    }

    /**
     * Returns key set of node.
     *
     * @return key set of node.
     */
    public Set<Integer> keySet() {
        return matrices.keySet();
    }

    /**
     * Checks if node contains specific matrix.
     *
     * @param matrix specific matrix.
     * @return returns true if node contains specific matrix.
     */
    public boolean contains(Matrix matrix) {
        return matrices.contains(matrix);
    }

    /**
     * Resets node and removes other data than constant data.
     *
     * @param resetDependentNodes if true resets also dependent nodes.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public void resetNode(boolean resetDependentNodes) throws MatrixException {
        if (getToNode() == null || resetDependentNodes) matrices = new MMatrix();
        gradients = new MMatrix();
        matrixBackup = new HashMap<>();
        super.resetNode(resetDependentNodes);
    }

    /**
     * Sets matrix of this node.
     *
     * @param matrix new matrix.
     */
    public void setMatrix(Matrix matrix) {
    }

    /**
     * Sets matrix of this node.
     *
     * @param index data index for matrix.
     * @param matrix new matrix.
     * @throws MatrixException throws exception if scalar type of node and matrix are not matching.
     */
    public void setMatrix(int index, Matrix matrix) throws MatrixException {
        super.setMatrix(index, matrix);
        matrices.put(index, matrix);
    }

    /**
     * Returns matrix of node.
     *
     * @return matrix of node.
     */
    public Matrix getMatrix() {
        return null;
    }

    /**
     * Returns matrix of node.
     *
     * @param index data index for matrix.
     * @return matrix of node.
     */
    public Matrix getMatrix(int index) {
        return matrices.get(index);
    }

    /**
     * Returns matrices of node.
     *
     * @return matrices of node.
     */
    public MMatrix getMatrices() {
        return matrices;
    }

    /**
     * Sets gradient matrix of node.
     *
     * @param index data index for gradient.
     * @param gradient gradient matrix of node.
     * @throws MatrixException throws exception if putting of matrix fails.
     */
    public void setGradient(int index, Matrix gradient) throws MatrixException {
        gradients.put(index, gradient);
    }

    /**
     * Returns gradient matrix of node.
     *
     * @return gradient matrix of node.
     */
    public Matrix getGradient() {
        return null;
    }

    /**
     * Returns gradient matrix of node.
     *
     * @param index data index of gradient.
     * @return gradient matrix of node.
     */
    public Matrix getGradient(int index) {
        return gradients.get(index);
    }

}
