/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.procedure.node;

import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 * Defines node with multiple matrices inside.
 *
 */
public class MultiNode extends AbstractNode {

    /**
     * Matrices for node.
     *
     */
    private transient TreeMap<Integer, Matrix> matrices;

    /**
     * Gradients for node.
     *
     */
    private transient TreeMap<Integer, Matrix> gradients;

    /**
     * Matrix backup for forward dependencies.
     *
     */
    private transient HashMap<Integer, TreeMap<Integer, Matrix>> matrixBackup = new HashMap<>();

    /**
     * Constructor for multi node.
     *
     * @param id id.
     * @param referenceMatrix reference matrix.
     * @throws MatrixException throws exception is matrix is not defined.
     */
    public MultiNode(int id, Matrix referenceMatrix) throws MatrixException {
        super(id, referenceMatrix);
        matrices = new TreeMap<>();
        matrices.put(0, referenceMatrix);
        gradients = new TreeMap<>();
    }

    /**
     * Constructor for multi node.
     *
     * @param id id.
     * @param referenceMatrix reference matrix.
     * @throws MatrixException throws exception is matrix is not defined.
     */
    public MultiNode(int id, MMatrix referenceMatrix) throws MatrixException {
        this(id, referenceMatrix.getReferenceMatrix());
        int depth = referenceMatrix.getDepth();
        for (int depthIndex = 0; depthIndex < depth; depthIndex++) {
            matrices.put(depthIndex, referenceMatrix.get(depthIndex));
        }
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
     */
    public void storeMatrixDependency(int backupIndex) {
        if (getToNode() == null) return;
        TreeMap<Integer, Matrix> matricesBackup = new TreeMap<>();
        for (Map.Entry<Integer, Matrix> entry : entrySet()) {
            int index = entry.getKey();
            Matrix matrix = entry.getValue();
            matricesBackup.put(index, matrix);
        }
        matrixBackup.put(backupIndex, matricesBackup);
    }

    /**
     * Restores matrix dependency.
     *
     * @param backupIndex backup index.
     */
    public void restoreMatrixDependency(int backupIndex) {
        if (getToNode() == null || matrixBackup == null) return;
        if (matrixBackup.containsKey(backupIndex)) {
            TreeMap<Integer, Matrix> matricesBackup = matrixBackup.get(backupIndex);
            for (Map.Entry<Integer, Matrix> entry : matricesBackup.entrySet()) {
                int index = entry.getKey();
                Matrix matrix = entry.getValue();
                getMatrices().put(index, matrix);
            }
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
     * Returns key set of node.
     *
     * @return key set of node.
     */
    public Set<Map.Entry<Integer, Matrix>> entrySet() {
        return matrices.entrySet();
    }

    /**
     * Checks if node contains specific matrix.
     *
     * @param matrix specific matrix.
     * @return returns true if node contains specific matrix.
     */
    public boolean contains(Matrix matrix) {
        return matrices.containsValue(matrix);
    }

    /**
     * Resets node and removes other data than constant data.
     *
     * @param resetDependentNodes if true resets also dependent nodes.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public void resetNode(boolean resetDependentNodes) throws MatrixException {
        if (getToNode() == null || resetDependentNodes) matrices = new TreeMap<>();
        gradients = new TreeMap<>();
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
    public TreeMap<Integer, Matrix> getMatrices() {
        return matrices;
    }

    /**
     * Sets gradient matrix of node.
     *
     * @param index data index for gradient.
     * @param gradient gradient matrix of node.
     */
    public void setGradient(int index, Matrix gradient) {
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
