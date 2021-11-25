/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
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
        this(id, referenceMatrix.get(referenceMatrix.firstKey()));
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
     * Creates copy of node.
     *
     * @param copyGradients if true copies also gradient information.
     * @throws MatrixException throws exception is matrix is not defined.
     * @return copy of node.
     */
    public Node copy(boolean copyGradients) throws MatrixException {
        Node node = new MultiNode(getId(), getReferenceMatrix());
        for (Integer index : keySet()) {
            node.setMatrix(index, getMatrix(index));
            if (copyGradients) node.setGradient(index, getGradient(index));
        }
        return node;
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
     * Returns first key of node.
     *
     * @return first key of node.
     */
    public int firstKey() {
        return matrices.firstKey();
    }

    /**
     * Returns last key of node.
     *
     * @return last key of node.
     */
    public int lastKey() {
        return matrices.lastKey();
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
     * Sets matrices for node.
     *
     * @param matrices matrices of node.
     */
    public void setMatrices(MMatrix matrices) {
        this.matrices = matrices;
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
     * @param gradient gradient matrix of node.
     */
    public void setGradient(Matrix gradient) {
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
     * Sets gradients for node.
     *
     * @param gradients gradients of node.
     */
    public void setGradients(MMatrix gradients) {
        this.gradients = gradients;
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

    /**
     * Returns gradients of node.
     *
     * @return gradients of node.
     */
    public MMatrix getGradients() {
        return gradients;
    }

}
