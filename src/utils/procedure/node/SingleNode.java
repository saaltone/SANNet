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
     * Constructor for node.
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
     * Constructor for node.
     *
     * @param id id.
     * @param referenceMatrix reference matrix.
     * @throws MatrixException throws exception is matrix is not defined.
     */
    public SingleNode(int id, MMatrix referenceMatrix) throws MatrixException {
        this(id, referenceMatrix.get(referenceMatrix.firstKey()));
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
     * Creates copy of node.
     *
     * @param copyGradients if true copies also gradient information.
     * @throws MatrixException throws exception is matrix is not defined.
     * @return copy of node.
     */
    public Node copy(boolean copyGradients) throws MatrixException {
        Node node = new SingleNode(getId(), getReferenceMatrix());
        if (copyGradients) node.setGradient(getGradient());
        return node;
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
     * Returns first key of node.
     *
     * @return first key of node.
     */
    public int firstKey() {
        return 0;
    }

    /**
     * Returns last key of node.
     *
     * @return last key of node.
     */
    public int lastKey() {
        return 0;
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
    public void resetNode(boolean resetDependentNodes) throws MatrixException {
        gradient = null;
        super.resetNode(resetDependentNodes);
    }

    /**
     * Sets matrices for node.
     *
     * @param matrices matrices of node.
     */
    public void setMatrices(MMatrix matrices) {
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
    public MMatrix getMatrices() {
        return null;
    }

    /**
     * Sets gradient matrix of node.
     *
     * @param gradient gradient matrix of node.
     */
    public void setGradient(Matrix gradient) {
        this.gradient = gradient;
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
     * Sets gradients for node.
     *
     * @param gradients gradients of node.
     */
    public void setGradients(MMatrix gradients) {
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

    /**
     * Returns gradients of node.
     *
     * @return gradients of node.
     */
    public MMatrix getGradients() {
        return null;
    }

}
