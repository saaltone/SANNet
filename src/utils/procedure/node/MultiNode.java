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
     * If true node is of type multi index.
     *
     * @return true node is of type multi index.
     */
    public boolean isMultiIndex() {
        return true;
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
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public void reset() throws MatrixException {
        if (getToNode() == null) matrices = new TreeMap<>();
        gradients = new TreeMap<>();
        super.reset();
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
