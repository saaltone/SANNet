/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 *
 ********************************************************/

package utils.procedure;

import core.normalization.Normalization;
import utils.matrix.DMatrix;
import utils.matrix.Init;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serializable;
import java.util.HashSet;
import java.util.Set;
import java.util.TreeMap;

/**
 * Class that implements node for expression calculation. Node contains value(s) of arguments for expression.<br>
 * Stores both matrices and gradients for multiple data indices.<br>
 * Supports constant node where data is shared between data indiced.<br>
 *
 */
public class Node implements Serializable {

    private static final long serialVersionUID = -1121024205323275937L;

    /**
     * Matrices for node.
     *
     */
    private final TreeMap<Integer, Matrix> matrices = new TreeMap<>();

    /**
     * Gradients for node.
     *
     */
    private final TreeMap<Integer, Matrix> gradients = new TreeMap<>();

    /**
     * If true node is treated as constant matrix.
     *
     */
    private boolean constantNode;

    /**
     * If true creates matrix is not existing when get.
     *
     */
    private boolean createMatrixIfNone;

    /**
     * If true matrix is of constant type.
     *
     */
    private boolean constantMatrix;

    /**
     * Procedure callback for node.
     *
     */
    private HashSet<Normalization> normalizers;

    /**
     * Number of rows in matrix.
     *
     */
    private int rows;

    /**
     * Number of columns in matrix.
     *
     */
    private int cols;

    /**
     * Constructor for node. Records dimensions of references matrix as node data dimensions.
     *
     * @param matrix reference matrix.
     * @param constantNode if true node is treated as constant node.
     * @throws MatrixException throws exception is matrix is not defined.
     */
    Node(Matrix matrix, boolean constantNode, boolean createMatrixIfNone, boolean constantMatrix, HashSet<Normalization> normalizers) throws MatrixException {
        if (matrix != null) {
            rows = matrix.getRows();
            cols = matrix.getCols();
        } else throw new MatrixException("Matrix is not defined for the node.");
        this.constantNode = constantNode;
        this.createMatrixIfNone = createMatrixIfNone;
        this.constantMatrix = constantMatrix;
        this.normalizers = normalizers;
    }

    /**
     * Make forward callback to constant (node) entry.
     *
     * @throws MatrixException throws exception is matrix operation fails.
     */
    public void forwardCallbackConstant() throws MatrixException {
        if (normalizers != null && constantNode) {
            for (Normalization normalizer : normalizers) setMatrix(0, normalizer.forward(getMatrix(0)));
        }
    }

    /**
     * Make forward callback to all entries of node.
     *
     * @throws MatrixException throws exception is matrix operation fails.
     */
    public void forwardCallback() throws MatrixException {
        if (normalizers != null) {
            for (Normalization normalizer : normalizers) normalizer.forward(this);
        }
    }

    /**
     * Make forward callback to specific entry (sample)
     *
     * @throws MatrixException throws exception is matrix operation fails.
     * @param sampleIndex sample index of specific entry.
     */
    public void forwardCallback(int sampleIndex) throws MatrixException {
        if (normalizers != null) {
            for (Normalization normalizer : normalizers) normalizer.forward(this, sampleIndex);
        }
    }

    /**
     * Make backward callback to constant (node) entry.
     *
     * @throws MatrixException throws exception is matrix operation fails.
     */
    public void backwardCallbackConstant() throws MatrixException {
        if (normalizers != null && constantNode) {
            for (Normalization normalizer : normalizers) setGradient(0, normalizer.backward(getMatrix(0), getGradient(0)));
        }
    }

    /**
     * Make backward callback to all entries of node.
     *
     * @throws MatrixException throws exception is matrix operation fails.
     */
    public void backwardCallback() throws MatrixException {
        if (normalizers != null) {
            for (Normalization normalizer : normalizers) normalizer.backward(this);
        }
    }

    /**
     * Make backward callback to specific entry (sample)
     *
     * @throws MatrixException throws exception is matrix operation fails.
     * @param sampleIndex sample index of specific entry.
     */
    public void backwardCallback(int sampleIndex) throws MatrixException {
        if (normalizers != null) {
            for (Normalization normalizer : normalizers) normalizer.backward(this, sampleIndex);
        }
    }

    /**
     * Returns if node is constant node type.
     *
     * @return true if node is constant node type otherwise false.
     */
    public boolean isConstantNode() {
        return !constantNode;
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
     * Returns empty matrix with size of reference matrix.
     *
     * @return empty matrix with size of reference matrix.
     * @throws MatrixException throws exception is matrix operation fails.
     */
    public Matrix getEmptyMatrix() throws MatrixException {
        return !constantMatrix ? new DMatrix(rows, cols) : new DMatrix(rows, cols, Init.CONSTANT);
    }

    /**
     * Resets node and removes other data than constant data.
     *
     */
    public void resetNode() {
        if (!constantNode) matrices.clear();
        resetGradient();
    }

    /**
     * Resets node for specific data index. Leaves data of constant node intact.
     *
     * @param index specific data index.
     */
    public void resetNode(int index) {
        if (!constantNode) matrices.remove(index);
        resetGradient(index);
    }

    /**
     * Resets gradient matrix and removes all gradient data.
     *
     */
    private void resetGradient() {
        gradients.clear();
    }

    /**
     * Resets and removes gradient for specific data index.
     *
     * @param index specific data index.
     */
    private void resetGradient(int index) {
        gradients.remove(index);
    }

    /**
     * Sets matrix of this node.
     *
     * @param index data index for matrix.
     * @param matrix new matrix.
     */
    public void setMatrix(int index, Matrix matrix) {
        if (!constantNode) matrices.put(index, matrix);
        else matrices.put(0, matrix);
    }

    /**
     * Returns matrix of node.
     *
     * @param index data index for matrix.
     * @return matrix of node.
     * @throws MatrixException throws exception is matrix operation fails.
     */
    public Matrix getMatrix(int index) throws MatrixException {
        if (!constantNode) {
            if (!createMatrixIfNone) return matrices.get(index);
            else {
                if (matrices.containsKey(index)) return matrices.get(index);
                else {
                    Matrix matrix = getEmptyMatrix();
                    matrices.put(index, matrix);
                    return matrix;
                }
            }
        }
        else return matrices.get(0);
    }

    /**
     * Sets gradient matrix of node.
     *
     * @param index data index for gradient.
     * @param gradient gradient matrix of node.
     */
    public void setGradient(int index, Matrix gradient) {
        if (!constantNode) gradients.put(index, gradient);
        else gradients.put(0, gradient);
    }

    /**
     * Returns gradient matrix of node.
     *
     * @param index data index for gradient.
     * @return gradient matrix of node.
     */
    public Matrix getGradient(int index) {
        if (!constantNode) return gradients.get(index);
        else return gradients.get(0);
    }

    /**
     * Removes procedure factory from node and it's matrices.
     *
     * @param index data index.
     * @throws MatrixException throws exception is matrix operation fails.
     */
    public void removeProcedureFactory(int index) throws MatrixException {
        if (getMatrix(index) != null) getMatrix(index).removeProcedureFactory();
        if (getGradient(index) != null) getGradient(index).removeProcedureFactory();
    }

    /**
     * Updates gradient.
     *
     * @param index data index.
     * @param outputGrad output gradient.
     * @param add if true output gradient contribution is added to node gradient otherwise subtracted.
     * @throws MatrixException throws exception is matrix operation fails.
     */
    public void updateGradient(int index, Matrix outputGrad, boolean add) throws MatrixException {
        if (getGradient(index) == null) setGradient(index, getEmptyMatrix());
        if (!getGradient(index).isConstant()) {
            if (add) getGradient(index).add(outputGrad, getGradient(index));
            else getGradient(index).subtract(outputGrad, getGradient(index));
        }
        else {
            if (add) getGradient(index).add(outputGrad.sum(), getGradient(index));
            else getGradient(index).subtract(outputGrad.sum(), getGradient(index));
        }
    }

}
