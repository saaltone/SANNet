/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package utils;

import java.io.Serializable;
import java.util.TreeMap;

/**
 * Class that implements node for expression calculation.<br>
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
    Node(Matrix matrix, boolean constantNode) throws MatrixException {
        if (matrix != null) {
            rows = matrix.getRows();
            cols = matrix.getCols();
        } else throw new MatrixException("Matrix is not defined for the node.");
        this.constantNode = constantNode;
    }

    /**
     * Gets if node is constant node type.
     *
     * @return true if node is constant node type otherwise false.
     */
    public boolean isConstantNode() {
        return constantNode;
    }

    /**
     * Returns empty matrix with size of reference matrix.
     *
     * @return empty matrix with size of reference matrix.
     */
    public Matrix getEmptyMatrix() {
        return new DMatrix(rows, cols);
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
     * Gets matrix of node.
     *
     * @param index data index for matrix.
     * @return matrix of node.
     */
    public Matrix getMatrix(int index) {
        if (!constantNode) return matrices.get(index);
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
     * Gets gradient matrix of node.
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
     */
    public void removeProcedureFactory(int index) {
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
        if (add) getGradient(index).add(outputGrad, getGradient(index));
        else getGradient(index).subtract(outputGrad, getGradient(index));
    }

}
