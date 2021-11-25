/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package utils.procedure.node;

import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;

/**
 * Class that implements abstract node for expression calculation.<br>
 * Stores both matrices and gradients for multiple data indices.<br>
 *
 */
public abstract class AbstractNode implements Node, Serializable {

    @Serial
    private static final long serialVersionUID = -1121024205323275937L;

    /**
     * ID for node.
     *
     */
    private final int id;

    /**
     * Reference matrix for node.
     *
     */
    private final Matrix referenceMatrix;

    /**
     * Sets dependency node backward.
     *
     */
    private Node fromNode;

    /**
     * Sets dependency node forward.
     *
     */
    private Node toNode;

    /**
     * Number of cumulated gradient entries.
     *
     */
    private transient int cumulatedGradientEntryCount = 0;

    /**
     * Is true gradient is not updated for this node.
     *
     */
    private boolean stopGradient = false;

    /**
     * Constructor for node.
     *
     * @param id id.
     * @param referenceMatrix reference matrix.
     * @throws MatrixException throws exception is matrix is not defined.
     */
    public AbstractNode(int id, Matrix referenceMatrix) throws MatrixException {
        if (referenceMatrix == null) throw new MatrixException("Reference matrix is not defined for the node.");
        this.id = id;
        this.referenceMatrix = referenceMatrix;
    }

    /**
     * Constructor for node.
     *
     * @param id id.
     * @param referenceMatrix reference matrix.
     * @throws MatrixException throws exception is matrix is not defined.
     */
    public AbstractNode(int id, MMatrix referenceMatrix) throws MatrixException {
        this(id, referenceMatrix.get(referenceMatrix.firstKey()));
    }

    /**
     * Returns reference matrix.
     *
     * @return reference matrix.
     */
    public Matrix getReferenceMatrix() {
        return referenceMatrix;
    }

    /**
     * Returns true if given matrix is reference matrix of this node.
     *
     * @param matrix given matrix
     * @return true if given matrix is reference matrix of this node.
     */
    public boolean isReferenceOf(Matrix matrix) {
        return referenceMatrix == matrix;
    }

    /**
     * Returns true is reference matrix is of scalar type otherwise false.
     *
     * @return true is reference matrix is of scalar type otherwise false.
     */
    public boolean isScalar() {
        return referenceMatrix.isScalar();
    }

    /**
     * Returns number of rows in reference matrix.
     *
     * @return number of rows in reference matrix.
     */
    public int getRows() {
        return referenceMatrix.getRows();
    }

    /**
     * Returns number of columns in reference matrix.
     *
     * @return number of columns in reference matrix.
     */
    public int getColumns() {
        return referenceMatrix.getColumns();
    }

    /**
     * Set dependency node backward.
     *
     * @param fromNode from node.
     */
    public void setFromNode(Node fromNode) {
        this.fromNode = fromNode;
    }

    /**
     * Returns from node.
     *
     * @return from node.
     */
    public Node getFromNode() {
        return fromNode;
    }

    /**
     * Set dependency node forward.
     *
     * @param toNode to node.
     */
    public void setToNode(Node toNode) {
        this.toNode = toNode;
    }

    /**
     * Returns to node.
     *
     * @return to node.
     */
    public Node getToNode() {
        return toNode;
    }

    /**
     * Updates matrix dependency to forward direction.
     *
     * @param index index
     * @throws MatrixException throws exception if scalar type of node and matrix are not matching or node is of type multi-index.
     */
    public void updateMatrixDependency(int index) throws MatrixException {
        if (fromNode != null) setMatrix(index, fromNode.getMatrix(index - 1) != null ? fromNode.getMatrix(index - 1) : getEmptyMatrix());
    }

    /**
     * Updates gradient dependency to backward direction.
     *
     * @param index index
     * @throws MatrixException throws exception if scalar type of node and matrix are not matching or node is of type multi-index.
     */
    public void updateGradientDependency(int index) throws MatrixException {
        if (toNode != null) setGradient(index, toNode.getGradient(index + 1) != null ? toNode.getGradient(index + 1) : getEmptyMatrix());
    }

    /**
     * Returns id of node.
     *
     * @return id of node.
     */
    public int getId() {
        return id;
    }

    /**
     * Return name of node
     *
     * @return name of node
     */
    public String getName() {
        return referenceMatrix.getName() != null ? referenceMatrix.getName() : "Node" + getId();
    }

    /**
     * Sets if gradient is updated for this node. If true gradient is not updated otherwise it is updated.
     *
     * @param stopGradient if true gradient is not updated otherwise it is updated.
     */
    public void setStopGradient(boolean stopGradient) {
        this.stopGradient = stopGradient;
    }

    /**
     * Returns if gradient is updated for this node. If true gradient is not updated otherwise it is updated.
     *
     * @return if true gradient is not updated otherwise it is updated.
     */
    public boolean isStopGradient() {
        return stopGradient;
    }

    /**
     * Returns empty matrix with dimensions of reference matrix.
     *
     * @return empty matrix with dimensions of reference matrix.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public Matrix getEmptyMatrix() throws MatrixException {
        return referenceMatrix.getNewMatrix();
    }

    /**
     * Resets node and removes other data than constant data.
     *
     * @param resetDependentNodes if true resets also dependent nodes.
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public void resetNode(boolean resetDependentNodes) throws MatrixException {
        cumulatedGradientEntryCount = 0;
    }

    /**
     * Returns new matrix of node.
     *
     * @param index data index for matrix.
     * @return matrix of node.
     * @throws MatrixException throws exception if scalar type of node and matrix are not matching.
     */
    public Matrix getNewMatrix(int index) throws MatrixException {
        Matrix newMatrix = getEmptyMatrix();
        setMatrix(index, newMatrix);
        return newMatrix;
    }

    /**
     * Sets matrix of this node.
     *
     * @param matrix new matrix.
     * @throws MatrixException throws exception if scalar type of node and matrix are not matching or node is of type multi-index.
     */
    public void setMatrix(Matrix matrix) throws MatrixException {
        if (matrix.isScalar() != referenceMatrix.isScalar()) throw new MatrixException("Scalar type of node and matrix is not matching.");
    }

    /**
     * Sets matrix of this node.
     *
     * @param index data index for matrix.
     * @param matrix new matrix.
     * @throws MatrixException throws exception if scalar type of node and matrix are not matching.
     */
    public void setMatrix(int index, Matrix matrix) throws MatrixException {
        if (matrix.isScalar() != referenceMatrix.isScalar()) throw new MatrixException("Scalar type of node and matrix is not matching.");
    }

    /**
     * Returns gradient mean (average).
     *
     * @return gradient mean (average).
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix getGradientMean() throws MatrixException {
        return cumulatedGradientEntryCount == 0 ? getEmptyMatrix() : getGradient().divide(cumulatedGradientEntryCount);
    }

    /**
     * Cumulates gradient.
     *
     * @param index data index.
     * @param outputGradient output gradient.
     * @param negateGradient if true output gradient contribution is negated prior being cumulated.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void cumulateGradient(int index, Matrix outputGradient, boolean negateGradient) throws MatrixException {
        if (getGradient(index) == null) setGradient(index, getEmptyMatrix());

        if (!negateGradient) getGradient(index).add(outputGradient, getGradient(index));
        else getGradient(index).subtract(outputGradient, getGradient(index));

        cumulatedGradientEntryCount++;
    }

}
