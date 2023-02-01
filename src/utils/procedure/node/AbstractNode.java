/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.procedure.node;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;

/**
 * Implements abstract node for expression calculation.<br>
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
     * Set of dependent backward node.
     *
     */
    private Node fromResultNode = null;

    /**
     * Set of dependent forward node.
     *
     */
    private Node toArgumentNode = null;

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
     * If true matrix dependencies will be reset otherwise not.
     *
     */
    private boolean resetDependencies = true;

    /**
     * Previous reset dependencies state.
     *
     */
    private boolean previousResetDependencies = true;

    /**
     * Latest matrix.
     *
     */
    private transient Matrix latestMatrix;

    /**
     * Constructor for abstract node.
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
     * Sets backward dependent node.
     *
     * @param fromResultNode from result node.
     */
    public void setFromResultNode(Node fromResultNode) {
        this.fromResultNode = fromResultNode;
    }

    /**
     * Sets forward dependent node.
     *
     * @param toArgumentNode to argument node.
     */
    public void setToArgumentNode(Node toArgumentNode) {
        this.toArgumentNode = toArgumentNode;
    }

    /**
     * Checks if node has from result node.
     *
     * @return if node has from result node returns true otherwise returns false.
     */
    protected boolean hasFromResultNode() {
        return fromResultNode != null;
    }

    /**
     * Checks if node has to argument node.
     *
     * @return if node has to argument node returns true otherwise returns false.
     */
    protected boolean hasToArgumentNode() {
        return toArgumentNode != null;
    }

    /**
     * Sets reset flag for matrix dependencies.
     *
     * @param resetDependencies if true matrix dependencies are reset otherwise false.
     */
    public void resetDependencies(boolean resetDependencies) {
        this.resetDependencies = resetDependencies;
        if (resetDependencies != previousResetDependencies) latestMatrix = null;
        previousResetDependencies = resetDependencies;
    }

    /**
     * Updates dependencies.
     *
     * @param index index.
     */
    public void updateDependencies(int index) {
        if (hasFromResultNode() && !resetDependencies) latestMatrix = fromResultNode.getMatrix(index);
    }

    /**
     * Updates matrix dependency to forward direction.
     *
     * @param index index
     * @param previousIndex previous index
     * @throws MatrixException throws exception if scalar type of node and matrix are not matching or node is of type multi-index.
     */
    public void updateMatrixDependency(int index, int previousIndex) throws MatrixException {
        if (hasFromResultNode()) {
            if (fromResultNode.getMatrix(previousIndex) != null) setMatrix(index, fromResultNode.getMatrix(previousIndex));
            else setMatrix(index, latestMatrix != null ? latestMatrix : getNewMatrix());
        }
    }

    /**
     * Updates gradient dependency to backward direction.
     *
     * @param index index
     * @param previousIndex previous index
     */
    public void updateGradientDependency(int index, int previousIndex) {
        if (hasToArgumentNode()) {
            if (toArgumentNode.getGradient(previousIndex) != null) setGradient(index, toArgumentNode.getGradient(previousIndex));
        }
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
    public Matrix getNewMatrix() throws MatrixException {
        return referenceMatrix.getNewMatrix();
    }

    /**
     * Resets node and removes other data than constant data.
     *
     * @throws MatrixException throws exception is dimensions of matrices are not matching or any matrix is scalar type.
     */
    public void reset() throws MatrixException {
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
        Matrix newMatrix = getNewMatrix();
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
        if (matrix.getRows() != referenceMatrix.getRows() || matrix.getColumns() != referenceMatrix.getColumns()) throw new MatrixException("Matrix dimensions (" + matrix.getRows() + "x" + matrix.getColumns() + ") are not matching with reference matrix dimensions (" + referenceMatrix.getRows() + "x" + referenceMatrix.getColumns() + ")");
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
        if (matrix.getRows() != referenceMatrix.getRows() || matrix.getColumns() != referenceMatrix.getColumns()) throw new MatrixException("Matrix dimensions (" + matrix.getRows() + "x" + matrix.getColumns() + ") are not matching with reference matrix dimensions (" + referenceMatrix.getRows() + "x" + referenceMatrix.getColumns() + ")");
    }

    /**
     * Returns gradient mean (average).
     *
     * @return gradient mean (average).
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix getGradientMean() throws MatrixException {
        return cumulatedGradientEntryCount == 0 ? getNewMatrix() : getGradient().divide(cumulatedGradientEntryCount);
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
        if (getGradient(index) == null) setGradient(index, getNewMatrix());

        if (!negateGradient) getGradient(index).incrementBy(outputGradient);
        else getGradient(index).decrementBy(outputGradient);

        cumulatedGradientEntryCount++;
    }

}
