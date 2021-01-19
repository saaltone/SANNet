/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.procedure;

import core.normalization.Normalization;
import core.regularization.Regularization;
import utils.DynamicParamException;
import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

/**
 * Class that implements node for expression calculation. Node contains value(s) of arguments for expression.<br>
 * Stores both matrices and gradients for multiple data indices.<br>
 * Supports constant node where data is shared between data indices.<br>
 *
 */
public class Node implements Serializable {

    private static final long serialVersionUID = -1121024205323275937L;

    /**
     * ID for node.
     *
     */
    private final int id;

    /**
     * Matrices for node.
     *
     */
    private transient MMatrix matrices;

    /**
     * Constant matrix if node is of type contains type
     *
     */
    private Matrix constantMatrix;

    /**
     * True if matrix is of type multi index.
     *
     */
    private boolean isMultiIndex;

    /**
     * Gradients for node.
     *
     */
    private transient MMatrix gradients;

    /**
     * Constant gradient if node is of type contains type
     *
     */
    private transient Matrix constantGradient;

    /**
     * If true node is treated as constant node.
     *
     */
    private final boolean isConstantNode;

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
     * Matrix backup for forward dependencies.
     *
     */
    private HashMap<Integer, MMatrix> matrixBackup = new HashMap<>();

    /**
     * Normalizers for node.
     *
     */
    private HashSet<Normalization> normalizers;

    /**
     * Regularizers for node.
     *
     */
    private HashSet<Regularization> regularizers;

    /**
     * Number of matrix / gradient entries.
     *
     */
    private int entryCount = 0;

    /**
     * Constructor for node. Records dimensions of references matrix as node data dimensions.
     *
     * @param referenceMatrix reference matrix.
     * @param isConstantNode if true node is treated as constant node.
     * @throws MatrixException throws exception is matrix is not defined.
     */
    Node(int id, Matrix referenceMatrix, boolean isConstantNode) throws MatrixException {
        if (referenceMatrix == null) throw new MatrixException("Reference matrix is not defined for the node.");
        this.id = id;
        this.referenceMatrix = referenceMatrix;
        this.isConstantNode = isConstantNode;
        if (isConstantNode) {
            constantMatrix = referenceMatrix;
            isMultiIndex = false;
        }
        else {
            matrices = new MMatrix();
            matrices.put(0, referenceMatrix);
            gradients = new MMatrix();
            isMultiIndex = true;
        }
    }

    /**
     * Constructor for node. Records dimensions of references matrix as node data dimensions.
     *
     * @param referenceMatrix reference matrix.
     * @param isConstantNode if true node is treated as constant node.
     * @throws MatrixException throws exception is matrix is not defined.
     */
    Node(int id, MMatrix referenceMatrix, boolean isConstantNode) throws MatrixException {
        this(id, referenceMatrix.get(referenceMatrix.firstKey()), isConstantNode);
        matrices = new MMatrix();
        for (Integer index : referenceMatrix.keySet()) matrices.put(index, referenceMatrix.get(index));
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
     * Set dependency node forward.
     *
     * @param toNode to node.
     */
    public void setToNode(Node toNode) {
        this.toNode = toNode;
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
     * Stores matrix dependency
     *
     * @param backupIndex backup index
     * @throws MatrixException throws exception if storing dependency fails.
     */
    public void storeMatrixDependency(int backupIndex) throws MatrixException {
        if (toNode == null) return;
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
        if (toNode == null) return;
        if (matrixBackup.containsKey(backupIndex)) {
            MMatrix matricesBackup = matrixBackup.get(backupIndex);
            for (Integer index : matricesBackup.keySet()) matrices.put(index, matricesBackup.get(index));
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
        Node node = new Node(id, referenceMatrix, isConstantNode);
        for (Integer index : keySet()) {
            node.setMatrix(index, getMatrix(index));
            if (copyGradients) node.setGradient(index, getGradient(index));
        }
        return node;
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
     * Returns if node is constant node type.
     *
     * @return true if node is constant node type otherwise false.
     */
    public boolean isConstantNode() {
        return isConstantNode;
    }

    /**
     * Sets multi index flag.
     *
     * @param isMultiIndex multi index flag.
     * @throws MatrixException throws exception if constant node is attempted to be set as of type multi-index.
     */
    public void setMultiIndex(boolean isMultiIndex) throws MatrixException {
        if (isConstantNode()) throw new MatrixException("Constant node cannot be of type multi-index.");
        this.isMultiIndex = isMultiIndex;
    }

    /**
     * Returns true if node is of type multi index.
     *
     * @return true if node is of type multi index.
     */
    public boolean isMultiIndex() {
        return isMultiIndex;
    }

    /**
     * Returns entry count.
     *
     * @return entry count.
     */
    public int getEntryCount() {
        return entryCount;
    }

    /**
     * Returns size of node.
     *
     * @return size of node.
     */
    public int size() {
        return isMultiIndex() ? matrices.size() : 1;
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
        return isMultiIndex() ? matrices.contains(matrix) : matrix == constantMatrix;
    }

    /**
     * Returns empty matrix with size of reference matrix.
     *
     * @return empty matrix with size of reference matrix.
     */
    public Matrix getEmptyMatrix() {
        return referenceMatrix.getNewMatrix();
    }

    /**
     * Resets node and removes other data than constant data.
     *
     * @param resetDependentNodes if true resets also dependent nodes.
     */
    public void resetNode(boolean resetDependentNodes) {
        if (isMultiIndex()) {
            if (toNode == null) matrices = new MMatrix();
            else if (resetDependentNodes) matrices = new MMatrix();
        }
        else if (!isConstantNode()) constantMatrix = getEmptyMatrix();
        gradients = new MMatrix();
        constantGradient = null;
        entryCount = 0;
        matrixBackup = new HashMap<>();
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
     * @throws MatrixException throws exception if scalar type of node and matrix are not matching or node is of type multi-index.
     */
    public void setMatrix(Matrix matrix) throws MatrixException {
        if (matrix.isScalar() != referenceMatrix.isScalar()) throw new MatrixException("Scalar type of node and matrix is not matching.");
        if (isMultiIndex()) throw new MatrixException("Node is of type multi-index");
        constantMatrix = matrix;
    }

    /**
     * Sets matrix of this node.
     *
     * @param index data index for matrix.
     * @param matrix new matrix.
     * @throws MatrixException throws exception if scalar type of node and matrix are not matching.
     */
    public void setMatrix(int index, Matrix matrix) throws MatrixException {
        if (!isConstantNode()) {
            if (matrix.isScalar() != referenceMatrix.isScalar()) throw new MatrixException("Scalar type of node and matrix is not matching.");
            if (isMultiIndex()) matrices.put(index, matrix);
            else constantMatrix = matrix;
        }
    }

    /**
     * Returns matrix of node.
     *
     * @return matrix of node.
     * @throws MatrixException throws exception if node is of type multi-index.
     */
    public Matrix getMatrix() throws MatrixException {
        if (isMultiIndex()) throw new MatrixException("Node is of type multi-index");
        return constantMatrix;
    }

    /**
     * Returns matrix of node.
     *
     * @param index data index for matrix.
     * @return matrix of node.
     */
    public Matrix getMatrix(int index) {
        return isMultiIndex() ? matrices.get(index) : constantMatrix;
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
     * @throws MatrixException throws exception if node is of type multi-index.
     */
    public void setGradient(Matrix gradient) throws MatrixException {
        if (isMultiIndex()) throw new MatrixException("Node is of type multi-index");
        constantGradient = gradient;
    }

    /**
     * Sets gradient matrix of node.
     *
     * @param index data index for gradient.
     * @param gradient gradient matrix of node.
     * @throws MatrixException throws exception if putting of matrix fails.
     */
    public void setGradient(int index, Matrix gradient) throws MatrixException {
        if (isMultiIndex()) gradients.put(index, gradient);
        else constantGradient = gradient;
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
     * @throws MatrixException throws exception if node is of type multi-index.
     */
    public Matrix getGradient() throws MatrixException {
        if (isMultiIndex()) throw new MatrixException("Node is of type multi-index");
        return constantGradient;
    }

    /**
     * Returns gradient matrix of node.
     *
     * @param index data index for gradient.
     * @return gradient matrix of node.
     */
    public Matrix getGradient(int index) {
        return isMultiIndex() ? gradients.get(index) : constantGradient;
    }

    /**
     * Returns gradients of node.
     *
     * @return gradients of node.
     */
    public MMatrix getGradients() {
        return gradients;
    }

    /**
     * Updates gradient.
     *
     * @param index data index.
     * @param outputGradient output gradient.
     * @param add if true output gradient contribution is added to node gradient otherwise subtracted.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void updateGradient(int index, Matrix outputGradient, boolean add) throws MatrixException {
        entryCount++;
        if (getGradient(index) == null) setGradient(index, getEmptyMatrix());
        if (!referenceMatrix.isScalar()) {
            if (add) getGradient(index).add(outputGradient, getGradient(index));
            else getGradient(index).subtract(outputGradient, getGradient(index));
        }
        else {
            if (add) getGradient(index).add(outputGradient.sum(), getGradient(index));
            else getGradient(index).subtract(outputGradient.sum(), getGradient(index));
        }
    }

    /**
     * Sets normalizers for node.
     *
     * @param normalizers normalizers for node.
     */
    public void setNormalizers(HashSet<Normalization> normalizers) {
        this.normalizers = normalizers;
    }

    /**
     * Sets regularizers for node.
     *
     * @param regularizers regularizers for node.
     */
    public void setRegularizers(HashSet<Regularization> regularizers) {
        this.regularizers = regularizers;
    }

    /**
     * Initializes normalization.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void initializeNormalization() throws MatrixException, DynamicParamException {
        if (referenceMatrix.isNormalized() && normalizers != null) {
            for (Normalization normalizer : normalizers) {
                if (isConstantNode()) normalizer.initialize(constantMatrix);
                else normalizer.initialize(this);
            }
        }
    }

    /**
     * Executes forward normalization to constant node.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void forwardNormalize() throws MatrixException, DynamicParamException {
        if (referenceMatrix.isNormalized() && normalizers != null) {
            for (Normalization normalizer : normalizers) {
                if (isConstantNode()) normalizer.forward(constantMatrix);
                else normalizer.forward(this);
            }
        }
    }

    /**
     * Executes forward normalization to specific entry (sample)
     *
     * @param sampleIndex sample index of specific entry.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void forwardNormalize(int sampleIndex) throws MatrixException, DynamicParamException {
        if (isConstantNode()) return;
        if (referenceMatrix.isNormalized() && normalizers != null) {
            for (Normalization normalizer : normalizers) {
                normalizer.forward(this, sampleIndex);
            }
        }
    }

    /**
     * Executes forward normalization finalization to constant node.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forwardNormalizeFinalize() throws MatrixException {
        if (!isConstantNode()) return;
        if (referenceMatrix.isNormalized() && normalizers != null) {
            for (Normalization normalizer : normalizers) {
                normalizer.forwardFinalize(constantMatrix);
            }
        }
    }

    /**
     * Executes backward normalization to constant entry of node.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void backwardNormalize() throws MatrixException, DynamicParamException {
        if (referenceMatrix.isNormalized() && normalizers != null) {
            for (Normalization normalizer : normalizers) {
                if (isConstantNode()) normalizer.backward(constantMatrix, constantGradient);
                else normalizer.backward(this);
            }
        }
    }

    /**
     * Executes backward normalization to specific entry (sample)
     *
     * @param sampleIndex sample index of specific entry.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void backwardNormalize(int sampleIndex) throws MatrixException, DynamicParamException {
        if (isConstantNode()) return;
        if (referenceMatrix.isNormalized() && normalizers != null) {
            for (Normalization normalizer : normalizers) {
                normalizer.backward(this, sampleIndex);
            }
        }
    }

    /**
     * Executes forward regularization step.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void forwardRegularize() throws MatrixException {
        if (isConstantNode()) return;
        if (referenceMatrix.isRegularized() && regularizers != null) {
            for (Regularization regularizer : regularizers) {
                regularizer.forward(matrices);
            }
        }
    }

    /**
     * Cumulates error from regularization.
     *
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @return updated error value.
     */
    public double cumulateRegularizationError() throws DynamicParamException, MatrixException {
        if (!isConstantNode()) return 0;
        double error = 0;
        if (referenceMatrix.isRegularized() && regularizers != null) {
            for (Regularization regularizer : regularizers) {
                error += regularizer.error(constantMatrix);
            }
        }
        return error;
    }

    /**
     * Executes backward regularization.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void backwardRegularize() throws MatrixException {
        if (!isConstantNode()) return;
        if (referenceMatrix.isRegularized() && regularizers != null) {
            for (Regularization regularizer : regularizers) {
                regularizer.backward(constantMatrix, constantGradient.divide(entryCount));
            }
        }
    }

}
