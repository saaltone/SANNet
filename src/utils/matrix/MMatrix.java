/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.matrix;

import utils.configurable.DynamicParamException;
import utils.procedure.ProcedureFactory;

import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

/**
 * Implements multi-matrix class that can execute matrix operations with multiple matrices such as adding multiple matrices together element by element.<br>
 * Class also has operations to calculate sum, mean, variance and standard deviation element by element over multiple matrices.<br>
 *
 */
public class MMatrix implements Cloneable, Serializable {

    @Serial
    private static final long serialVersionUID = 2208329722377770337L;

    /**
     * Map for matrices.
     *
     */
    private Matrix[] matrices;

    /**
     * Reference entry for multi-matrix.
     *
     */
    private Matrix referenceMatrix;

    /**
     * Number of rows in each matrix.
     *
     */
    private int rows = -1;

    /**
     * Number of columns in each matrix.
     *
     */
    private int columns = -1;

    /**
     * Procedure factory reference for matrix.<br>
     * Procedure factory records chain of executed matrix operations enabling dynamic construction of procedure and its gradient.<br>
     *
     */
    private transient ProcedureFactory procedureFactory = null;

    /**
     * Name for multi-matrix.
     *
     */
    private String name;

    /**
     * Constructor with depth limitation.
     *
     * @param depth depth of multi-matrix.
     * @throws MatrixException throws exception if depth is less than 1.
     */
    public MMatrix(int depth) throws MatrixException {
        if (depth < 1) throw new MatrixException("Depth must be at least 1.");
        matrices = new Matrix[depth];
    }

    /**
     * Constructor for sample with depth assumption of 1.
     *
     * @param matrix single entry for sample with assumption of maxSize 1.
     * @throws MatrixException throws exception if matrix is exceeding its depth or matrix is not defined.
     */
    public MMatrix(Matrix matrix) throws MatrixException {
        this(1, matrix);
    }

    /**
     * Constructor with depth limitation.
     *
     * @param depth depth of multi-matrix.
     * @param name name of matrix.
     * @throws MatrixException throws exception if depth is less than 1.
     */
    public MMatrix(int depth, String name) throws MatrixException {
        this(depth);
        this.name = name;
    }

    /**
     * Constructor for sample with depth limitation.
     *
     * @param depth depth of multi-matrix.
     * @param matrix single entry for sample.
     * @throws MatrixException throws exception if depth is less than 1.
     */
    public MMatrix(int depth, Matrix matrix) throws MatrixException {
        this(depth);
        rows = matrix.getTotalRows();
        columns = matrix.getTotalColumns();
        referenceMatrix = matrix;
        put(0, matrix);
    }

    /**
     * Constructor for multi-matrix without depth limitation.
     *
     * @param newMatrices matrices for multi-matrix.
     * @throws MatrixException throws exception if number of rows and columns are not matching for inserted matrices.
     */
    public MMatrix(MMatrix newMatrices) throws MatrixException {
        this(newMatrices.getDepth(), newMatrices.getName());
        int depth = getDepth();
        for (int index = 0; index < depth; index++) put(index, newMatrices.get(index));
    }

    /**
     * Sets name for matrix.
     *
     * @param name matrix name.
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * Sets name for matrix.
     *
     * @param name matrix name.
     * @param assignToMatrices assigns same name for all matrices contained currently by this matrix.
     */
    public void setName(String name, boolean assignToMatrices) {
        setName(name);
        if (assignToMatrices) for (Matrix matrix : matrices) matrix.setName(name);
    }

    /**
     * Returns name of matrix.
     *
     * @return name of matrix.
     */
    public String getName() {
        return name;
    }

    /**
     * Function used to reinitialize matrix and it's mask.
     *
     */
    public void clear() {
        matrices = new Matrix[matrices.length];
    }

    /**
     * Returns matrix from specific index.
     *
     * @param index specific index
     * @return matrix.
     */
    public Matrix get(int index) {
        return matrices[index];
    }

    /**
     * Puts matrix into specific index.
     *
     * @param index index
     * @param matrix matrix
     * @throws MatrixException throws exception if matrix is exceeding its depth or matrix is not defined.
     */
    public void put(int index, Matrix matrix) throws MatrixException {
        if (matrix == null) throw new MatrixException("Matrix is not defined.");
        if (index > matrices.length - 1) throw new MatrixException("Index is exceeding defined depth.");
        if (rows == -1 && columns == -1) {
            rows = matrix.getTotalRows();
            columns = matrix.getTotalColumns();
            referenceMatrix = matrix;
        }
        else if (rows != matrix.getTotalRows() || columns != matrix.getTotalColumns()) throw new MatrixException("Number of rows and columns are not matching for inserted matrices.");
        matrices[index] = matrix;
    }

    /**
     * Checks if multi-matrix contains specific entry.
     *
     * @param matrix specific entry.
     * @return returns true is matrix is contained inside sample.
     */
    public boolean contains(Matrix matrix) {
        for (Matrix thisMatrix : matrices) if (thisMatrix != matrix) return false;
        return true;
    }

    /**
     * Returns depth of multi-matrix.
     *
     * @return depth of multi-matrix.
     */
    public int getDepth() {
        return matrices.length;
    }

    /**
     * Returns new multi-matrix.
     *
     * @return new multi-matrix.
     * @throws MatrixException throws exception if creation of new matrix fails.
     */
    public MMatrix getNewMMatrix() throws MatrixException {
        return new MMatrix(getDepth());
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
     * Returns new matrix based on reference matrix.
     *
     * @return new matrix based on reference matrix.
     * @throws MatrixException throws exception if creation of new matrix fails.
     */
    public Matrix getNewMatrix() throws MatrixException {
        return referenceMatrix != null ? referenceMatrix.getNewMatrix() : null;
    }

    /**
     * Creates new multi-matrix with object reference to the matrix data of this multi-matrix.
     *
     * @return newly created reference matrix.
     * @throws MatrixException throws exception if mask operation fails or cloning of matrix fails.
     */
    public MMatrix reference() throws MatrixException {
        MMatrix newMMatrix;
        // Make shallow copy of matrix leaving references internal objects which are shared.
        try {
            newMMatrix = (MMatrix)super.clone();
        } catch (CloneNotSupportedException exception) {
            throw new MatrixException("Cloning of matrix failed.");
        }
        return newMMatrix;
    }

    /**
     * Creates new multi-matrix with object full copy of this multi-matrix.
     *
     * @return newly created reference matrix.
     * @throws MatrixException throws exception if mask is not set or cloning of matrix fails.
     */
    public MMatrix copy() throws MatrixException {
        MMatrix newMMatrix;
        // Make shallow copy of matrix leaving references internal objects which are shared.
        try {
            newMMatrix = (MMatrix)super.clone();
            // Copy matrix data
            newMMatrix.copyMatrixData(this);
        } catch (CloneNotSupportedException exception) {
            throw new MatrixException("Cloning of matrix failed.");
        }
        return newMMatrix;
    }

    /**
     * Makes full copy of multi-matrix content.
     *
     * @param newMMatrix multi-matrix to be copied.
     * @throws MatrixException throws exception if copying fails.
     */
    private void copyMatrixData(MMatrix newMMatrix) throws MatrixException {
        if (!hasEqualSize(newMMatrix)) {
            throw new MatrixException("Incompatible target matrix depth: " + newMMatrix.getDepth());
        }
        int depth = getDepth();
        for (int index = 0; index < depth; index++) put(index, newMMatrix.get(index).copy());
    }

    /**
     * Checks if this multi-matrix and other multi-matrix are equal in size.
     *
     * @param other other multi-matrix to be compared against.
     * @return true if multi-matrices are of same size otherwise false.
     */
    public boolean hasEqualSize(MMatrix other) {
        return other.getDepth() == getDepth();
    }

    /**
     * Sets procedure factory for multi-matrix.
     *
     * @param procedureFactory new procedure factory.
     */
    public void setProcedureFactory(ProcedureFactory procedureFactory) {
        this.procedureFactory = procedureFactory;
    }

    /**
     * Sets procedure factory for multi-matrix.
     *
     * @param procedureFactory new procedure factory.
     * @param setForMatrices if true set procedure factory for all matrices of multi-matrix.
     */
    public void setProcedureFactory(ProcedureFactory procedureFactory, boolean setForMatrices) {
        setProcedureFactory(procedureFactory);
        if (setForMatrices) for (Matrix matrix : matrices) matrix.setProcedureFactory(procedureFactory);
    }

    /**
     * Returns current procedure factory of multi-matrix.
     *
     * @return current procedure factory.
     */
    public ProcedureFactory getProcedureFactory() {
        return procedureFactory;
    }

    /**
     * Removes procedure factory.
     *
     */
    public void removeProcedureFactory() {
        for (Matrix matrix : matrices) matrix.removeProcedureFactory();
        this.procedureFactory = null;
    }

    /**
     * Returns true if multi-matrix has procedure factory otherwise false.
     *
     * @return true if multi-matrix has procedure factory otherwise false.
     */
    public boolean hasProcedureFactory() {
        return procedureFactory != null;
    }

    /**
     * Synchronizes this and other multi-matrix procedure factories.
     *
     * @param other other multi-matrix
     * @throws MatrixException throws exception if his and other matrices have conflicting procedure factories.
     */
    private void synchronizeProcedureFactory(MMatrix other) throws MatrixException {
        ProcedureFactory otherProcedureFactory = other.getProcedureFactory();
        if (procedureFactory != otherProcedureFactory) {
            if (procedureFactory == null) setProcedureFactory(otherProcedureFactory);
            else {
                if (otherProcedureFactory == null) other.setProcedureFactory(procedureFactory);
                else throw new MatrixException("This and other matrices have conflicting procedure factories.");
            }
        }
    }

    /**
     * Synchronizes this and other matrix procedure factories.
     *
     * @param other other matrix
     * @throws MatrixException throws exception if his and other matrices have conflicting procedure factories.
     */
    private void synchronizeProcedureFactory(Matrix other) throws MatrixException {
        ProcedureFactory otherProcedureFactory = other.getProcedureFactory();
        if (procedureFactory != otherProcedureFactory) {
            if (procedureFactory == null) setProcedureFactory(otherProcedureFactory);
            else {
                if (otherProcedureFactory == null) other.setProcedureFactory(procedureFactory);
                else throw new MatrixException("This and other matrices have conflicting procedure factories.");
            }
        }
    }

    /**
     * Makes current multi-matrix data equal to other multi-matrix data.
     *
     * @param other other multi-matrix to be copied as data of this multi-matrix.
     * @throws MatrixException throws MatrixException if this and other multi-matrix are not of equal dimensions.
     */
    public void setEqualTo(MMatrix other) throws MatrixException {
        if (!hasEqualSize(other)) {
            throw new MatrixException("Incompatible target matrix depth: " + other.getDepth());
        }
        int depth = getDepth();
        for (int index = 0; index < depth; index++) put(index, other.get(index));
    }

    /**
     * Checks if data of other multi-matrix is equal to data of this multi-matrix
     *
     * @param other multi-matrix to be compared.
     * @return true is data of this and other multi-matrix are equal otherwise false.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public boolean equals(MMatrix other) throws MatrixException {
        if (!hasEqualSize(other)) {
            throw new MatrixException("Incompatible target matrix depth: " + other.getDepth());
        }
        int depth = getDepth();
        for (int index = 0; index < depth; index++) if(!get(index).equals(other.get(index))) return false;
        return true;
    }

    /**
     * Applies unaryFunction to this multi-matrix.<br>
     * Example of operation can be applying square root operation to this multi-matrix.<br>
     * Applies masking if multi-matrix is masked.<br>
     *
     * @param unaryFunction unaryFunction to be applied.
     * @param result result multi-matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void apply(MMatrix result, UnaryFunction unaryFunction) throws MatrixException {
        if (getDepth() != result.getDepth()) throw new MatrixException("Depth of matrices are not matching.");
        int depth = getDepth();
        if (!hasProcedureFactory()) {
            for (int index = 0; index < depth; index++) result.put(index, get(index).apply(unaryFunction));
        }
        else {
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (int index = 0; index < depth; index++) result.put(index, get(index).apply(unaryFunction));
            procedureFactory.createUnaryFunctionExpression(expressionLock, this, result, unaryFunction);
        }
    }

    /**
     * Applies unaryFunction to this multi-matrix.<br>
     * Example of operation can be applying square root operation to this multi-matrix.<br>
     * Applies masking if multi-matrix is masked.<br>
     *
     * @param unaryFunction unaryFunction to be applied.
     * @return multi-matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public MMatrix apply(UnaryFunction unaryFunction) throws MatrixException {
        MMatrix result = getNewMMatrix();
        apply(result, unaryFunction);
        return result;
    }

    /**
     * Applies binaryFunction to this multi-matrix.<br>
     * Example of operation can be subtraction of other multi-matrix from this multi-matrix.<br>
     * Applies masking if multi-matrix is masked.<br>
     *
     * @param other other multi-matrix
     * @param result result multi-matrix.
     * @param binaryFunction binaryFunction to be applied.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void applyBi(MMatrix other, MMatrix result, BinaryFunction binaryFunction) throws MatrixException {
        if (getDepth() != other.getDepth() || getDepth() != result.getDepth()) throw new MatrixException("Depth of matrices are not matching.");
        int depth = getDepth();
        if (!hasProcedureFactory()) {
            for (int index = 0; index < depth; index++) result.put(index, get(index).applyBi(other.get(index), binaryFunction));
        }
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (int index = 0; index < depth; index++) result.put(index, get(index).applyBi(other.get(index), binaryFunction));
            procedureFactory.createBinaryFunctionExpression(expressionLock, this, other, result, binaryFunction);
        }
    }

    /**
     * Applies binaryFunction to this multi-matrix.<br>
     * Example of operation can be subtraction of other multi-matrix from this multi-matrix.<br>
     * Applies masking if multi-matrix is masked.<br>
     *
     * @param other other multi-matrix
     * @param binaryFunction binaryFunction to be applied.
     * @return multi-matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public MMatrix applyBi(MMatrix other, BinaryFunction binaryFunction) throws MatrixException {
        MMatrix result = getNewMMatrix();
        applyBi(other, result, binaryFunction);
        return result;
    }

    /**
     * Applies binaryFunction to this multi-matrix.<br>
     * Example of operation can be subtraction of other multi-matrix from this multi-matrix.<br>
     * Applies masking if multi-matrix is masked.<br>
     *
     * @param other other matrix
     * @param result result multi-matrix.
     * @param binaryFunction binaryFunction to be applied.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void applyBi(Matrix other, MMatrix result, BinaryFunction binaryFunction) throws MatrixException {
        if (getDepth() != result.getDepth()) throw new MatrixException("Depth of matrices are not matching.");
        int depth = getDepth();
        if (!hasProcedureFactory()) {
            for (int index = 0; index < depth; index++) result.put(index, get(index).applyBi(other, binaryFunction));
        }
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (int index = 0; index < depth; index++) result.put(index, get(index).applyBi(other, binaryFunction));
            procedureFactory.createBinaryFunctionExpression(expressionLock, this, other, result, binaryFunction);
        }
    }

    /**
     * Applies binaryFunction to this multi-matrix.<br>
     * Example of operation can be subtraction of other matrix from this multi-matrix.<br>
     * Applies masking if multi-matrix is masked.<br>
     *
     * @param other other matrix
     * @param binaryFunction binaryFunction to be applied.
     * @return multi-matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public MMatrix applyBi(Matrix other, BinaryFunction binaryFunction) throws MatrixException {
        MMatrix result = getNewMMatrix();
        applyBi(other, result, binaryFunction);
        return result;
    }

    /**
     * Adds other multi-matrix to this multi-matrix.
     *
     * @param other multi-matrix which acts as second variable in the operation.
     * @param result multi-matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void add(MMatrix other, MMatrix result) throws MatrixException {
        if (getDepth() != other.getDepth() || getDepth() != result.getDepth()) throw new MatrixException("Depth of matrices are not matching.");
        int depth = getDepth();
        if (!hasProcedureFactory()) {
            for (int index = 0; index < depth; index++) result.put(index, get(index).add(other.get(index)));
        }
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (int index = 0; index < depth; index++) result.put(index, get(index).add(other.get(index)));
            procedureFactory.createAddExpression(expressionLock, this, other, result);
        }
    }

    /**
     * Adds other multi-matrix to this multi-matrix.
     *
     * @param other multi-matrix which acts as second variable in the operation.
     * @return multi-result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public MMatrix add(MMatrix other) throws MatrixException {
        MMatrix result = getNewMMatrix();
        add(other, result);
        return result;
    }

    /**
     * Adds other matrix to this multi-matrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result multi-matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void add(Matrix other, MMatrix result) throws MatrixException {
        if (getDepth() != result.getDepth()) throw new MatrixException("Depth of matrices are not matching.");
        int depth = getDepth();
        if (!hasProcedureFactory()) {
            for (int index = 0; index < depth; index++) result.put(index, get(index).add(other));
        }
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (int index = 0; index < depth; index++) result.put(index, get(index).add(other));
            procedureFactory.createAddExpression(expressionLock, this, other, result);
        }
    }

    /**
     * Adds other matrix to this multi-matrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @return multi-result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public MMatrix add(Matrix other) throws MatrixException {
        MMatrix result = getNewMMatrix();
        add(other, result);
        return result;
    }

    /**
     * Subtracts other multi-matrix from this multi-matrix.
     *
     * @param other multi-matrix which acts as second variable in the operation.
     * @param result multi-matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void subtract(MMatrix other, MMatrix result) throws MatrixException {
        if (getDepth() != other.getDepth() || getDepth() != result.getDepth()) throw new MatrixException("Depth of matrices are not matching.");
        int depth = getDepth();
        if (!hasProcedureFactory()) {
            for (int index = 0; index < depth; index++) result.put(index, get(index).subtract(other.get(index)));
        }
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (int index = 0; index < depth; index++) result.put(index, get(index).subtract(other.get(index)));
            procedureFactory.createSubtractExpression(expressionLock, this, other, result);
        }
    }

    /**
     * Subtracts other multi-matrix from this multi-matrix.
     *
     * @param other multi-matrix which acts as second variable in the operation.
     * @return result multi-matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public MMatrix subtract(MMatrix other) throws MatrixException {
        MMatrix result = getNewMMatrix();
        subtract(other, result);
        return result;
    }

    /**
     * Subtracts other matrix from this multi-matrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result multi-matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void subtract(Matrix other, MMatrix result) throws MatrixException {
        if (getDepth() != result.getDepth()) throw new MatrixException("Depth of matrices are not matching.");
        int depth = getDepth();
        if (!hasProcedureFactory()) {
            for (int index = 0; index < depth; index++) result.put(index, get(index).subtract(other));
        }
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (int index = 0; index < depth; index++) result.put(index, get(index).subtract(other));
            procedureFactory.createSubtractExpression(expressionLock, this, other, result);
        }
    }

    /**
     * Subtracts other matrix from this multi-matrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @return result multi-matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public MMatrix subtract(Matrix other) throws MatrixException {
        MMatrix result = getNewMMatrix();
        subtract(other, result);
        return result;
    }

    /**
     * Multiplies other multi-matrix with this multi-matrix.
     *
     * @param other multi-matrix which acts as second variable in the operation.
     * @param result multi-matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void multiply(MMatrix other, MMatrix result) throws MatrixException {
        if (getDepth() != other.getDepth() || getDepth() != result.getDepth()) throw new MatrixException("Depth of matrices are not matching.");
        int depth = getDepth();
        if (!hasProcedureFactory()) {
            for (int index = 0; index < depth; index++) result.put(index, get(index).multiply(other.get(index)));
        }
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (int index = 0; index < depth; index++) result.put(index, get(index).multiply(other.get(index)));
            procedureFactory.createMultiplyExpression(expressionLock, this, other, result);
        }
    }

    /**
     * Multiplies other multi-matrix with this multi-matrix.
     *
     * @param other multi-matrix which acts as second variable in the operation.
     * @return result multi-matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public MMatrix multiply(MMatrix other) throws MatrixException {
        MMatrix result = getNewMMatrix();
        multiply(other, result);
        return result;
    }

    /**
     * Multiplies other matrix with this multi-matrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result multi-matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void multiply(Matrix other, MMatrix result) throws MatrixException {
        if (getDepth() != result.getDepth()) throw new MatrixException("Depth of matrices are not matching.");
        int depth = getDepth();
        if (!hasProcedureFactory()) {
            for (int index = 0; index < depth; index++) result.put(index, get(index).multiply(other));
        }
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (int index = 0; index < depth; index++) result.put(index, get(index).multiply(other));
            procedureFactory.createMultiplyExpression(expressionLock, this, other, result);
        }
    }

    /**
     * Multiplies other matrix with this multi-matrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @return result multi-matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public MMatrix multiply(Matrix other) throws MatrixException {
        MMatrix result = getNewMMatrix();
        multiply(other, result);
        return result;
    }

    /**
     * Dots other multi-matrix with this multi-matrix.
     *
     * @param other multi-matrix which acts as second variable in the operation.
     * @param result multi-matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void dot(MMatrix other, MMatrix result) throws MatrixException {
        if (getDepth() != other.getDepth() || getDepth() != result.getDepth()) throw new MatrixException("Depth of matrices are not matching.");
        int depth = getDepth();
        if (!hasProcedureFactory()) {
            for (int index = 0; index < depth; index++) result.put(index, get(index).dot(other.get(index)));
        }
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (int index = 0; index < depth; index++) result.put(index, get(index).dot(other.get(index)));
            procedureFactory.createDotExpression(expressionLock, this, other, result);
        }
    }

    /**
     * Dots other multi-matrix with this multi-matrix.
     *
     * @param other multi-matrix which acts as second variable in the operation.
     * @return result multi-matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public MMatrix dot(MMatrix other) throws MatrixException {
        MMatrix result = getNewMMatrix();
        dot(other, result);
        return result;
    }

    /**
     * Dots other matrix with this multi-matrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result multi-matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void dot(Matrix other, MMatrix result) throws MatrixException {
        if (getDepth() != result.getDepth()) throw new MatrixException("Depth of matrices are not matching.");
        int depth = getDepth();
        if (!hasProcedureFactory()) {
            for (int index = 0; index < depth; index++) result.put(index, get(index).dot(other));
        }
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (int index = 0; index < depth; index++) result.put(index, get(index).dot(other));
            procedureFactory.createDotExpression(expressionLock, this, other, result);
        }
    }

    /**
     * Dots other matrix with this multi-matrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @return result multi-matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public MMatrix dot(Matrix other) throws MatrixException {
        MMatrix result = getNewMMatrix();
        dot(other, result);
        return result;
    }

    /**
     * Divides this multi-matrix with other multi-matrix.
     *
     * @param other multi-matrix which acts as second variable in the operation.
     * @param result multi-matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void divide(MMatrix other, MMatrix result) throws MatrixException {
        if (getDepth() != other.getDepth() || getDepth() != result.getDepth()) throw new MatrixException("Depth of matrices are not matching.");
        int depth = getDepth();
        if (!hasProcedureFactory()) {
            for (int index = 0; index < depth; index++) result.put(index, get(index).divide(other.get(index)));
        }
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (int index = 0; index < depth; index++) result.put(index, get(index).divide(other.get(index)));
            procedureFactory.createDivideExpression(expressionLock, this, other, result);
        }
    }

    /**
     * Divides this multi-matrix with other multi-matrix.
     *
     * @param other multi-other matrix.
     * @return result multi-matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public MMatrix divide(MMatrix other) throws MatrixException {
        MMatrix result = getNewMMatrix();
        divide(other, result);
        return result;
    }

    /**
     * Divides this multi-matrix with other matrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result multi-matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void divide(Matrix other, MMatrix result) throws MatrixException {
        if (getDepth() != result.getDepth()) throw new MatrixException("Depth of matrices are not matching.");
        int depth = getDepth();
        if (!hasProcedureFactory()) {
            for (int index = 0; index < depth; index++) result.put(index, get(index).divide(other));
        }
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (int index = 0; index < depth; index++) result.put(index, get(index).divide(other));
            procedureFactory.createDivideExpression(expressionLock, this, other, result);
        }
    }

    /**
     * Divides this multi-matrix with other matrix.
     *
     * @param other other matrix.
     * @return result multi-matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public MMatrix divide(Matrix other) throws MatrixException {
        MMatrix result = getNewMMatrix();
        divide(other, result);
        return result;
    }

    /**
     * Calculates sum or mean.
     *
     * @param asMean if true returns mean otherwise sum.
     * @return result of sum or mean
     * @throws MatrixException throws exception if row or column vectors are incorrectly provided.
     */
    public Matrix count(boolean asMean) throws MatrixException {
        return count(asMean, getNewMatrix());
    }

    /**
     * Calculates sum or mean.
     *
     * @param asMean if true returns mean otherwise sum.
     * @param result result matrix.
     * @return result of sum or mean
     * @throws MatrixException throws exception if row or column vectors are incorrectly provided.
     */
    public Matrix count(boolean asMean, Matrix result) throws MatrixException {
        for (Matrix matrix : matrices) {
            result.add(matrix, result);
        }
        return asMean ? result.divide(getDepth()) : result;
    }

    /**
     * Calculates sum or mean.
     *
     * @param matrices matrices.
     * @param asMean if true returns mean otherwise sum.
     * @return result of sum or mean
     * @throws MatrixException throws exception if row or column vectors are incorrectly provided.
     */
    public static Matrix count(TreeMap<Integer, Matrix> matrices, boolean asMean) throws MatrixException {
        Matrix result = null;
        for (Matrix matrix : matrices.values()) {
            if (result == null) result = matrix.getNewMatrix();
            result.add(matrix, result);
        }
        return asMean ? result == null ? null : result.divide(matrices.size()) : result;
    }

    /**
     * Calculates sum.
     *
     * @param matrices matrices.
     * @return resulting sum
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     */
    public static Matrix sum(TreeMap<Integer, Matrix> matrices) throws MatrixException {
        return MMatrix.count(matrices, false);
    }

    /**
     * Calculates sum.
     *
     * @return resulting sum
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     */
    public Matrix sum() throws MatrixException {
        return sum(getNewMatrix());
    }

    /**
     * Calculates sum.
     *
     * @param result result matrix.
     * @return resulting sum
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     */
    public Matrix sum(Matrix result) throws MatrixException {
        if (!hasProcedureFactory()) return count(false, result);
        else {
            double expressionLock = procedureFactory.startExpression(this);
            count(false, result);
            result.setProcedureFactory(procedureFactory);
            procedureFactory.createSumExpression(expressionLock, this, result);
            return result;
        }
    }

    /**
     * Calculates mean.
     *
     * @param matrices matrices.
     * @return resulting mean
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     */
    public static Matrix mean(TreeMap<Integer, Matrix> matrices) throws MatrixException {
        return MMatrix.count(matrices, true);
    }

    /**
     * Calculates mean.
     *
     * @return resulting mean
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     */
    public Matrix mean() throws MatrixException {
        return mean(getNewMatrix());
    }

    /**
     * Calculates mean.
     *
     * @param result result matrix.
     * @return resulting mean
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     */
    public Matrix mean(Matrix result) throws MatrixException {
        if (!hasProcedureFactory()) return count(true, result);
        else {
            double expressionLock = procedureFactory.startExpression(this);
            count(true, result);
            result.setProcedureFactory(procedureFactory);
            procedureFactory.createMeanExpression(expressionLock, this, result);
            return result;
        }
    }

    /**
     * Calculates variance.
     *
     * @param matrices matrices.
     * @return resulting variance
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public static Matrix variance(TreeMap<Integer, Matrix> matrices) throws MatrixException, DynamicParamException {
        return MMatrix.variance(matrices, MMatrix.mean(matrices));
    }

    /**
     * Calculates variance.
     *
     * @param matrices matrices.
     * @param meanMatrix matrix containing mean values for variance calculation.
     * @return resulting variance
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public static Matrix variance(TreeMap<Integer, Matrix> matrices, Matrix meanMatrix) throws MatrixException, DynamicParamException {
        if (meanMatrix == null) throw new MatrixException("Mean matrix is not defined");
        Matrix result = null;
        for (Matrix matrix : matrices.values()) {
            if (result == null) result = matrix.getNewMatrix();
            result.add(matrix.subtract(meanMatrix).power(2), result);
        }
        return result == null ? null : result.divide(matrices.size());
    }

    /**
     * Calculates variance.
     *
     * @return resulting variance
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix variance() throws MatrixException, DynamicParamException {
        return variance(mean(), getNewMatrix());
    }

    /**
     * Calculates variance.
     *
     * @param meanMatrix matrix containing mean values for variance calculation.
     * @return resulting variance
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix variance(Matrix meanMatrix) throws MatrixException, DynamicParamException {
        return variance(meanMatrix, getNewMatrix());
    }

    /**
     * Calculates variance.
     *
     * @param meanMatrix matrix containing mean values for variance calculation.
     * @param result result matrix.
     * @return resulting variance
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix variance(Matrix meanMatrix, Matrix result) throws MatrixException, DynamicParamException {
        if (meanMatrix == null) throw new MatrixException("Mean matrix is not defined");
        if (!hasProcedureFactory()) {
            for (Matrix matrix : matrices) {
                result.add(matrix.subtract(meanMatrix).power(2), result);
            }
            result.divide(getDepth(), result);
        }
        else {
            double expressionLock = procedureFactory.startExpression(this);
            for (Matrix matrix : matrices) {
                result.add(matrix.subtract(meanMatrix).power(2), result);
            }
            result.divide(getDepth(), result);
            result.setProcedureFactory(procedureFactory);
            procedureFactory.createVarianceExpression(expressionLock, this, result);
        }
        return result;
    }

    /**
     * Calculates standard deviation.
     *
     * @return resulting standard deviation
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix standardDeviation() throws MatrixException, DynamicParamException {
        return standardDeviation(mean(), getNewMatrix());
    }

    /**
     * Calculates standard deviation.
     *
     * @param matrices matrices.
     * @return resulting standard deviation
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public static Matrix standardDeviation(TreeMap<Integer, Matrix> matrices) throws MatrixException, DynamicParamException {
        return MMatrix.variance(matrices, MMatrix.mean(matrices)).multiply(matrices.size()).divide(matrices.size() - 1).apply(UnaryFunctionType.SQRT);
    }

    /**
     * Calculates standard deviation.
     *
     * @param matrices matrices.
     * @param meanMatrix matrix containing mean values for standard deviation calculation.
     * @return resulting standard deviation
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public static Matrix standardDeviation(TreeMap<Integer, Matrix> matrices, Matrix meanMatrix) throws MatrixException, DynamicParamException {
        return MMatrix.variance(matrices, meanMatrix).multiply(matrices.size()).divide(matrices.size() - 1).apply(UnaryFunctionType.SQRT);
    }

    /**
     * Calculates standard deviation.
     *
     * @param meanMatrix matrix containing mean values for standard deviation calculation.
     * @return resulting standard deviation
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix standardDeviation(Matrix meanMatrix) throws MatrixException, DynamicParamException {
        return standardDeviation(meanMatrix, getNewMatrix());
    }

    /**
     * Calculates standard deviation.
     *
     * @param meanMatrix matrix containing mean values for standard deviation calculation.
     * @param result result matrix.
     * @return resulting standard deviation
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix standardDeviation(Matrix meanMatrix, Matrix result) throws MatrixException, DynamicParamException {
        if (!hasProcedureFactory()) variance(meanMatrix).multiply(getDepth()).divide(getDepth() - 1).apply(result, UnaryFunctionType.SQRT);
        else {
            double expressionLock = procedureFactory.startExpression(this);
            variance(meanMatrix).multiply(getDepth()).divide(getDepth() - 1).apply(result, UnaryFunctionType.SQRT);
            result.setProcedureFactory(procedureFactory);
            procedureFactory.createStandardDeviationExpression(expressionLock, this, result);
        }
        return result;
    }

    /**
     * Calculates softmax.
     *
     * @param result result multi-matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void softmax(MMatrix result) throws MatrixException {
        int depth = getDepth();
        for (int index = 0; index < depth; index++) result.put(index, get(index).softmax());
    }

    /**
     * Calculates softmax.
     *
     * @return multi-matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public MMatrix softmax() throws MatrixException {
        MMatrix result = getNewMMatrix();
        softmax(result);
        return result;
    }

    /**
     * Calculates Gumbel softmax.
     *
     * @param result result multi-matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void gumbelSoftmax(MMatrix result) throws MatrixException {
        gumbelSoftmax(result, 1);
    }

    /**
     * Calculates Gumbel softmax.
     *
     * @param gumbelSoftmaxTau tau value for Gumbel Softmax.
     * @param result result multi-matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void gumbelSoftmax(MMatrix result, double gumbelSoftmaxTau) throws MatrixException {
        int depth = getDepth();
        for (int index = 0; index < depth; index++) result.put(index, get(index).gumbelSoftmax(gumbelSoftmaxTau));
    }

    /**
     * Calculates Gumbel softmax.
     *
     * @return multi-matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public MMatrix gumbelSoftmax() throws MatrixException {
        MMatrix result = getNewMMatrix();
        gumbelSoftmax(result, 1);
        return result;
    }

    /**
     * Calculates Gumbel softmax.
     *
     * @param gumbelSoftmaxTau tau value for Gumbel Softmax.
     * @return multi-matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public MMatrix gumbelSoftmax(double gumbelSoftmaxTau) throws MatrixException {
        MMatrix result = getNewMMatrix();
        gumbelSoftmax(result, gumbelSoftmaxTau);
        return result;
    }

    /**
     * Splits matrix at defined position. If splitVertical is true splits vertically otherwise horizontally.
     *
     * @param position position of split
     * @param splitVertically if true splits vertically otherwise horizontally.
     * @return split multi-matrix.
     * @throws MatrixException throws matrix exception if splitting fails.
     */
    public MMatrix split(int position, boolean splitVertically) throws MatrixException {
        MMatrix splitMMatrix = getNewMMatrix();
        int depth = getDepth();
        for (int index = 0; index < depth; index++) splitMMatrix.put(index, get(index).split(position, splitVertically));
        return splitMMatrix;
    }

    /**
     * Flattens multi-matrix into one dimensional column vector (matrix)
     *
     * @return flattened multi-matrix
     * @throws MatrixException throws exception if creation of multi-matrix fails.
     */
    public MMatrix flatten() throws MatrixException {
        int rows = this.rows;
        int cols = this.columns;
        Matrix flattenedMatrix = new DMatrix(rows * cols * getDepth(), 1);
        MMatrix flattenedMMatrix = new MMatrix(1, flattenedMatrix);
        int depth = getDepth();
        for (int index = 0; index < depth; index++) {
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    flattenedMatrix.setValue(getPosition(rows, cols, row, col, index), 0 , get(index).getValue(row, col));
                }
            }
        }
        return flattenedMMatrix;
    }

    /**
     * Returns unflattened multi-matrix i.e. samples that have been unflattened from single column vector.
     *
     * @param width width of unflattened multi-matrix.
     * @param height height of unflattened multi-matrix.
     * @param depth depth of unflattened multi-matrix.
     * @return unflattened multi-matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public MMatrix unflatten(int width, int height, int depth) throws MatrixException {
        if (getDepth() != 1) throw new MatrixException("MMatrix cannot be unflattened since it is not column vector (size equals 1).");
        MMatrix mMatrix = new MMatrix(depth);
        for (int index = 0; index < depth; index++) {
            Matrix matrix = new DMatrix(width, height);
            mMatrix.put(index, matrix);
            for (int row = 0; row < width; row++) {
                for (int col = 0; col < height; col++) {
                    matrix.setValue(row, col, get(0).getValue(getPosition(width, height, row, col, index), 0));
                }
            }
        }
        return mMatrix;
    }

    /**
     * Returns one dimensional index calculated based on width, height and depth.
     *
     * @param w weight as input
     * @param h height as input
     * @param d depth as input
     * @return one dimensional index
     */
    private int getPosition(int maxWidth, int maxHeight, int w, int h, int d) {
        return w + maxWidth * h + maxWidth * maxHeight * d;
    }

    /**
     * Joins multi-matrices.
     *
     * @param mMatrices multi-matrices.
     * @param joinedVertically if true MMatrices are joint vertically otherwise horizontally.
     * @return joint multi-matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public static MMatrix join(MMatrix[] mMatrices, boolean joinedVertically) throws MatrixException {
        int depth = -1;
        HashMap<Integer, ArrayList<Matrix>> mMatrixArrays = new HashMap<>();
        MMatrix joinedMMatrix = null;
        for (MMatrix mMatrix : mMatrices) {
            if (joinedMMatrix == null) joinedMMatrix = mMatrix.getNewMMatrix();
            if (depth == -1) depth = mMatrix.getDepth();
            for (int depthIndex = 0; depthIndex < depth; depthIndex++) {
                Matrix matrix = mMatrix.get(depthIndex);
                if (matrix == null) throw new MatrixException("Other multi-matrix does not contain depth index: " + depthIndex);
                mMatrixArrays.computeIfAbsent(depthIndex, k -> new ArrayList<>()).add(matrix);
            }
        }
        for (int depthIndex = 0; depthIndex < depth; depthIndex++) {
            joinedMMatrix.put(depthIndex, new JMatrix(mMatrixArrays.get(depthIndex), joinedVertically));
        }
        return joinedMMatrix;
    }

    /**
     * Unjoins multi-matrix.
     *
     * @param mMatrix multi-matrix.
     * @return unjoined multi-matrices.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public static MMatrix[] unjoin(MMatrix mMatrix) throws MatrixException {
        int depth = mMatrix.getDepth();
        int subMatricesSize = -1;
        MMatrix[] unjoinedMMatrices = null;
        for (int depthIndex = 0; depthIndex < depth; depthIndex++) {
            ArrayList<Matrix> subMatrices = mMatrix.get(depthIndex).getSubMatrices();
            if (unjoinedMMatrices == null) {
                subMatricesSize = subMatrices.size();
                unjoinedMMatrices = new MMatrix[subMatricesSize];
                for (int subMatrixIndex = 0; subMatrixIndex < subMatricesSize; subMatrixIndex++) unjoinedMMatrices[subMatrixIndex] = new MMatrix(depth);
            }
            for (int subMatrixIndex = 0; subMatrixIndex < subMatricesSize; subMatrixIndex++) {
                unjoinedMMatrices[subMatrixIndex].put(depthIndex, subMatrices.get(subMatrixIndex));
            }
        }
        return unjoinedMMatrices;
    }

    /**
     * Creates multi-matrices out of matrices.
     *
     * @param matrices matrices
     * @return multi-matrices.
     * @throws MatrixException throws exception if matrix is exceeding its depth or matrix is not defined.
     */
    public static TreeMap<Integer, MMatrix> getMMatrices(TreeMap<Integer, Matrix> matrices) throws MatrixException {
        TreeMap<Integer, MMatrix> mMatrices = new TreeMap<>();
        for (Map.Entry<Integer, Matrix> entry : matrices.entrySet()) mMatrices.put(entry.getKey(), new MMatrix(entry.getValue()));
        return mMatrices;
    }

}
