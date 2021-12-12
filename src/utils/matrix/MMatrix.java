/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package utils.matrix;

import utils.configurable.DynamicParamException;
import utils.procedure.ProcedureFactory;

import java.io.Serial;
import java.io.Serializable;
import java.util.*;

/**
 * Defines multi-matrix class that can execute matrix operations with multiple matrices such as adding multiple matrices together element by element.<br>
 * Class also has operations to calculate sum, mean, variance and standard deviation over multiple matrices element by element.<br>
 *
 */
public class MMatrix implements Cloneable, Serializable {

    @Serial
    private static final long serialVersionUID = 2208329722377770337L;

    /**
     * Depth for MMatrix.
     *
     */
    private final int depth;

    /**
     * Map for matrices.
     *
     */
    private HashMap<Integer, Matrix> matrices;

    /**
     * Reference entry for MMatrix.
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
     * Procedure factory reference for matrix.
     * Procedure factory records chain of executed matrix operations enabling dynamic construction of procedure and it's gradient.
     *
     */
    private transient ProcedureFactory procedureFactory = null;

    /**
     * Name for MMatrix.
     *
     */
    private String name;

    /**
     * Constructor for MMatrix without depth limitation.
     *
     */
    public MMatrix() {
        matrices = new HashMap<>();
        depth = -1;
    }

    /**
     * Constructor for sample with depth assumption of 1.
     *
     * @param matrix single entry for sample with assumption of maxSize 1.
     */
    public MMatrix(Matrix matrix) {
        rows = matrix.getTotalRows();
        columns = matrix.getTotalColumns();
        matrices = new HashMap<>();
        depth = 1;
        matrices.put(0, matrix);
        referenceMatrix = matrix;
    }

    /**
     * Constructor with depth limitation.
     *
     * @param depth depth of MMatrix.
     * @throws MatrixException throws exception if depth is less than 1.
     */
    public MMatrix(int depth) throws MatrixException {
        if (depth < 1) throw new MatrixException("Depth must be at least 1.");
        matrices = new HashMap<>();
        this.depth = depth;
    }

    /**
     * Constructor with depth limitation.
     *
     * @param depth depth of MMatrix.
     * @param name name of matrix.
     * @throws MatrixException throws exception if depth is less than 1.
     */
    public MMatrix(int depth, String name) throws MatrixException {
        if (depth < 1) throw new MatrixException("Depth must be at least 1.");
        matrices = new HashMap<>();
        this.depth = depth;
        this.name = name;
    }

    /**
     * Constructor for sample with depth limitation.
     *
     * @param depth depth of MMatrix.
     * @param matrix single entry for sample.
     * @throws MatrixException throws exception if depth is less than 1.
     */
    public MMatrix(int depth, Matrix matrix) throws MatrixException {
        this(depth);
        rows = matrix.getTotalRows();
        columns = matrix.getTotalColumns();
        matrices.put(0, matrix);
        referenceMatrix = matrix;
    }

    /**
     * Constructor for MMatrix without depth limitation.
     *
     * @param matrices matrices for MMatrix.
     * @throws MatrixException throws exception if number of rows and columns are not matching for inserted matrices.
     */
    public MMatrix(HashMap<Integer, Matrix> matrices) throws MatrixException {
        depth = -1;
        for (Integer depthIndex : matrices.keySet()) {
            Matrix matrix = matrices.get(depthIndex);
            if (rows == -1 && columns == -1) {
                rows = matrix.getTotalRows();
                columns = matrix.getTotalColumns();
                referenceMatrix = matrix;
            }
            else if (matrix.getTotalRows() != rows || matrix.getTotalColumns() != columns) throw new MatrixException("Number of rows and columns are not matching for inserted matrices.");
            matrices.put(depthIndex, matrix);
        }
    }

    /**
     * Constructor for MMatrix without depth limitation.
     *
     * @param matrices matrices for MMatrix.
     * @throws MatrixException throws exception if number of rows and columns are not matching for inserted matrices.
     */
    public MMatrix(MMatrix matrices) throws MatrixException {
        depth = -1;
        for (Integer depthIndex : matrices.keySet()) {
            Matrix matrix = matrices.get(depthIndex);
            if (rows == -1 && columns == -1) {
                rows = matrix.getTotalRows();
                columns = matrix.getTotalColumns();
                referenceMatrix = matrix;
            }
            else if (matrix.getTotalRows() != rows || matrix.getTotalColumns() != columns) throw new MatrixException("Number of rows and columns are not matching for inserted matrices.");
            this.matrices.put(depthIndex, matrix);
        }
        this.name = matrices.getName();
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
        if (assignToMatrices) for (Matrix matrix : matrices.values()) matrix.setName(name);
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
        matrices = new HashMap<>();
    }

    /**
     * Returns true if contains matrix at specific index.
     *
     * @param index index
     * @return true if contains matrix at specific index otherwise false.
     */
    public boolean containsKey(int index) {
        return matrices.containsKey(index);
    }

    /**
     * Returns matrices.
     *
     * @return matrices.
     */
    public Map<Integer, Matrix> get() {
        return matrices;
    }

    /**
     * Returns matrix from specific index.
     *
     * @param index specific index
     * @return matrix.
     */
    public Matrix get(int index) {
        return matrices.get(index);
    }

    /**
     * Puts matrix into specific index.
     *
     * @param index index
     * @param matrix matrix
     * @throws MatrixException throws exception if matrix is exceeding its depth.
     */
    public void put(int index, Matrix matrix) throws MatrixException {
        if (!matrices.containsKey(index) && depth > 0 && matrices.size() >= depth) throw new MatrixException("MMatrix is exceeding defined depth.");
        if (rows == -1 && columns == -1) {
            rows = matrix.getTotalRows();
            columns = matrix.getTotalColumns();
            referenceMatrix = matrix;
        }
        else if (rows != matrix.getTotalRows() || columns != matrix.getTotalColumns()) throw new MatrixException("Number of rows and columns are not matching for inserted matrices.");
        matrices.put(index, matrix);
    }

    /**
     * Removes matrix at specific index.
     *
     * @param index specific index.
     */
    public void remove(int index) {
        matrices.remove(index);
    }

    /**
     * Returns key set containing matrix indices.
     *
     * @return key set containing matrix indices.
     */
    public Set<Integer> keySet() {
        return matrices.keySet();
    }

    /**
     * Returns matrices as collection.
     *
     * @return matrices as collection.
     */
    public Collection<Matrix> values() {
        return matrices.values();
    }

    /**
     * Checks if MMatrix contains specific entry.
     *
     * @param matrix specific entry.
     * @return returns true is matrix is contained inside sample.
     */
    public boolean contains(Matrix matrix) {
        return matrices.containsValue(matrix);
    }

    /**
     * Returns depth of MMatrix.
     *
     * @return depth of MMatrix.
     */
    public int getDepth() {
        return depth;
    }

    /**
     * Returns number of matrices stored.
     *
     * @return number of matrices stored.
     */
    public int size() {
        return matrices.size();
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
     * Creates new MMatrix with object reference to the matrix data of this MMatrix.
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
     * Creates new MMatrix with object full copy of this MMatrix.
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
     * Makes full copy of MMatrix content.
     *
     * @param newMMatrix MMatrix to be copied.
     * @throws MatrixException throws exception if copying fails.
     */
    private void copyMatrixData(MMatrix newMMatrix) throws MatrixException {
        for (Integer index : newMMatrix.keySet()) put(index, newMMatrix.get(index).copy());
    }

    /**
     * Checks if this matrix and other matrix are equal in size.
     *
     * @param other other matrix to be compared against.
     * @return true if matrices are of same size otherwise false.
     */
    public boolean hasEqualSize(MMatrix other) {
        return other.size() == size();
    }

    /**
     * Sets procedure factory for matrix.
     *
     * @param procedureFactory new procedure factory.
     */
    public void setProcedureFactory(ProcedureFactory procedureFactory) {
        this.procedureFactory = procedureFactory;
    }

    /**
     * Returns current procedure factory of matrix.
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
        for (Matrix matrix : matrices.values()) matrix.removeProcedureFactory();
        this.procedureFactory = null;
    }

    /**
     * Returns true if matrix has procedure factory otherwise false.
     *
     * @return true if matrix has procedure factory otherwise false.
     */
    public boolean hasProcedureFactory() {
        return procedureFactory != null;
    }

    /**
     * Synchronizes this and other MMatrix procedure factories.
     *
     * @param other other MMatrix
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
     * Makes current MMatrix data equal to other MMatrix data.
     *
     * @param other other matrix to be copied as data of this matrix.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void setEqualTo(MMatrix other) throws MatrixException {
        if (!hasEqualSize(other)) {
            throw new MatrixException("Incompatible target matrix size: " + other.size());
        }
        for (Integer index : other.keySet()) put(index, other.get(index));
    }

    /**
     * Checks if data of other MMatrix is equal to data of this MMatrix
     *
     * @param other matrix to be compared.
     * @return true is data of this and other matrix are equal otherwise false.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public boolean equals(MMatrix other) throws MatrixException {
        if (!hasEqualSize(other)) {
            throw new MatrixException("Incompatible target matrix size: " + other.size());
        }
        for (Integer index : other.keySet()) if (get(index) != other.get(index)) return false;
        return true;
    }

    /**
     * Applies unaryFunction to this MMatrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if MMatrix is masked.<br>
     *
     * @param unaryFunction unaryFunction to be applied.
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void apply(MMatrix result, UnaryFunction unaryFunction) throws MatrixException {
        if (!hasProcedureFactory()) for (Integer index : matrices.keySet()) result.put(index, get(index).apply(unaryFunction));
        else {
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (Integer index : matrices.keySet()) result.put(index, get(index).apply(unaryFunction));
            procedureFactory.createUnaryFunctionExpression(expressionLock, this, result, unaryFunction);
        }
    }

    /**
     * Applies unaryFunction to this MMatrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if MMatrix is masked.<br>
     *
     * @param unaryFunction unaryFunction to be applied.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public MMatrix apply(UnaryFunction unaryFunction) throws MatrixException {
        MMatrix result = new MMatrix();
        apply(result, unaryFunction);
        return result;
    }

    /**
     * Applies binaryFunction to this MMatrix.<br>
     * Example of operation can be subtraction of other MMatrix from this MMatrix.<br>
     * Applies masking if MMatrix is masked.<br>
     *
     * @param other other matrix
     * @param result result matrix.
     * @param binaryFunction binaryFunction to be applied.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void applyBi(MMatrix other, MMatrix result, BinaryFunction binaryFunction) throws MatrixException {
        if (!hasProcedureFactory()) for (Integer index : matrices.keySet()) result.put(index, get(index).applyBi(other.get(index), binaryFunction));
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (Integer index : matrices.keySet()) result.put(index, get(index).applyBi(other.get(index), binaryFunction));
            procedureFactory.createBinaryFunctionExpression(expressionLock, this, other, result, binaryFunction);
        }
    }

    /**
     * Applies binaryFunction to this MMatrix.<br>
     * Example of operation can be subtraction of other MMatrix from this MMatrix.<br>
     * Applies masking if MMatrix is masked.<br>
     *
     * @param other other matrix
     * @param binaryFunction binaryFunction to be applied.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public MMatrix applyBi(MMatrix other, BinaryFunction binaryFunction) throws MatrixException {
        MMatrix result = new MMatrix();
        applyBi(other, result, binaryFunction);
        return result;
    }

    /**
     * Applies binaryFunction to this MMatrix.<br>
     * Example of operation can be subtraction of other MMatrix from this MMatrix.<br>
     * Applies masking if MMatrix is masked.<br>
     *
     * @param other other matrix
     * @param result result matrix.
     * @param binaryFunction binaryFunction to be applied.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void applyBi(Matrix other, MMatrix result, BinaryFunction binaryFunction) throws MatrixException {
        if (!hasProcedureFactory()) for (Integer index : matrices.keySet()) result.put(index, get(index).applyBi(other, binaryFunction));
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (Integer index : matrices.keySet()) result.put(index, get(index).applyBi(other, binaryFunction));
            procedureFactory.createBinaryFunctionExpression(expressionLock, this, other, result, binaryFunction);
        }
    }

    /**
     * Applies binaryFunction to this MMatrix.<br>
     * Example of operation can be subtraction of other MMatrix from this MMatrix.<br>
     * Applies masking if MMatrix is masked.<br>
     *
     * @param other other matrix
     * @param binaryFunction binaryFunction to be applied.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public MMatrix applyBi(Matrix other, BinaryFunction binaryFunction) throws MatrixException {
        MMatrix result = new MMatrix();
        applyBi(other, result, binaryFunction);
        return result;
    }

    /**
     * Adds other MMatrix to this MMatrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void add(MMatrix other, MMatrix result) throws MatrixException {
        if (size() != other.size()) throw new MatrixException("Size of matrices are not matching.");
        if (!hasProcedureFactory()) for (Integer index : matrices.keySet()) result.put(index, get(index).add(other.get(index)));
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (Integer index : matrices.keySet()) result.put(index, get(index).add(other.get(index)));
            procedureFactory.createAddExpression(expressionLock, this, other, result);
        }
    }

    /**
     * Adds other MMatrix to this MMatrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @return result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public MMatrix add(MMatrix other) throws MatrixException {
        MMatrix result = new MMatrix();
        add(other, result);
        return result;
    }

    /**
     * Adds other MMatrix to this MMatrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void add(Matrix other, MMatrix result) throws MatrixException {
        if (!hasProcedureFactory()) for (Integer index : matrices.keySet()) result.put(index, get(index).add(other));
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (Integer index : matrices.keySet()) result.put(index, get(index).add(other));
            procedureFactory.createAddExpression(expressionLock, this, other, result);
        }
    }

    /**
     * Adds other MMatrix to this MMatrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @return result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public MMatrix add(Matrix other) throws MatrixException {
        MMatrix result = new MMatrix();
        add(other, result);
        return result;
    }

    /**
     * Subtracts other MMatrix from this MMatrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void subtract(MMatrix other, MMatrix result) throws MatrixException {
        if (size() != other.size()) throw new MatrixException("Size of matrices are not matching.");
        if (!hasProcedureFactory()) for (Integer index : matrices.keySet()) result.put(index, get(index).subtract(other.get(index)));
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (Integer index : matrices.keySet()) result.put(index, get(index).subtract(other.get(index)));
            procedureFactory.createSubtractExpression(expressionLock, this, other, result);
        }
    }

    /**
     * Subtracts other MMatrix from this MMatrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @return result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public MMatrix subtract(MMatrix other) throws MatrixException {
        MMatrix result = new MMatrix();
        subtract(other, result);
        return result;
    }

    /**
     * Subtracts other MMatrix from this MMatrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void subtract(Matrix other, MMatrix result) throws MatrixException {
        if (!hasProcedureFactory()) for (Integer index : matrices.keySet()) result.put(index, get(index).subtract(other));
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (Integer index : matrices.keySet()) result.put(index, get(index).subtract(other));
            procedureFactory.createSubtractExpression(expressionLock, this, other, result);
        }
    }

    /**
     * Subtracts other MMatrix from this MMatrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @return result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public MMatrix subtract(Matrix other) throws MatrixException {
        MMatrix result = new MMatrix();
        subtract(other, result);
        return result;
    }

    /**
     * Multiplies other MMatrix with this MMatrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void multiply(MMatrix other, MMatrix result) throws MatrixException {
        if (size() != other.size()) throw new MatrixException("Size of matrices are not matching.");
        if (!hasProcedureFactory()) for (Integer index : matrices.keySet()) result.put(index, get(index).multiply(other.get(index)));
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (Integer index : matrices.keySet()) result.put(index, get(index).multiply(other.get(index)));
            procedureFactory.createMultiplyExpression(expressionLock, this, other, result);
        }
    }

    /**
     * Multiplies other MMatrix with this MMatrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @return result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public MMatrix multiply(MMatrix other) throws MatrixException {
        MMatrix result = new MMatrix();
        multiply(other, result);
        return result;
    }

    /**
     * Multiplies other MMatrix with this MMatrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void multiply(Matrix other, MMatrix result) throws MatrixException {
        if (!hasProcedureFactory()) for (Integer index : matrices.keySet()) result.put(index, get(index).multiply(other));
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (Integer index : matrices.keySet()) result.put(index, get(index).multiply(other));
            procedureFactory.createMultiplyExpression(expressionLock, this, other, result);
        }
    }

    /**
     * Multiplies other MMatrix with this MMatrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @return result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public MMatrix multiply(Matrix other) throws MatrixException {
        MMatrix result = new MMatrix();
        multiply(other, result);
        return result;
    }

    /**
     * Dots other MMatrix with this MMatrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void dot(MMatrix other, MMatrix result) throws MatrixException {
        if (size() != other.size()) throw new MatrixException("Size of matrices are not matching.");
        if (!hasProcedureFactory()) for (Integer index : matrices.keySet()) result.put(index, get(index).dot(other.get(index)));
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (Integer index : matrices.keySet()) result.put(index, get(index).dot(other.get(index)));
            procedureFactory.createDotExpression(expressionLock, this, other, result);
        }
    }

    /**
     * Dots other MMatrix with this MMatrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @return result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public MMatrix dot(MMatrix other) throws MatrixException {
        MMatrix result = new MMatrix();
        dot(other, result);
        return result;
    }

    /**
     * Dots other MMatrix with this MMatrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void dot(Matrix other, MMatrix result) throws MatrixException {
        if (!hasProcedureFactory()) for (Integer index : matrices.keySet()) result.put(index, get(index).dot(other));
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (Integer index : matrices.keySet()) result.put(index, get(index).dot(other));
            procedureFactory.createDotExpression(expressionLock, this, other, result);
        }
    }

    /**
     * Dots other MMatrix with this MMatrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @return result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public MMatrix dot(Matrix other) throws MatrixException {
        MMatrix result = new MMatrix();
        dot(other, result);
        return result;
    }

    /**
     * Divides this MMatrix with other MMatrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void divide(MMatrix other, MMatrix result) throws MatrixException {
        if (size() != other.size()) throw new MatrixException("Size of matrices are not matching.");
        if (!hasProcedureFactory()) for (Integer index : matrices.keySet()) result.put(index, get(index).divide(other.get(index)));
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (Integer index : matrices.keySet()) result.put(index, get(index).divide(other.get(index)));
            procedureFactory.createDivideExpression(expressionLock, this, other, result);
        }
    }

    /**
     * Divides this MMatrix with other MMatrix.
     *
     * @param other other matrix.
     * @return result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public MMatrix divide(MMatrix other) throws MatrixException {
        MMatrix result = new MMatrix();
        divide(other, result);
        return result;
    }

    /**
     * Divides this MMatrix with other MMatrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void divide(Matrix other, MMatrix result) throws MatrixException {
        if (!hasProcedureFactory()) for (Integer index : matrices.keySet()) result.put(index, get(index).divide(other));
        else {
            synchronizeProcedureFactory(other);
            result.setProcedureFactory(procedureFactory);
            double expressionLock = procedureFactory.startExpression(this);
            for (Integer index : matrices.keySet()) result.put(index, get(index).divide(other));
            procedureFactory.createDivideExpression(expressionLock, this, other, result);
        }
    }

    /**
     * Divides this MMatrix with other MMatrix.
     *
     * @param other other matrix.
     * @return result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public MMatrix divide(Matrix other) throws MatrixException {
        MMatrix result = new MMatrix();
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
        for (Integer index : keySet()) result.add(get(index), result);
        return asMean ? result.divide(size()) : result;
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
            for (Integer index : matrices.keySet()) result.add(matrices.get(index).subtract(meanMatrix).power(2), result);
            result.divide(size(), result);
        }
        else {
            double expressionLock = procedureFactory.startExpression(this);
            for (Integer index : matrices.keySet()) result.add(matrices.get(index).subtract(meanMatrix).power(2), result);
            result.divide(size(), result);
            result.setProcedureFactory(procedureFactory);
            procedureFactory.createVarianceExpression(expressionLock, this, result);
        }
        return result;
    }

    /**
     * Calculates standard deviation.
     *
     * @return resulting variance
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix standardDeviation() throws MatrixException, DynamicParamException {
        return standardDeviation(mean(), getNewMatrix());
    }

    /**
     * Calculates standard deviation.
     *
     * @param meanMatrix matrix containing mean values for variance calculation.
     * @return resulting variance
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix standardDeviation(Matrix meanMatrix) throws MatrixException, DynamicParamException {
        return standardDeviation(meanMatrix, getNewMatrix());
    }

    /**
     * Calculates standard deviation.
     *
     * @param meanMatrix matrix containing mean values for variance calculation.
     * @param result result matrix.
     * @return resulting variance
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix standardDeviation(Matrix meanMatrix, Matrix result) throws MatrixException, DynamicParamException {
        if (!hasProcedureFactory()) variance(meanMatrix).multiply(matrices.size()).divide(matrices.size() - 1).apply(result, UnaryFunctionType.SQRT);
        else {
            double expressionLock = procedureFactory.startExpression(this);
            variance(meanMatrix).multiply(matrices.size()).divide(matrices.size() - 1).apply(result, UnaryFunctionType.SQRT);
            result.setProcedureFactory(procedureFactory);
            procedureFactory.createStandardDeviationExpression(expressionLock, this, result);
        }
        return result;
    }

    /**
     * Calculates softmax.
     *
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void softmax(MMatrix result) throws MatrixException {
        for (Integer index : matrices.keySet()) result.put(index, get(index).softmax());
    }

    /**
     * Calculates softmax.
     *
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public MMatrix softmax() throws MatrixException {
        MMatrix result = new MMatrix();
        softmax(result);
        return result;
    }

    /**
     * Calculates Gumbel softmax.
     *
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void gumbelSoftmax(MMatrix result) throws MatrixException {
        gumbelSoftmax(result, 1);
    }

    /**
     * Calculates Gumbel softmax.
     *
     * @param gumbelSoftmaxTau tau value for Gumbel Softmax.
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void gumbelSoftmax(MMatrix result, double gumbelSoftmaxTau) throws MatrixException {
        for (Integer index : matrices.keySet()) result.put(index, get(index).gumbelSoftmax(gumbelSoftmaxTau));
    }

    /**
     * Calculates Gumbel softmax.
     *
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public MMatrix gumbelSoftmax() throws MatrixException {
        MMatrix result = new MMatrix();
        gumbelSoftmax(result, 1);
        return result;
    }

    /**
     * Calculates Gumbel softmax.
     *
     * @param gumbelSoftmaxTau tau value for Gumbel Softmax.
     * @return matrix which stores operation result.
     * @throws MatrixException not thrown in any situation.
     */
    public MMatrix gumbelSoftmax(double gumbelSoftmaxTau) throws MatrixException {
        MMatrix result = new MMatrix();
        gumbelSoftmax(result, gumbelSoftmaxTau);
        return result;
    }

    /**
     * Splits matrix at defined position. If splitVertical is true splits vertically otherwise horizontally.
     *
     * @param position position of split
     * @param splitVertically if true splits vertically otherwise horizontally.
     * @return splitted matrix as JMatrix.
     * @throws MatrixException throws matrix exception if splitting fails.
     */
    public MMatrix split(int position, boolean splitVertically) throws MatrixException {
        MMatrix splitMMatrix = new MMatrix(getDepth());
        for (Integer index : matrices.keySet()) {
            splitMMatrix.put(index, matrices.get(index).split(position, splitVertically));
        }
        return splitMMatrix;
    }

    /**
     * Flattens MMatrix into one dimensional column vector (matrix)
     *
     * @return flattened MMatrix
     * @throws MatrixException throws exception if creation of MMatrix fails.
     */
    public MMatrix flatten() throws MatrixException {
        int rows = this.rows;
        int cols = this.columns;
        Matrix matrix = new DMatrix(rows * cols * size(), 1);
        MMatrix mmatrix = new MMatrix(1, matrix);
        for (Integer index : keySet()) {
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    matrix.setValue(getPosition(rows, cols, row, col, index), 0 , get(index).getValue(row, col));
                }
            }
        }
        return mmatrix;
    }

    /**
     * Returns unflattened MMatrix i.e. samples that have been unflattened from single column vector.
     *
     * @param width width of unflattened MMatrix.
     * @param height height of unflattened MMatrix.
     * @param depth depth of unflattened MMatrix.
     * @return unflattened MMatrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public MMatrix unflatten(int width, int height, int depth) throws MatrixException {
        if (size() != 1) throw new MatrixException("MMatrix cannot be unflattened since it is not column vector (size equals 1).");
        MMatrix mmatrix = new MMatrix(depth);
        for (int index = 0; index < depth; index++) {
            Matrix matrix = new DMatrix(width, height);
            mmatrix.put(index, matrix);
            for (int row = 0; row < width; row++) {
                for (int col = 0; col < height; col++) {
                    matrix.setValue(row, col, get(0).getValue(getPosition(width, height, row, col, index), 0));
                }
            }
        }
        return mmatrix;
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
     * Joins this MMatrix with other MMatrix.
     *
     * @param otherMMatrix other MMatrix.
     * @param joinedVertically if true MMatrices are joint vertically otherwise horizontally.
     * @return joined MMatrices.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public MMatrix join(MMatrix otherMMatrix, boolean joinedVertically) throws MatrixException {
        if (getDepth() != otherMMatrix.getDepth()) throw new MatrixException("Depth of this MMatrix " + getDepth() + " and other MMatrix " + otherMMatrix.getDepth() + " do not match.");
        MMatrix joinedMMatrix = new MMatrix(getDepth());
        for (Integer entryIndex : keySet()) {
            if (!otherMMatrix.containsKey(entryIndex)) throw new MatrixException("Other MMatrix does not contain entry index: " + entryIndex);
            joinedMMatrix.put(entryIndex, new JMatrix(new Matrix[] { get(entryIndex), otherMMatrix.get(entryIndex) }, joinedVertically));
        }
        return joinedMMatrix;
    }

    /**
     * Unjoins joined matrix by returning matrix corresponding given sub matrix index.
     *
     * @param subMatrixIndex sub matrix index.
     * @return unjoined matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public MMatrix unjoin(int subMatrixIndex) throws MatrixException {
        MMatrix unjoinedMMatrix = new MMatrix(getDepth());
        for (Integer entryIndex : keySet()) {
            ArrayList<Matrix> subMatrices = get(entryIndex).getSubMatrices();
            if (subMatrixIndex < 0 || subMatrixIndex > subMatrices.size() - 1) throw new MatrixException("Joined matrix does not have sub matrix index: " + subMatrixIndex);
            unjoinedMMatrix.put(entryIndex, subMatrices.get(subMatrixIndex));
        }
        return unjoinedMMatrix;
    }

}
