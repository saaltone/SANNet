/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package utils.matrix;

import utils.configurable.DynamicParamException;
import utils.procedure.ProcedureFactory;

import java.io.Serial;
import java.io.Serializable;
import java.util.Collection;
import java.util.Set;
import java.util.TreeMap;

/**
 * Defines multi-matrix class that can execute matrix operations with multiple matrices such as adding multiple matrices together element by element.<br>
 * Class also has operations to calculate sum, mean, variance and standard deviation over multiple matrices element by element.<br>
 *
 */
public class MMatrix implements Cloneable, Serializable {

    @Serial
    private static final long serialVersionUID = 2208329722377770337L;

    /**
     * Capacity for MMatrix.
     *
     */
    private final int capacity;

    /**
     * Map for matrices.
     *
     */
    private TreeMap<Integer, Matrix> matrices;

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
     * Constructor for MMatrix without capacity limitation.
     *
     */
    public MMatrix() {
        matrices = new TreeMap<>();
        capacity = -1;
    }

    /**
     * Constructor for sample with capacity assumption of 1.
     *
     * @param entry single entry for sample with assumption of maxSize 1.
     */
    public MMatrix(Matrix entry) {
        matrices = new TreeMap<>();
        capacity = 1;
        matrices.put(0, entry);
    }

    /**
     * Constructor with capacity limitation.
     *
     * @param capacity capacity of MMatrix.
     * @throws MatrixException throws exception if depth is less than 1.
     */
    public MMatrix(int capacity) throws MatrixException {
        if (capacity < 1) throw new MatrixException("Capacity must be at least 1.");
        matrices = new TreeMap<>();
        this.capacity = capacity;
    }

    /**
     * Constructor with capacity limitation.
     *
     * @param capacity capacity of MMatrix.
     * @param name name of matrix.
     * @throws MatrixException throws exception if depth is less than 1.
     */
    public MMatrix(int capacity, String name) throws MatrixException {
        if (capacity < 1) throw new MatrixException("Capacity must be at least 1.");
        matrices = new TreeMap<>();
        this.capacity = capacity;
        this.name = name;
    }

    /**
     * Constructor for sample with capacity limitation.
     *
     * @param capacity capacity of MMatrix.
     * @param entry single entry for sample.
     * @throws MatrixException throws exception if depth is less than 1.
     */
    public MMatrix(int capacity, Matrix entry) throws MatrixException {
        this(capacity);
        matrices.put(0, entry);
    }

    /**
     * Constructor for MMatrix without capacity limitation.
     *
     * @param matrices matrices for MMatrix.
     */
    public MMatrix(TreeMap<Integer, Matrix> matrices) {
        capacity = -1;
        this.matrices = matrices;
    }

    /**
     * Constructor for MMatrix without capacity limitation.
     *
     * @param matrices matrices for MMatrix.
     */
    public MMatrix(MMatrix matrices) {
        capacity = -1;
        this.matrices = matrices.get();
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
        matrices = new TreeMap<>();
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
    public TreeMap<Integer, Matrix> get() {
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
     * @throws MatrixException throws exception if matrix is exceeding its capacity.
     */
    public void put(int index, Matrix matrix) throws MatrixException {
        if (!matrices.containsKey(index) && capacity > 0 && matrices.size() >= capacity) throw new MatrixException("MMatrix is exceeding defined capacity.");
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
     * Returns first key of matrices.
     *
     * @return first key of matrices.
     */
    public int firstKey() {
        return matrices.firstKey();
    }

    /**
     * Returns last key of matrices.
     *
     * @return last key of matrices.
     */
    public int lastKey() {
        return matrices.lastKey();
    }

    /**
     * Returns first value of matrices.
     *
     * @return first value of matrices.
     */
    public Matrix firstValue() {
        return matrices.get(matrices.firstKey());
    }

    /**
     * Returns last value of matrices.
     *
     * @return last value of matrices.
     */
    public Matrix lastValue() {
        return matrices.get(matrices.lastKey());
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
     * Returns capacity of MMatrix.
     *
     * @return capacity of MMatrix.
     */
    public int getCapacity() {
        return capacity;
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
        return count(asMean, get(firstKey()).getNewMatrix());
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
        return sum(get(firstKey()).getNewMatrix());
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
        return mean(get(firstKey()).getNewMatrix());
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
        return variance(mean(), get(firstKey()).getNewMatrix());
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
        return variance(meanMatrix, get(firstKey()).getNewMatrix());
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
        return standardDeviation(mean(), get(firstKey()).getNewMatrix());
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
        return standardDeviation(meanMatrix, get(firstKey()).getNewMatrix());
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
     * Flattens MMatrix into one dimensional column vector (matrix)
     *
     * @return flattened MMatrix
     * @throws MatrixException throws exception if creation of MMatrix fails.
     */
    public MMatrix flatten() throws MatrixException {
        int rows = firstValue().getRows();
        int cols = firstValue().getColumns();
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

}
