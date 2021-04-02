/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils.matrix;

import utils.DynamicParamException;
import utils.procedure.ProcedureFactory;

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

    private static final long serialVersionUID = 2208329722377770337L;

    /**
     * Max size for MMatrix.
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
     * Name for matrix.
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
     * Constructor for sample with capacity limitation.
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
     * Constructor for sample with capacity limitation.
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
     * Returns sample entries.
     *
     * @return sample entries.
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
     * Returns last key of node.
     *
     * @return last key of node.
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
     * @return last key of matrices.
     */
    public Matrix lastValue() {
        return matrices.get(matrices.lastKey());
    }

    /**
     * Returns entries of sample as collection.
     *
     * @return entries of sample as collection.
     */
    public Collection<Matrix> values() {
        return matrices.values();
    }

    /**
     * Checks if sample contains specific entry.
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
     * Creates new matrix with object reference to the matrix data of this matrix.
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
     * Creates new matrix with object full copy of this matrix.
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
     * Checks if this matrix and other matrix are equal in dimensions (rows x columns).
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
     * Synchronizes this and other matrix procedure factories.
     *
     * @param other other matrix
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
     * Makes current matrix data equal to other matrix data.
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
     * Checks if data of other matrix is equal to data of this matrix
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
     * Applies unaryFunction to this matrix.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param unaryFunction unaryFunction to be applied.
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void apply(MMatrix result, UnaryFunction unaryFunction) throws MatrixException {
        double expressionLock = 0;
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        for (Integer index : matrices.keySet()) result.put(index, get(index).apply(unaryFunction));
        if (procedureFactory != null) procedureFactory.createUnaryFunctionExpression(expressionLock, this, result, unaryFunction);
    }

    /**
     * Applies unaryFunction to this matrix and return operation result.<br>
     * Example of operation can be applying square root operation to this matrix.<br>
     * Applies masking if matrix is masked.<br>
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
     * Applies binaryFunction to this matrix.<br>
     * Example of operation can be applying power operation to this and other matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param other other matrix
     * @param result result matrix.
     * @param binaryFunction binaryFunction to be applied.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void applyBi(MMatrix other, MMatrix result, BinaryFunction binaryFunction) throws MatrixException {
        double expressionLock = 0;
        synchronizeProcedureFactory(other);
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        for (Integer index : matrices.keySet()) result.put(index, get(index).applyBi(other.get(index), binaryFunction));
        if (procedureFactory != null) procedureFactory.createBinaryFunctionExpression(expressionLock, this, other, result, binaryFunction);
    }

    /**
     * Applies binaryFunction to this matrix and return operation result.<br>
     * Example of operation can be applying power operation to this and other matrix.<br>
     * Applies masking if matrix is masked.<br>
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
     * Applies binaryFunction to this matrix.<br>
     * Example of operation can be applying power operation to this and other matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param other other matrix
     * @param result result matrix.
     * @param binaryFunction binaryFunction to be applied.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void applyBi(Matrix other, MMatrix result, BinaryFunction binaryFunction) throws MatrixException {
        double expressionLock = 0;
        synchronizeProcedureFactory(other);
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        for (Integer index : matrices.keySet()) result.put(index, get(index).applyBi(other, binaryFunction));
        if (procedureFactory != null) procedureFactory.createBinaryFunctionExpression(expressionLock, this, other, result, binaryFunction);
    }

    /**
     * Applies binaryFunction to this matrix and return operation result.<br>
     * Example of operation can be applying power operation to this and other matrix.<br>
     * Applies masking if matrix is masked.<br>
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
     * Adds other matrix to this matrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void add(MMatrix other, MMatrix result) throws MatrixException {
        if (size() != other.size()) throw new MatrixException("Size of matrices are not matching.");
        double expressionLock = 0;
        synchronizeProcedureFactory(other);
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        for (Integer index : matrices.keySet()) result.put(index, get(index).add(other.get(index)));
        if (procedureFactory != null) procedureFactory.createAddExpression(expressionLock, this, other, result);
    }

    /**
     * Adds other matrix to this matrix.
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
     * Adds other matrix to this matrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void add(Matrix other, MMatrix result) throws MatrixException {
        double expressionLock = 0;
        synchronizeProcedureFactory(other);
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        for (Integer index : matrices.keySet()) result.put(index, get(index).add(other));
        if (procedureFactory != null) procedureFactory.createAddExpression(expressionLock, this, other, result);
    }

    /**
     * Adds other matrix to this matrix.
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
     * Subtracts other matrix from this matrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void subtract(MMatrix other, MMatrix result) throws MatrixException {
        if (size() != other.size()) throw new MatrixException("Size of matrices are not matching.");
        double expressionLock = 0;
        synchronizeProcedureFactory(other);
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        for (Integer index : matrices.keySet()) result.put(index, get(index).subtract(other.get(index)));
        if (procedureFactory != null) procedureFactory.createSubtractExpression(expressionLock, this, other, result);
    }

    /**
     * Subtracts other matrix from this matrix.
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
     * Subtracts other matrix to this matrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void subtract(Matrix other, MMatrix result) throws MatrixException {
        double expressionLock = 0;
        synchronizeProcedureFactory(other);
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        for (Integer index : matrices.keySet()) result.put(index, get(index).subtract(other));
        if (procedureFactory != null) procedureFactory.createSubtractExpression(expressionLock, this, other, result);
    }

    /**
     * Subtracts other matrix from this matrix.
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
     * Multiplies other matrix with this matrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void multiply(MMatrix other, MMatrix result) throws MatrixException {
        if (size() != other.size()) throw new MatrixException("Size of matrices are not matching.");
        double expressionLock = 0;
        synchronizeProcedureFactory(other);
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        for (Integer index : matrices.keySet()) result.put(index, get(index).multiply(other.get(index)));
        if (procedureFactory != null) procedureFactory.createMultiplyExpression(expressionLock, this, other, result);
    }

    /**
     * Multiplies other matrix with this matrix.
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
     * Multiplies other matrix with this matrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void multiply(Matrix other, MMatrix result) throws MatrixException {
        double expressionLock = 0;
        synchronizeProcedureFactory(other);
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        for (Integer index : matrices.keySet()) result.put(index, get(index).multiply(other));
        if (procedureFactory != null) procedureFactory.createMultiplyExpression(expressionLock, this, other, result);
    }

    /**
     * Multiplies other matrix with this matrix.
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
     * Dots other matrix with this matrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void dot(MMatrix other, MMatrix result) throws MatrixException {
        if (size() != other.size()) throw new MatrixException("Size of matrices are not matching.");
        double expressionLock = 0;
        synchronizeProcedureFactory(other);
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        for (Integer index : matrices.keySet()) result.put(index, get(index).dot(other.get(index)));
        if (procedureFactory != null) procedureFactory.createDotExpression(expressionLock, this, other, result);
    }

    /**
     * Dots other matrix with this matrix.
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
     * Dots other matrix with this matrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void dot(Matrix other, MMatrix result) throws MatrixException {
        double expressionLock = 0;
        synchronizeProcedureFactory(other);
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        for (Integer index : matrices.keySet()) result.put(index, get(index).dot(other));
        if (procedureFactory != null) procedureFactory.createDotExpression(expressionLock, this, other, result);
    }

    /**
     * Dots other matrix with this matrix.
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
     * Divides this matrix with other matrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void divide(MMatrix other, MMatrix result) throws MatrixException {
        if (size() != other.size()) throw new MatrixException("Size of matrices are not matching.");
        double expressionLock = 0;
        synchronizeProcedureFactory(other);
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        for (Integer index : matrices.keySet()) result.put(index, get(index).divide(other.get(index)));
        if (procedureFactory != null) procedureFactory.createDivideExpression(expressionLock, this, other, result);
    }

    /**
     * Divides this matrix with other matrix.
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
     * Divides this matrix with other matrix.
     *
     * @param other matrix which acts as second variable in the operation.
     * @param result matrix which stores operation result.
     * @throws MatrixException throws MatrixException if this and other matrix are not of equal dimensions.
     */
    public void divide(Matrix other, MMatrix result) throws MatrixException {
        double expressionLock = 0;
        synchronizeProcedureFactory(other);
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        for (Integer index : matrices.keySet()) result.put(index, get(index).divide(other));
        if (procedureFactory != null) procedureFactory.createDivideExpression(expressionLock, this, other, result);
    }

    /**
     * Divides this matrix with other matrix.
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
     * @return result of counting
     * @throws MatrixException throws exception if row or column vectors are incorrectly provided.
     */
    public Matrix count(boolean asMean) throws MatrixException {
        Matrix sumMatrix = new DMatrix(get(firstKey()).getRows(), get(firstKey()).getColumns());
        for (Integer index : keySet()) sumMatrix.add(get(index), sumMatrix);
        return asMean ? sumMatrix.divide(size()) : sumMatrix;
    }

    /**
     * Calculates sum.
     *
     * @return resulting sum
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     */
    public Matrix sum() throws MatrixException {
        double expressionLock = 0;
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        Matrix result = count(false);
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) procedureFactory.createSumExpression(expressionLock, this, result);
        return result;
    }

    /**
     * Calculates mean.
     *
     * @return resulting mean
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     */
    public Matrix mean() throws MatrixException {
        double expressionLock = 0;
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        Matrix result = count(true);
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) procedureFactory.createMeanExpression(expressionLock, this, result);
        return result;
    }

    /**
     * Calculates variance.
     *
     * @return resulting variance
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix variance() throws MatrixException, DynamicParamException {
        double expressionLock = 0;
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        Matrix result = variance(mean());
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) procedureFactory.createVarianceExpression(expressionLock, this, result);
        return result;
    }

    /**
     * Calculates variance over multiple matrices.
     *
     * @param meanMatrix matrix containing mean values for variance calculation.
     * @return resulting variance
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix variance(Matrix meanMatrix) throws MatrixException, DynamicParamException {
        if (meanMatrix == null) throw new MatrixException("Mean matrix is not defined");
        double expressionLock = 0;
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        Matrix varianceMatrix = new DMatrix(get(firstKey()).getRows(), get(firstKey()).getColumns());
        for (Integer index : matrices.keySet()) varianceMatrix.add(matrices.get(index).subtract(meanMatrix).power(2), varianceMatrix);
        Matrix result = varianceMatrix.divide(size());
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) procedureFactory.createVarianceExpression(expressionLock, this, result);
        return result;
    }

    /**
     * Calculates standard deviation over multiple matrices.
     *
     * @param meanMatrix matrix containing mean values for variance calculation.
     * @return resulting variance
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix standardDeviation(Matrix meanMatrix) throws MatrixException, DynamicParamException {
        double expressionLock = 0;
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        Matrix result = variance(meanMatrix).multiply(matrices.size()).divide(matrices.size() - 1).apply(UnaryFunctionType.SQRT);
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) procedureFactory.createStandardDeviationExpression(expressionLock, this, result);
        return result;
    }

    /**
     * Calculates standard deviation over multiple matrices.
     *
     * @return resulting variance
     * @throws MatrixException throws exception if matrices are incorrectly provided.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public Matrix standardDeviation() throws MatrixException, DynamicParamException {
        double expressionLock = 0;
        if (procedureFactory != null) expressionLock = procedureFactory.startExpression(this);
        Matrix result = variance().multiply(matrices.size()).divide(matrices.size() - 1).apply(UnaryFunctionType.SQRT);
        result.setProcedureFactory(procedureFactory);
        if (procedureFactory != null) procedureFactory.createStandardDeviationExpression(expressionLock, this, result);
        return result;
    }

    /**
     * Applies softmax to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void softmax(MMatrix result) throws MatrixException {
        for (Integer index : matrices.keySet()) result.put(index, get(index).softmax());
    }

    /**
     * Applies softmax to this matrix.<br>
     * Applies masking if matrix is masked.<br>
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
     * Applies Gumbel softmax to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void gumbelSoftmax(MMatrix result) throws MatrixException {
        gumbelSoftmax(result, 1);
    }

    /**
     * Applies Gumbel softmax to this matrix.<br>
     * Applies masking if matrix is masked.<br>
     *
     * @param gumbelSoftmaxTau tau value for Gumbel Softmax.
     * @param result result matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void gumbelSoftmax(MMatrix result, double gumbelSoftmaxTau) throws MatrixException {
        for (Integer index : matrices.keySet()) result.put(index, get(index).gumbelSoftmax(gumbelSoftmaxTau));
    }

    /**
     * Applies Gumbel softmax to this matrix.<br>
     * Applies masking if matrix is masked.<br>
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
     * Applies Gumbel softmax to this matrix.<br>
     * Applies masking if matrix is masked.<br>
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
