/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package utils.sampling;

import utils.matrix.AbstractMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;
import java.util.*;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Implements sequence for samples.<br>
 *
 */
public class Sequence implements Serializable {

    @Serial
    private static final long serialVersionUID = 4183245025751674913L;

    /**
     * Ordered map of samples.
     *
     */
    private final TreeMap<Integer, Matrix> samples = new TreeMap<>();

    /**
     * Re-entrant read write lock for sequence access.
     *
     */
    private final ReentrantReadWriteLock accessLock = new ReentrantReadWriteLock();

    /**
     * Constructor for sequence.
     *
     */
    public Sequence() {
    }

    /**
     * Constructor for sequence.
     *
     * @param newSamples samples to be added into this sequence.
     */
    public Sequence(HashMap<Integer, Matrix> newSamples) {
        accessLock.writeLock().lock();
        try {
            samples.putAll(newSamples);
        }
        finally {
            accessLock.writeLock().unlock();
        }
    }


    /**
     * Constructor for sequence.
     *
     * @param matrix matrix.
     */
    public Sequence(Matrix matrix) {
        this(new HashMap<>() {{ put(0, matrix); }});
    }

    /**
     * Resets sequence.
     *
     */
    public void reset() {
        accessLock.writeLock().lock();
        try {
            samples.clear();
        }
        finally {
            accessLock.writeLock().unlock();
        }
    }

    /**
     * Checks if sequence is empty.
     *
     * @return returns true if sequence is empty otherwise returns false.
     */
    public boolean isEmpty() {
        accessLock.readLock().lock();
        try {
            return samples.isEmpty();
        }
        finally {
            accessLock.readLock().unlock();
        }
    }

    /**
     * Returns number of samples in sequence.
     *
     * @return number of samples in sequence.
     */
    public int sampleSize() {
        accessLock.readLock().lock();
        try {
            return samples.size();
        }
        finally {
            accessLock.readLock().unlock();
        }
    }

    /**
     * Returns total size (amount of sample * depth of samples) of sequence.
     *
     * @return total size of sequence.
     */
    public int totalSize() {
        accessLock.readLock().lock();
        try {
            return samples.size();
        }
        finally {
            accessLock.readLock().unlock();
        }
    }

    /**
     * Puts sample into specific sample index.
     *
     * @param sampleIndex sample index.
     * @param sample sample to be inserted.
     */
    public void put(int sampleIndex, Matrix sample) {
        accessLock.writeLock().lock();
        try {
            samples.put(sampleIndex, sample);
        }
        finally {
            accessLock.writeLock().unlock();
        }
    }

    /**
     * Puts all samples into sequence.
     *
     * @param sequence sequence containing new samples for this sequence.
     */
    public void putAll(Sequence sequence) {
        accessLock.writeLock().lock();
        try {
            samples.putAll(sequence.get());
        }
        finally {
            accessLock.writeLock().unlock();
        }
    }

    /**
     * Returns sample at specific sample index.
     *
     * @param sampleIndex sample index.
     * @return requested sample.
     */
    public Matrix get(int sampleIndex) {
        accessLock.readLock().lock();
        try {
            return samples.get(sampleIndex);
        }
        finally {
            accessLock.readLock().unlock();
        }
    }

    /**
     * Returns all samples inside sequence as ordered map.
     *
     * @return all samples inside sequence as ordered map.
     */
    public TreeMap<Integer, Matrix> get() {
        accessLock.readLock().lock();
        try {
            return samples;
        }
        finally {
            accessLock.readLock().unlock();
        }
    }

    /**
     * Returns sample values.
     *
     * @return sample values.
     */
    public Collection<Matrix> values() {
        accessLock.readLock().lock();
        try {
            return samples.values();
        }
        finally {
            accessLock.readLock().unlock();
        }
    }

    /**
     * Returns sample index key set.
     *
     * @return sample index key set.
     */
    public Set<Integer> keySet() {
        accessLock.readLock().lock();
        try {
            return samples.keySet();
        }
        finally {
            accessLock.readLock().unlock();
        }
    }

    /**
     * Returns sample index entry set.
     *
     * @return sample index entry set.
     */
    public Set<Map.Entry<Integer, Matrix>> entrySet() {
        accessLock.readLock().lock();
        try {
            return samples.entrySet();
        }
        finally {
            accessLock.readLock().unlock();
        }
    }

    /**
     * Returns descending sample index entry set.
     *
     * @return descending sample index entry set.
     */
    public Set<Map.Entry<Integer, Matrix>> descendingEntrySet() {
        accessLock.readLock().lock();
        try {
            return samples.descendingMap().entrySet();
        }
        finally {
            accessLock.readLock().unlock();
        }
    }

    /**
     * Returns sample index key set in descending order.
     *
     * @return sample index key set in descending order.
     */
    public Set<Integer> descendingKeySet() {
        accessLock.readLock().lock();
        try {
            return samples.descendingKeySet();
        }
        finally {
            accessLock.readLock().unlock();
        }
    }

    /**
     * Returns first index of sequence.
     *
     * @return first index of sequence.
     */
    public Integer firstKey() {
        accessLock.readLock().lock();
        try {
            return samples.firstKey();
        }
        finally {
            accessLock.readLock().unlock();
        }
    }

    /**
     * Returns last index of sequence.
     *
     * @return last index of sequence.
     */
    public Integer lastKey() {
        accessLock.readLock().lock();
        try {
            return samples.lastKey();
        }
        finally {
            accessLock.readLock().unlock();
        }
    }

    /**
     * Sequences together by sample indices.
     *
     * @param sequences sequences to be joined.
     * @param joinedVertically if true sequences are joined together vertically otherwise horizontally.
     * @return joined sequence.
     * @throws MatrixException throws exception if joining of matrices fails.
     */
    public static TreeMap<Integer, Sequence> join(TreeMap<Integer, Sequence> sequences, boolean joinedVertically) throws MatrixException {
        Sequence joinedSequence = new Sequence();
        for (Integer sampleIndex : sequences.get(0).keySet()) {
            Matrix[] matrices = new Matrix[sequences.size()];
            for (Map.Entry<Integer, Sequence> entry : sequences.entrySet()) {
                matrices[entry.getKey()] = entry.getValue().get(sampleIndex);
            }
            joinedSequence.put(sampleIndex, Objects.requireNonNull(AbstractMatrix.join(matrices, joinedVertically)));
        }
        return new TreeMap<>() {{ put(0, joinedSequence); }};
    }

    /**
     * Increments sequence with other sequence.
     *
     * @param sequence other sequence.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void increment(Sequence sequence) throws MatrixException {
        for (Map.Entry<Integer, Matrix> entry : sequence.entrySet()) {
            increment(entry.getKey(), entry.getValue());
        }
    }

    /**
     * Increments other matrix to specific matrix.
     *
     * @param sampleIndex sample index.
     * @param matrix other matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void increment(int sampleIndex, Matrix matrix) throws MatrixException {
        accessLock.writeLock().lock();
        try {
            Matrix currentMatrix = get(sampleIndex);
            if (currentMatrix != null) currentMatrix.addBy(matrix);
            else put(sampleIndex, matrix);
        }
        finally {
            accessLock.writeLock().unlock();
        }
    }

    /**
     * Creates sequences out of matrices.
     *
     * @param matrices matrices
     * @return sequences
     */
    public static TreeMap<Integer, Sequence> getSequencesFromMatrices(TreeMap<Integer, Matrix> matrices) {
        TreeMap<Integer, Sequence> sequences = new TreeMap<>();
        for (Map.Entry<Integer, Matrix> entry : matrices.entrySet()) sequences.put(entry.getKey(), new Sequence(entry.getValue()));
        return sequences;
    }

}
