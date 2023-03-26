/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.sampling;

import utils.matrix.AbstractMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;
import java.util.*;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

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
     * Access lock for sequence data.
     *
     */
    private final Lock accessLock = new ReentrantLock();

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
        for (Map.Entry<Integer, Matrix> entry : newSamples.entrySet()) {
            int key = entry.getKey();
            Matrix value = entry.getValue();
            samples.put(key, value);
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
        accessLock.lock();
        samples.clear();
        accessLock.unlock();
    }

    /**
     * Checks if sequence is empty.
     *
     * @return returns true if sequence is empty otherwise returns false.
     */
    public boolean isEmpty() {
        accessLock.lock();
        try {
            return samples.isEmpty();
        }
        finally {
            accessLock.unlock();
        }
    }

    /**
     * Returns number of samples in sequence.
     *
     * @return number of samples in sequence.
     */
    public int sampleSize() {
        accessLock.lock();
        try {
            return samples.size();
        }
        finally {
            accessLock.unlock();
        }
    }

    /**
     * Returns total size (amount of sample * depth of samples) of sequence.
     *
     * @return total size of sequence.
     */
    public int totalSize() {
        accessLock.lock();
        try {
            return samples.size();
        }
        finally {
            accessLock.unlock();
        }
    }

    /**
     * Puts sample into specific sample index.
     *
     * @param sampleIndex sample index.
     * @param sample sample to be inserted.
     */
    public void put(int sampleIndex, Matrix sample) {
        accessLock.lock();
        samples.put(sampleIndex, sample);
        accessLock.unlock();
    }

    /**
     * Puts all samples into sequence.
     *
     * @param sequence sequence containing new samples for this sequence.
     */
    public void putAll(Sequence sequence) {
        accessLock.lock();
        samples.putAll(sequence.get());
        accessLock.unlock();
    }

    /**
     * Returns sample at specific sample index.
     *
     * @param sampleIndex sample index.
     * @return requested sample.
     */
    public Matrix get(int sampleIndex) {
        accessLock.lock();
        try {
            return samples.get(sampleIndex);
        }
        finally {
            accessLock.unlock();
        }
    }

    /**
     * Returns all samples inside sequence as ordered map.
     *
     * @return all samples inside sequence as ordered map.
     */
    public TreeMap<Integer, Matrix> get() {
        accessLock.lock();
        try {
            return samples;
        }
        finally {
            accessLock.unlock();
        }
    }

    /**
     * Returns sample values.
     *
     * @return sample values.
     */
    public Collection<Matrix> values() {
        accessLock.lock();
        try {
            return samples.values();
        }
        finally {
            accessLock.unlock();
        }
    }

    /**
     * Returns sample index key set.
     *
     * @return sample index key set.
     */
    public Set<Integer> keySet() {
        accessLock.lock();
        try {
            return samples.keySet();
        }
        finally {
            accessLock.unlock();
        }
    }

    /**
     * Returns sample index entry set.
     *
     * @return sample index entry set.
     */
    public Set<Map.Entry<Integer, Matrix>> entrySet() {
        accessLock.lock();
        try {
            return samples.entrySet();
        }
        finally {
            accessLock.unlock();
        }
    }

    /**
     * Returns descending sample index entry set.
     *
     * @return descending sample index entry set.
     */
    public Set<Map.Entry<Integer, Matrix>> descendingEntrySet() {
        accessLock.lock();
        try {
            return samples.descendingMap().entrySet();
        }
        finally {
            accessLock.unlock();
        }
    }

    /**
     * Returns sample index key set in descending order.
     *
     * @return sample index key set in descending order.
     */
    public Set<Integer> descendingKeySet() {
        accessLock.lock();
        try {
            return samples.descendingKeySet();
        }
        finally {
            accessLock.unlock();
        }
    }

    /**
     * Returns first index of sequence.
     *
     * @return first index of sequence.
     */
    public Integer firstKey() {
        accessLock.lock();
        try {
            return samples.firstKey();
        }
        finally {
            accessLock.unlock();
        }
    }

    /**
     * Returns last index of sequence.
     *
     * @return last index of sequence.
     */
    public Integer lastKey() {
        accessLock.lock();
        try {
            return samples.lastKey();
        }
        finally {
            accessLock.unlock();
        }
    }

    /**
     * Returns flattened sequence i.e. samples that have been flattened into single row with depth one.
     *
     * @return flattened sequence.
     */
    public Sequence flatten() {
        accessLock.lock();
        try {
            Sequence flattenedSequence = new Sequence();
            for (Map.Entry<Integer, Matrix> entry : entrySet()) {
                flattenedSequence.put(entry.getKey(), entry.getValue().flatten());
            }
            return flattenedSequence;
        }
        finally {
            accessLock.unlock();
        }
    }

    /**
     * Returns unflattened sequence i.e. samples that have been unflattened from single row with depth one.
     *
     * @param width width of unflattened sequence.
     * @param height height of unflattened sequence.
     * @param depth depth of unflattened sequence.
     * @return unflattened sequence.
     */
    public Sequence unflatten(int width, int height, int depth) {
        accessLock.lock();
        try {
            Sequence unflattenedSequence = new Sequence();
            for (Map.Entry<Integer, Matrix> entry : entrySet()) {
                unflattenedSequence.put(entry.getKey(), entry.getValue().unflatten(width, height, depth));
            }
            return unflattenedSequence;
        }
        finally {
            accessLock.unlock();
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
        accessLock.lock();
        try {
            for (Map.Entry<Integer, Matrix> entry : sequence.entrySet()) {
                increment(entry.getKey(), entry.getValue());
            }
        }
        finally {
            accessLock.unlock();
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
        accessLock.lock();
        try {
            Matrix currentMatrix = get(sampleIndex);
            if (currentMatrix != null) currentMatrix.addBy(matrix);
            else put(sampleIndex, matrix);
        }
        finally {
            accessLock.unlock();
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
