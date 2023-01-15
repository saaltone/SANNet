/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package utils.sampling;

import utils.matrix.MMatrix;
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
     * Sample depth.
     *
     */
    private int depth;

    /**
     * Ordered map of samples.
     *
     */
    private final TreeMap<Integer, MMatrix> samples = new TreeMap<>();

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
        depth = -1;
    }

    /**
     * Constructor for sequence.
     *
     * @param newSamples samples to be added into this sequence.
     * @throws MatrixException throws exception if depth of samples are not equal.
     */
    public Sequence(HashMap<Integer, MMatrix> newSamples) throws MatrixException {
        depth = newSamples.get(0).getDepth();
        for (Map.Entry<Integer, MMatrix> entry : newSamples.entrySet()) {
            int key = entry.getKey();
            MMatrix value = entry.getValue();
            if (value.getDepth() != depth) throw new MatrixException("Depths of all samples are not equal.");
            samples.put(key, value);
        }
    }


    /**
     * Constructor for sequence.
     *
     * @param mMatrix multi-matrix.
     * @throws MatrixException throws exception if depth of samples are not equal.
     */
    public Sequence(MMatrix mMatrix) throws MatrixException {
        this(new HashMap<>() {{ put(0, mMatrix); }});
    }

    /**
     * Resets sequence.
     *
     */
    public void reset() {
        accessLock.lock();
        samples.clear();
        depth = -1;
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
     * Returns depth of sample.
     *
     * @return depth of sample.
     */
    public int getDepth() {
        accessLock.lock();
        try {
            return depth;
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
            return samples.size() * depth;
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
     * @throws MatrixException throws exception if depth of sample is not matching depth of sequence.
     */
    public void put(int sampleIndex, MMatrix sample) throws MatrixException {
        accessLock.lock();
        if (depth == -1) depth = sample.getDepth();
        else if (depth != sample.getDepth()) throw new MatrixException("Depth of sample is not matching depth of sequence: " + getDepth());
        samples.put(sampleIndex, sample);
        accessLock.unlock();
    }

    /**
     * Puts all samples into sequence.
     *
     * @param sequence sequence containing new samples for this sequence.
     * @throws MatrixException throws exception if depth of sequence is not matching depth of this sequence.
     */
    public void putAll(Sequence sequence) throws MatrixException {
        accessLock.lock();
        if (depth == -1) depth = sequence.getDepth();
        else if (depth != sequence.getDepth()) throw new MatrixException("Depth of sequence is not matching depth of this sequence: " + getDepth());
        samples.putAll(sequence.get());
        accessLock.unlock();
    }

    /**
     * Returns sample at specific sample index.
     *
     * @param sampleIndex sample index.
     * @return requested sample.
     */
    public MMatrix get(int sampleIndex) {
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
    public TreeMap<Integer, MMatrix> get() {
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
    public Collection<MMatrix> values() {
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
    public Set<Map.Entry<Integer, MMatrix>> entrySet() {
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
    public Set<Map.Entry<Integer, MMatrix>> descendingEntrySet() {
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
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Sequence flatten() throws MatrixException {
        accessLock.lock();
        try {
            Sequence flattenedSequence = new Sequence();
            for (Map.Entry<Integer, MMatrix> entry : entrySet()) {
                int sampleIndex = entry.getKey();
                MMatrix mMatrix = entry.getValue();
                flattenedSequence.put(sampleIndex, mMatrix.flatten());
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
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Sequence unflatten(int width, int height, int depth) throws MatrixException {
        accessLock.lock();
        try {
            Sequence unflattenedSequence = new Sequence();
            for (Map.Entry<Integer, MMatrix> entry : entrySet()) {
                int sampleIndex = entry.getKey();
                MMatrix mMatrix = entry.getValue();
                unflattenedSequence.put(sampleIndex, mMatrix.unflatten(width, height, depth));
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
            MMatrix[] mMatrices = new MMatrix[sequences.size()];
            for (Map.Entry<Integer, Sequence> entry : sequences.entrySet()) {
                mMatrices[entry.getKey()] = entry.getValue().get(sampleIndex);
            }
            joinedSequence.put(sampleIndex, MMatrix.join(mMatrices, joinedVertically));
        }
        return new TreeMap<>() {{ put(0, joinedSequence); }};
    }

    /**
     * Unjoins sequence
     *
     * @param sequence sequence
     * @param unjoinedSequence unjoined sequence
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public static void unjoinAsMap(Sequence sequence, TreeMap<Integer, Sequence> unjoinedSequence) throws MatrixException {
        for (Map.Entry<Integer, MMatrix> entry : sequence.entrySet()) {
            int sampleIndex = entry.getKey();
            MMatrix mMatrix = entry.getValue();
            MMatrix[] unjoinedMMatrix = MMatrix.unjoin(mMatrix);
            for (int index = 0; index < unjoinedMMatrix.length; index++) {
                unjoinedSequence.get(index).put(sampleIndex, unjoinedMMatrix[index]);
            }
        }
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
            for (Map.Entry<Integer, MMatrix> entry : sequence.entrySet()) {
                increment(entry.getKey(), entry.getValue());
            }
        }
        finally {
            accessLock.unlock();
        }
    }

    /**
     * Increments other multi-matrix to specific multi-matrix.
     *
     * @param sampleIndex sample index.
     * @param mMatrix other multi-matrix.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void increment(int sampleIndex, MMatrix mMatrix) throws MatrixException {
        accessLock.lock();
        try {
            MMatrix currentMMatrix = get(sampleIndex);
            if (currentMMatrix != null) currentMMatrix.add(mMatrix, currentMMatrix);
            else put(sampleIndex, mMatrix);
        }
        finally {
            accessLock.unlock();
        }
    }

    /**
     * Creates sequences out of multi-matrices.
     *
     * @param mMatrices multi-matrices
     * @return sequences
     * @throws MatrixException throws exception if depth of samples are not equal.
     */
    public static TreeMap<Integer, Sequence> getSequencesFromMMatrices(TreeMap<Integer, MMatrix> mMatrices) throws MatrixException {
        TreeMap<Integer, Sequence> sequences = new TreeMap<>();
        for (Map.Entry<Integer, MMatrix> entry : mMatrices.entrySet()) sequences.put(entry.getKey(), new Sequence(entry.getValue()));
        return sequences;
    }

    /**
     * Creates sequences out of matrices.
     *
     * @param matrices matrices
     * @return sequences
     * @throws MatrixException throws exception if depth of samples are not equal.
     */
    public static TreeMap<Integer, Sequence> getSequencesFromMatrices(TreeMap<Integer, Matrix> matrices) throws MatrixException {
        TreeMap<Integer, Sequence> sequences = new TreeMap<>();
        for (Map.Entry<Integer, Matrix> entry : matrices.entrySet()) sequences.put(entry.getKey(), new Sequence(new MMatrix(entry.getValue())));
        return sequences;
    }


}
