/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package utils.sampling;

import utils.matrix.MMatrix;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;
import java.util.*;

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
    private int depth = -1;

    /**
     * Ordered map of samples.
     *
     */
    private final TreeMap<Integer, MMatrix> samples = new TreeMap<>();

    /**
     * Constructor for sequence.
     *
     */
    public Sequence() {
    }

    /**
     * Constructor for sequence.
     *
     * @param sequence sequence to be applied into this sequence.
     * @throws MatrixException throws exception if depth of sequence is not matching depth of this sequence.
     */
    public Sequence(Sequence sequence) throws MatrixException {
        depth = sequence.getDepth();
        putAll(sequence);
    }

    /**
     * Constructor for sequence.
     *
     * @param newSamples samples to be added into this sequence.
     * @throws MatrixException throws exception if depth of samples are not equal.
     */
    public Sequence(HashMap<Integer, MMatrix> newSamples) throws MatrixException {
        this.depth = newSamples.get(0).getDepth();
        for (Map.Entry<Integer, MMatrix> entry : newSamples.entrySet()) {
            int key = entry.getKey();
            MMatrix value = entry.getValue();
            if (value.getDepth() != depth) throw new MatrixException("Depths of all samples are not equal.");
            samples.put(key, value);
        }
    }

    /**
     * Checks if sequence is empty.
     *
     * @return returns true if sequence is empty otherwise returns false.
     */
    public boolean isEmpty() {
        return samples.isEmpty();
    }

    /**
     * Returns number of samples in sequence.
     *
     * @return number of samples in sequence.
     */
    public int sampleSize() {
        return samples.size();
    }

    /**
     * Returns depth of sample.
     *
     * @return depth of sample.
     */
    public int getDepth() {
        return depth;
    }

    /**
     * Returns total size (amount of sample * depth of samples) of sequence.
     *
     * @return total size of sequence.
     */
    public int totalSize() {
        return samples.size() * depth;
    }

    /**
     * Puts sample into specific sample index.
     *
     * @param sampleIndex sample index.
     * @param sample sample to be inserted.
     * @throws MatrixException throws exception if depth of sample is not matching depth of sequence.
     */
    public void put(int sampleIndex, MMatrix sample) throws MatrixException {
        if (depth == -1) depth = sample.getDepth();
        else if (depth != sample.getDepth()) throw new MatrixException("Depth of sample is not matching depth of sequence: " + getDepth());
        samples.put(sampleIndex, sample);
    }

    /**
     * Puts all samples into sequence.
     *
     * @param sequence sequence containing new samples for this sequence.
     * @throws MatrixException throws exception if depth of sequence is not matching depth of this sequence.
     */
    public void putAll(Sequence sequence) throws MatrixException {
        if (depth == -1) depth = sequence.getDepth();
        else if (depth != sequence.getDepth()) throw new MatrixException("Depth of sequence is not matching depth of this sequence: " + getDepth());
        samples.putAll(sequence.get());
    }

    /**
     * Returns sample at specific sample index.
     *
     * @param sampleIndex sample index.
     * @return requested sample.
     */
    public MMatrix get(int sampleIndex) {
        return samples.get(sampleIndex);
    }

    /**
     * Returns all samples inside sequence as ordered map.
     *
     * @return all samples inside sequence as ordered map.
     */
    public TreeMap<Integer, MMatrix> get() {
        return samples;
    }

    /**
     * Returns sample values.
     *
     * @return sample values.
     */
    public Collection<MMatrix> values() {
        return samples.values();
    }

    /**
     * Returns sample index key set.
     *
     * @return sample index key set.
     */
    public Set<Integer> keySet() {
        return samples.keySet();
    }

    /**
     * Returns sample index entry set.
     *
     * @return sample index entry set.
     */
    public Set<Map.Entry<Integer, MMatrix>> entrySet() {
        return samples.entrySet();
    }

    /**
     * Returns descending sample index entry set.
     *
     * @return descending sample index entry set.
     */
    public Set<Map.Entry<Integer, MMatrix>> descendingEntrySet() {
        return samples.descendingMap().entrySet();
    }

    /**
     * Returns sample index key set in descending order.
     *
     * @return sample index key set in descending order.
     */
    public Set<Integer> descendingKeySet() {
        return samples.descendingKeySet();
    }

    /**
     * Returns first index of sequence.
     *
     * @return first index of sequence.
     */
    public Integer firstKey() {
        return samples.firstKey();
    }

    /**
     * Returns last index of sequence.
     *
     * @return last index of sequence.
     */
    public Integer lastKey() {
        return samples.lastKey();
    }

    /**
     * Returns flattened sequence i.e. samples that have been flattened into single row with depth one.
     *
     * @return flattened sequence.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Sequence flatten() throws MatrixException {
        Sequence flattenedSequence = new Sequence();
        for (Map.Entry<Integer, MMatrix> entry : entrySet()) {
            int sampleIndex = entry.getKey();
            MMatrix mMatrix = entry.getValue();
            flattenedSequence.put(sampleIndex, mMatrix.flatten());
        }
        return flattenedSequence;
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
        Sequence unflattenedSequence = new Sequence();
        for (Map.Entry<Integer, MMatrix> entry : entrySet()) {
            int sampleIndex = entry.getKey();
            MMatrix mMatrix = entry.getValue();
            unflattenedSequence.put(sampleIndex, mMatrix.unflatten(width, height, depth));
        }
        return unflattenedSequence;
    }

    /**
     * Joins this and other sequence together by sample indices.
     *
     * @param otherSequence other sequence.
     * @param joinedVertically if true sequences are joined together vertically otherwise horizontally.
     * @return joined sequence.
     * @throws MatrixException throws exception if joining of matrices fails.
     */
    public Sequence join(Sequence otherSequence, boolean joinedVertically) throws MatrixException {
        if (getDepth() != otherSequence.getDepth()) throw new MatrixException("Depth of this sequence + " + getDepth() + " and other sequence " + otherSequence.getDepth() + " do not match.");
        Sequence joinedSequence = new Sequence();
        for (Map.Entry<Integer, MMatrix> entry : entrySet()) {
            int sampleIndex = entry.getKey();
            MMatrix mMatrix = entry.getValue();
            if (mMatrix == null) throw new MatrixException("Other sequence does not contain sample index: " + sampleIndex);
            joinedSequence.put(sampleIndex, mMatrix.join(mMatrix, joinedVertically));
        }
        return joinedSequence;
    }

    /**
     * Unjoins sequence by return specific submatrices of JMatrix.
     *
     * @param subMatrixIndex sub matrix index.
     * @return unjoined sequence.
     * @throws MatrixException throws exception if unjoining of matrices fails.
     */
    public Sequence unjoin(int subMatrixIndex) throws MatrixException {
        Sequence unjoinedSequence = new Sequence();
        for (Map.Entry<Integer, MMatrix> entry : entrySet()) {
            int sampleIndex = entry.getKey();
            MMatrix mMatrix = entry.getValue();
            unjoinedSequence.put(sampleIndex, mMatrix.unjoin(subMatrixIndex));
        }
        return unjoinedSequence;
    }

}
