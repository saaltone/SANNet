/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2020 Simo Aaltonen
 */

package utils;

import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serial;
import java.io.Serializable;
import java.util.*;

/**
 * Class that implements sequence of samples.<br>
 *
 */
public class Sequence implements Serializable {

    @Serial
    private static final long serialVersionUID = 4183245025751674913L;

    /**
     * Sample depth.
     *
     */
    private final int depth;

    /**
     * Ordered map of samples.
     *
     */
    private TreeMap<Integer, MMatrix> entries = new TreeMap<>();

    /**
     * Constructor for sequence.
     *
     * @param depth depth of samples.
     * @throws MatrixException throws exception if depth of matrix is less than 1.
     */
    public Sequence(int depth) throws MatrixException {
        if (depth < 1) throw new MatrixException("Depth of sequence must be at least 1.");
        this.depth = depth;
    }

    /**
     * Constructor for sequence.
     *
     * @param sequence sequence to be applied into this sequence.
     */
    public Sequence(Sequence sequence) {
        this.depth = sequence.getDepth();
        putAll(sequence);
    }

    /**
     * Constructor for sequence.
     *
     * @param samples samples to be added into this sequence.
     * @throws MatrixException throws exception if depth of samples are not equal.
     */
    public Sequence(LinkedHashMap<Integer, MMatrix> samples) throws MatrixException {
        this.depth = samples.get(0).getCapacity();
        for (Integer entry : samples.keySet()) {
            if (samples.get(entry).getCapacity() != depth) throw new MatrixException("Depths of all samples are not equal.");
            entries.put(entry, samples.get(entry));
        }
    }

    /**
     * Checks if sequence is empty.
     *
     * @return returns true if sequence is empty otherwise returns false.
     */
    public boolean isEmpty() {
        return entries.isEmpty();
    }

    /**
     * Returns number of samples in sequence.
     *
     * @return number of samples in sequence.
     */
    public int sampleSize() {
        return entries.size();
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
        return entries.size() * depth;
    }

    /**
     * Clears sequence i.e. removes any existing samples.
     *
     */
    public void clear() {
        entries = new TreeMap<>();
    }

    /**
     * Puts new sample into specific sample index and entry (sample depth) index.
     *
     * @param sampleIndex sample index.
     * @param entryIndex entry (sample depth) index.
     * @param entry entry to be inserted.
     * @throws MatrixException throws exception if put operation fails.
     */
    public void put(int sampleIndex, int entryIndex, Matrix entry) throws MatrixException {
        if (entries.containsKey(sampleIndex)) entries.get(sampleIndex).put(entryIndex, entry);
        else {
            MMatrix sample = new MMatrix(depth);
            sample.put(entryIndex, entry);
            entries.put(sampleIndex, sample);
        }
    }

    /**
     * Puts sample into specific sample index.
     *
     * @param sampleIndex sample index.
     * @param sample sample to be inserted.
     */
    public void put(int sampleIndex, MMatrix sample) {
        entries.put(sampleIndex, sample);
    }

    /**
     * Returns entry at specific sample index and entry (sample depth) index.
     *
     * @param sampleIndex sample index.
     * @param entryIndex entry (sample depth) index.
     * @return requested entry.
     */
    public Matrix get(int sampleIndex, int entryIndex) {
        return entries.get(sampleIndex).get(entryIndex);
    }

    /**
     * Replaces all samples in sequence.
     *
     * @param sequence sequence containing new samples for this sequence.
     */
    public void replaceAll(Sequence sequence) {
        clear();
        putAll(sequence);
    }

    /**
     * Puts all samples into sequence.
     *
     * @param sequence sequence containing new samples for this sequence.
     */
    public void putAll(Sequence sequence) {
        entries.putAll(sequence.get());
    }

    /**
     * Returns sample at specific sample index.
     *
     * @param sampleIndex sample index.
     * @return requested sample.
     */
    public MMatrix get(int sampleIndex) {
        return entries.get(sampleIndex);
    }

    /**
     * Returns all samples inside sequence as ordered map.
     *
     * @return all samples inside sequence as ordered map.
     */
    public TreeMap<Integer, MMatrix> get() {
        return entries;
    }

    /**
     * Returns sample index key set.
     *
     * @return sample index key set.
     */
    public Set<Integer> keySet() {
        return entries.keySet();
    }

    /**
     * Returns sample index key set in descending order.
     *
     * @return sample index key set in descending order.
     */
    public Set<Integer> descendingKeySet() {
        return entries.descendingKeySet();
    }

    /**
     * Returns samples contained inside sequence as collection.
     *
     * @return samples contained inside sequence as collection.
     */
    public Collection<MMatrix> values() {
        return entries.values();
    }

    /**
     * Returns sample key set.
     *
     * @return sample key set.
     */
    public Set<Integer> sampleKeySet() {
        return entries.get(entries.firstKey()).keySet();
    }

    /**
     * Returns first index of sequence.
     *
     * @return first index of sequence.
     */
    public Integer firstKey() {
        return entries.firstKey();
    }

    /**
     * Returns first sample of sequence.
     *
     * @return first sample of sequence.
     */
    public MMatrix firstValue() {
        return entries.get(entries.firstKey());
    }

    /**
     * Returns last index of sequence.
     *
     * @return last index of sequence.
     */
    public Integer lastKey() {
        return entries.lastKey();
    }

    /**
     * Returns last sample of sequence.
     *
     * @return last sample of sequence.
     */
    public MMatrix lastValue() {
        return entries.get(entries.lastKey());
    }

    /**
     * Returns flattened sequence i.e. samples that have been flattened into single row with depth one.
     *
     * @return flattened sequence.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Sequence flatten() throws MatrixException {
        Sequence flattenedSequence = new Sequence(1);
        for (Integer sampleIndex : keySet()) flattenedSequence.put(sampleIndex, get(sampleIndex).flatten());
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
        Sequence unflattenedSequence = new Sequence(depth);
        for (Integer sampleIndex : keySet()) unflattenedSequence.put(sampleIndex, get(sampleIndex).unflatten(width, height, depth));
        return unflattenedSequence;
    }

}
