package utils;

import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.Serializable;
import java.util.Collection;
import java.util.HashMap;
import java.util.Set;

/**
 * Class that defines sample containing one or several matrices in depth direction.
 *
 */
public class Sample implements Serializable {

    private static final long serialVersionUID = -8196355331550904092L;

    /**
     * Depth of sample.
     *
     */
    private int depth;

    /**
     * Entries for sample in depth direction.
     *
     */
    private final HashMap<Integer, Matrix> entries = new HashMap<>();

    /**
     * Constructor for sample.
     *
     * @param depth depth of sample.
     * @throws MatrixException throws exception if depth is less than 11.
     */
    public Sample(int depth) throws MatrixException {
        if (depth < 1) throw new MatrixException("Sample depth must be at least 1.");
        this.depth = depth;
    }

    /**
     * Constructor for sample adding samples with specific depth.
     *
     * @param depth depth of sample.
     * @param entries sample entries to be added.
     * @throws MatrixException throws exception if entries are not fitting within depth.
     */
    public Sample(int depth, Sample entries) throws MatrixException {
        this(depth);
        putAll(entries);
    }

    /**
     * Returns size (depth) of sample.
     *
     * @return size (depth) of sample.
     */
    public int size() {
        return entries.size();
    }

    /**
     * Puts sample entry to specific depth index.
     *
     * @param index depth index.
     * @param entry specific entry.
     * @throws MatrixException throws exception if entry is not fitting within depth.
     */
    public void put(int index, Matrix entry) throws MatrixException {
        if (index < 0 || index > depth - 1) throw new MatrixException("Index violation: " + index + ". Sample index must be between 0 and depth " + (depth -1));
        entries.put(index, entry);
    }

    /**
     * Replaces all entries within sample.
     *
     * @param samples entries to be added.
     * @throws MatrixException throws exception if entry is not fitting within depth.
     */
    public void replaceAll(Sample samples) throws MatrixException {
        entries.clear();
        putAll(samples);
    }

    /**
     * Puts all entries into sample.
     *
     * @param samples entries to be added.
     * @throws MatrixException throws exception if entry is not fitting within depth.
     */
    public void putAll(Sample samples) throws MatrixException {
        if (samples.size() > depth) throw new MatrixException("Sample size " + samples.size() + " is beyond sample depth " + depth);
        entries.putAll(samples.get());
    }

    /**
     * Returns sample entries.
     *
     * @return sample entries.
     */
    public HashMap<Integer, Matrix> get() {
        return entries;
    }

    /**
     * Returns entry at specific sample (depth) index.
     *
     * @param index sample (depth) index.
     * @return entry to be returned.
     */
    public Matrix get(int index) {
        return entries.get(index);
    }

    /**
     * Returns index set of sample.
     *
     * @return index set of sample.
     */
    public Set<Integer> keySet() {
        return entries.keySet();
    }

    /**
     * Returns entries of sample as collection.
     *
     * @return entries of sample as collection.
     */
    public Collection<Matrix> values() {
        return entries.values();
    }

    /**
     * Checks if sample contains specific entry.
     *
     * @param matrix specific entry.
     * @return returns true is matrix is contained inside sample.
     */
    public boolean contains(Matrix matrix) {
        return entries.containsValue(matrix);
    }

}
