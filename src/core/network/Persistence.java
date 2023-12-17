/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.network;

import java.io.*;

/**
 * Implements persistence functionality for neural network.<br>
 * Persistence is used to store (serialize) neural network into file and restore (deserialize) neural network from file.<br>
 *
 */
public class Persistence implements Serializable {

    @Serial
    private static final long serialVersionUID = -6113168833637330117L;

    /**
     * Attribute to define if snapshots are stored at specific intervals.
     *
     */
    private boolean snapshot = false;

    /**
     * Interval between snapshots.
     *
     */
    private int interval = 0;

    /**
     * Reference to neural network instance to be made persistent.
     *
     */
    private NeuralNetwork neuralNetwork = null;

    /**
     * Filename into which persistent data of neural network is stored.<br>
     * Storage happens via Java serialization.<br>
     *
     */
    private String filename;

    /**
     * Define if potentially existing file is overwritten.<br>
     * If existing file is not to be overwritten then it is versioned by instance count.<br>
     *
     */
    private boolean overwrite;

    /**
     * Count for measuring if interval threshold has been reached.
     *
     */
    private int count = 0;

    /**
     * Counter for versioning.
     *
     */
    private int totalCount = 0;

    /**
     * Default constructor for persistence class.
     *
     */
    public Persistence() {
    }

    /**
     * Constructor for persistence class.
     *
     * @param snapshot true if snapshots are taken otherwise false
     * @param interval interval (in iterations) between snapshots.
     * @param neuralNetwork reference to neural network instance to be made persistent.
     * @param filename file name into which persistent neural network data is to be stored.
     * @param overwrite true if existing file name is to be overwritten.
     */
    public Persistence(boolean snapshot, int interval, NeuralNetwork neuralNetwork, String filename, boolean overwrite) {
        this.snapshot = snapshot;
        this.interval = interval;
        this.neuralNetwork = neuralNetwork;
        this.filename = filename;
        this.overwrite = overwrite;
    }

    /**
     * Returns reference to persistence.
     *
     * @param neuralNetwork reference to neural network instance to be made persistent.
     * @return reference to persistence.
     */
    public Persistence reference(NeuralNetwork neuralNetwork) {
        return new Persistence(snapshot, interval, neuralNetwork, filename, overwrite);
    }

    /**
     * Activates snapshots.
     *
     * @param interval interval (in iterations) between snapshots.
     * @param neuralNetwork reference to neural network instance to be made persistent.
     * @param filename file name into which persistent neural network data is to be stored.
     * @param overwrite true if existing file name is to be overwritten.
     */
    public void setSnapshot(int interval, NeuralNetwork neuralNetwork, String filename, boolean overwrite) {
        snapshot = true;
        this.interval = interval;
        this.neuralNetwork = neuralNetwork;
        this.filename = filename;
        this.overwrite = overwrite;
    }

    /**
     * Disables snapshots.
     *
     */
    public void unsetSnapshot() {
        snapshot =false;
        interval = 0;
        filename = "";
    }

    /**
     * Cycle function for regular storing of snapshots.
     *
     * @throws IOException throws exception if serialization of neural network object into file fails.
     */
    public void cycle() throws IOException {
        if (!snapshot) return;
        count++;
        totalCount++;
        if (count >= interval) {
            String currentFilename = filename;
            if (!overwrite) currentFilename += "-"+ totalCount;
            if (filename != null) saveNeuralNetwork(currentFilename, neuralNetwork);
            count = 0;
        }
    }

    /**
     * Resets snapshots counters.
     *
     */
    public void reset() {
        count = 0;
        totalCount = 0;
    }

    /**
     * Saves neural network into file.
     *
     * @param filename file name into which persistent neural network data is to be stored.
     * @param neuralNetwork reference to neural network instance to be made persistent.
     * @throws IOException throws exception if serialization of neural network object into file fails.
     */
    public static void saveNeuralNetwork(String filename, NeuralNetwork neuralNetwork) throws IOException {
        FileOutputStream fileOutputStream = new FileOutputStream(filename + ".ser");
        ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream);
        objectOutputStream.writeObject(neuralNetwork);
        objectOutputStream.flush();
        objectOutputStream.close();
    }

    /**
     * Restores neural network from file.
     *
     * @param filename file name from which persistent neural network data is restored from.
     * @return de-serialized neural network instance.
     * @throws IOException throws exception if deserialization of neural network object from file fails.
     * @throws ClassNotFoundException throws exception if instantiation of neural network from serialized object stream fails.
     */
    public static NeuralNetwork restoreNeuralNetwork(String filename) throws IOException, ClassNotFoundException {
        FileInputStream fileInputStream = new FileInputStream(filename + ".ser");
        ObjectInputStream objectInputStream = new ObjectInputStream(fileInputStream);
        NeuralNetwork neuralNetwork = (NeuralNetwork)objectInputStream.readObject();
        objectInputStream.close();
        return neuralNetwork;
    }

}
