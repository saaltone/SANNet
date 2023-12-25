/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.preprocess;

import utils.matrix.*;

import javax.sound.midi.*;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * Implements functionality reading, encoding, playing and writing out MIDI file.<br>
 *
 */
public class ReadMIDI {

    /**
     * Default constructor for read MIDI file utility.
     *
     */
    public ReadMIDI() {
    }

    /**
     * Defines metadata for MIDI
     *
     */
    public static class Metadata {

        /**
         * Division type for MIDI sequence.
         *
         */
        private final float divisionType;

        /**
         * Resolution for MIDI sequence.
         *
         */
        public final int resolution;

        /**
         * If true note off are encoded otherwise not.
         *
         */
        public boolean encodeNoteOffs;

        /**
         * Maximum number of encoded ticks.
         *
         */
        public final int maxNumberOfEncodedTicks;

        /**
         * Number of encoded ticks.
         *
         */
        public int numberOfEncodedTicks;

        /**
         * Tick value mapping.
         *
         */
        public final HashMap<Long, Integer> tickValueMapping = new HashMap<>();

        /**
         * Tick value reverse mapping.
         *
         */
        public final HashMap<Integer, Long> tickValueReverseMapping = new HashMap<>();

        /**
         * Minimum tick value found.
         *
         */
        public long minTickValue = Long.MAX_VALUE;

        /**
         * Maximum tick value found.
         *
         */
        public long maxTickValue = Long.MIN_VALUE;

        /**
         * Constructor for MetaData.
         *
         * @param divisionType division type
         * @param resolution resolution
         */
        Metadata(float divisionType, int resolution) {
            this.maxNumberOfEncodedTicks = 25;
            this.divisionType = divisionType;
            this.resolution = resolution;
        }

        /**
         * Constructor for MetaData.
         *
         * @param maxNumberOfEncodedTicks maximum number of encoded ticks.
         * @param divisionType division type
         * @param resolution resolution
         */
        Metadata(int maxNumberOfEncodedTicks, boolean encodeNoteOffs, float divisionType, int resolution) {
            this.maxNumberOfEncodedTicks = maxNumberOfEncodedTicks;
            this.encodeNoteOffs = encodeNoteOffs;
            this.divisionType = divisionType;
            this.resolution = resolution;
        }

        /**
         * Returns division type
         *
         * @return division type
         */
        private float getDivisionType() {
            return divisionType;
        }

        /**
         * Returns resolution
         *
         * @return resolution
         */
        private int getResolution() {
            return resolution;
        }

        /**
         * Returns maximum number of encoded ticks
         *
         * @return maximum number of encoded ticks
         */
        private int getMaxNumberOfEncodedTicks() {
            return maxNumberOfEncodedTicks;
        }

        /**
         * Returns note offset
         *
         * @return note offset
         */
        public int getNoteOffset() {
            return checkEncodeNoteOffs() ? 1 : 0;
        }

        /**
         * Checks if note offs are encoded.
         *
         * @return true if note offs are encoded.
         */
        public boolean checkEncodeNoteOffs() {
            return encodeNoteOffs;
        }

    }

    /**
     * Metadata for MIDI record.
     *
     */
    private Metadata metadata = null;

    /**
     * Returns metadata for MIDI record.
     *
     * @return metadata for MIDI record.
     */
    public Metadata getMetadata() {
        return metadata;
    }

    /**
     * Reads and encodes MIDI file.
     *
     * @param fileName file name
     * @return encoded MIDI file
     * @throws InvalidMidiDataException if opening file fails throws exception.
     * @throws IOException if opening file fails throws exception.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public HashMap<Integer, HashMap<Integer, HashMap<Integer, Matrix>>> readFile(String fileName) throws InvalidMidiDataException, IOException, MatrixException {
        return readFile(new ArrayList<>() {{ add(fileName); }}, 1, 1, 1, false, 0, Long.MAX_VALUE, 25);
    }

    /**
     * Reads and encodes MIDI files.
     *
     * @param fileNames              MIDI files.
     * @param numberOfKeyInputs      number of inputs.
     * @param numberOfVelocityInputs number of velocity inputs.
     * @param numberOfTickInputs     number of tick inputs.
     * @param encodeNoteOffs         if true encodes note offs otherwise does not include note offs.
     * @param minTickDelta           minimum tick delta
     * @param maxTickDelta           maximum tick delta
     * @param maxEncodedTicks        maximum number of encoded ticks.
     * @return encoded inputs and outputs.
     * @throws InvalidMidiDataException if opening file fails throws exception.
     * @throws IOException              if opening file fails throws exception.
     * @throws MatrixException          throws exception if matrix operation fails.
     */
    public HashMap<Integer, HashMap<Integer, HashMap<Integer, Matrix>>> readFile(ArrayList<String> fileNames, int numberOfKeyInputs, int numberOfVelocityInputs, int numberOfTickInputs, boolean encodeNoteOffs, long minTickDelta, long maxTickDelta, int maxEncodedTicks) throws InvalidMidiDataException, IOException, MatrixException {
        ArrayList<Integer> keyDataAsInteger = new ArrayList<>();
        ArrayList<Integer> velocityDataAsInteger = new ArrayList<>();
        ArrayList<Long> tickData = new ArrayList<>();

        TreeMap<Long, Integer> tickDataDistribution = new TreeMap<>();

        for (String fileName : fileNames) {
            Sequence sequence = MidiSystem.getSequence(new File(fileName));
            if (metadata == null) metadata = new Metadata(maxEncodedTicks, encodeNoteOffs, sequence.getDivisionType(), sequence.getResolution());
            if (sequence.getDivisionType() != Sequence.PPQ) {
                System.out.println("File: '" + fileName + "' omitted because Sequence must have division type of PPQ.");
                continue;
            }
            Track[] tracks = sequence.getTracks();
            for (Track track : tracks) {
                long previousTick = 0;
                int trackSize = track.size();
                for (int trackIndex = 0; trackIndex < trackSize; trackIndex++) {
                    MidiEvent midiEvent = track.get(trackIndex);
                    if (midiEvent.getMessage() instanceof ShortMessage shortMessage) {
                        int keyValue = Math.min(127, Math.max(0, shortMessage.getData1())) + metadata.getNoteOffset();
                        int velocityValue = Math.min(127, Math.max(0, shortMessage.getData2())) + metadata.getNoteOffset();
                        long tick = midiEvent.getTick();
                        long tickDelta = Math.min(maxTickDelta, Math.max(minTickDelta, tick - previousTick));
                        previousTick = tick;
                        switch (shortMessage.getCommand()) {
                            case ShortMessage.NOTE_ON -> {
                                if (velocityValue > 0) {
                                    keyDataAsInteger.add(keyValue);
                                    velocityDataAsInteger.add(velocityValue);
                                    tickData.add(tickDelta);
                                    if (tickDataDistribution.containsKey(tickDelta)) tickDataDistribution.put(tickDelta, tickDataDistribution.get(tickDelta) + 1);
                                    else tickDataDistribution.put(tickDelta, 1);
                                }
                            }
                            case ShortMessage.NOTE_OFF -> {
                                if (metadata.checkEncodeNoteOffs()) {
                                    keyDataAsInteger.add(keyValue);
                                    velocityDataAsInteger.add(0);
                                    tickData.add(0L);
                                    if (tickDataDistribution.containsKey(tickDelta)) tickDataDistribution.put(tickDelta, tickDataDistribution.get(tickDelta) + 1);
                                    else tickDataDistribution.put(0L, 1);
                                }
                            }
                            default -> {
                            }
                        }
                    }
                }
            }
        }

        ArrayList<Matrix> inputKeyDataAsMatrix = encodeDataIntoMatrix(keyDataAsInteger, true, metadata.checkEncodeNoteOffs() ? 129 : 128, metadata.checkEncodeNoteOffs() ? 8 : 7);
        ArrayList<Matrix> outputKeyDataAsMatrix = encodeDataIntoMatrix(keyDataAsInteger, false, metadata.checkEncodeNoteOffs() ? 129 : 128, metadata.checkEncodeNoteOffs() ? 8 : 7);
        ArrayList<Matrix> inputVelocityDataAsMatrix = encodeDataIntoMatrix(velocityDataAsInteger, true, metadata.checkEncodeNoteOffs() ? 129 : 128, metadata.checkEncodeNoteOffs() ? 8 : 7);
        ArrayList<Matrix> outputVelocityDataAsMatrix = encodeDataIntoMatrix(velocityDataAsInteger, false, metadata.checkEncodeNoteOffs() ? 129 : 128, metadata.checkEncodeNoteOffs() ? 8 : 7);
        ArrayList<Matrix> tickDataAsMatrix = scaleTickData(tickData, metadata);

        HashMap<Integer, HashMap<Integer, HashMap<Integer, Matrix>>> result = new HashMap<>();
        int numberOfSamples = Math.min(inputKeyDataAsMatrix.size() - numberOfKeyInputs - 1, inputKeyDataAsMatrix.size() - numberOfKeyInputs - 1);
        numberOfSamples = Math.min(numberOfSamples, tickDataAsMatrix.size() - numberOfTickInputs - 1);
        result.put(0, sequenceData(numberOfKeyInputs, inputKeyDataAsMatrix, numberOfSamples));
        result.put(1, sequenceData(numberOfKeyInputs, outputKeyDataAsMatrix, numberOfSamples));
        result.put(2, sequenceData(numberOfVelocityInputs, inputVelocityDataAsMatrix, numberOfSamples));
        result.put(3, sequenceData(numberOfVelocityInputs, outputVelocityDataAsMatrix, numberOfSamples));
        result.put(4, sequenceData(numberOfTickInputs, tickDataAsMatrix, numberOfSamples));

        return result;
    }

    /**
     * Sequences input data.
     *
     * @param numberOfInputs number of inputs.
     * @param dataAsMatrix data as matrix.
     * @param numberOfSamples number of samples.
     * @return sequenced input data.
     */
    private HashMap<Integer, HashMap<Integer, Matrix>> sequenceData(int numberOfInputs, ArrayList<Matrix> dataAsMatrix, int numberOfSamples) {
        HashMap<Integer, HashMap<Integer, Matrix>> result = new HashMap<>();
        int numberOfInputsOutputs = numberOfInputs + 1;
        for (int index = 0; index < numberOfInputsOutputs; index++) result.put(index, new HashMap<>());
        for (int dataIndex = 0; dataIndex < numberOfSamples; dataIndex++) {
            for (int entryIndex = 0; entryIndex < numberOfInputsOutputs; entryIndex++) {
                result.get(entryIndex).put(dataIndex, dataAsMatrix.get(dataIndex + entryIndex));
            }
        }
        return result;
    }

    /**
     * Encodes key and velocity entries into matrix form.
     *
     * @param integerData  key or velocity integer data
     * @param encodeAsBits if true encodes as bits otherwise encodes as one hot vector
     * @param outputSize   output size
     * @param maxBits      maximum number of bits in encoding
     * @return encoded matrices.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private ArrayList<Matrix> encodeDataIntoMatrix(ArrayList<Integer> integerData, boolean encodeAsBits, int outputSize, int maxBits) throws MatrixException {
        ArrayList<Matrix> dataAsMatrix = new ArrayList<>();
        for (Integer integerDataItem : integerData) {
            if (encodeAsBits) {
                if (maxBits > 0) dataAsMatrix.add(AbstractMatrix.encodeValueToBitColumnVector(integerDataItem, maxBits));
            }
            else {
                if (outputSize > 0) dataAsMatrix.add(SMatrix.getOneHotVector(outputSize, integerDataItem));
            }
        }
        return dataAsMatrix;
    }

    /**
     * Scales tick data between zero and one and encodes into matrix form.
     *
     * @param inputTickData input tick data.
     * @param metadata record data.
     * @return scaled tick data in matrix form.
     */
    private ArrayList<Matrix> scaleTickData(ArrayList<Long> inputTickData, Metadata metadata) throws MatrixException {

        class TickFrequency implements Comparable<TickFrequency> {

            /**
             * Frequency.
             *
             */
            int frequency;

            /**
             * Tick
             *
             */
            final long tick;

            /**
             * Constructor for tick frequency.
             *
             * @param tick tick.
             */
            TickFrequency(long tick) {
                frequency = 1;
                this.tick = tick;
            }

            /**
             * Increments reported frequency.
             *
             */
            void increment() {
                frequency++;
            }

            /**
             * Compares this and other tick frequency
             *
             * @param otherTickFrequency the object to be compared.
             * @return result of comparison.
             */
            public int compareTo(TickFrequency otherTickFrequency) {
                return Integer.compare(frequency, otherTickFrequency.frequency);
            }
        }

        ArrayList<TickFrequency> tickFrequencyDistribution = new ArrayList<>();

        HashMap<Long, TickFrequency> tickFrequencyMap = new HashMap<>();
        for (Long inputTick : inputTickData) {
            if (!tickFrequencyMap.containsKey(inputTick)) {
                TickFrequency tickFrequency = new TickFrequency(inputTick);
                tickFrequencyMap.put(inputTick, tickFrequency);
                tickFrequencyDistribution.add(tickFrequency);
            }
            else tickFrequencyMap.get(inputTick).increment();
        }
        Collections.sort(tickFrequencyDistribution);

        TreeMap<Long, TickFrequency> updatedTickFrequencyMap = new TreeMap<>();
        ArrayList<Long> tickOrder = new ArrayList<>();
        for (TickFrequency tickFrequency : tickFrequencyDistribution) {
            int tickOrderSize = tickOrder.size();
            if (tickOrderSize < metadata.getMaxNumberOfEncodedTicks()) {
                updatedTickFrequencyMap.put(tickFrequency.tick, tickFrequency);
                tickOrder.add(tickFrequency.tick);
                metadata.tickValueMapping.put(tickFrequency.tick, tickOrderSize);
                metadata.tickValueReverseMapping.put(tickOrderSize, tickFrequency.tick);
            }
        }

        ArrayList<Matrix> outputTickData = new ArrayList<>();

        metadata.numberOfEncodedTicks = updatedTickFrequencyMap.size();
        for (Long inputTick : inputTickData) {
            long tick;
            if (updatedTickFrequencyMap.containsKey(inputTick)) tick = inputTick;
            else {
                long floorTick = updatedTickFrequencyMap.floorKey(inputTick) == null ? updatedTickFrequencyMap.firstKey() : updatedTickFrequencyMap.floorKey(inputTick);
                long ceilingTick = updatedTickFrequencyMap.ceilingKey(inputTick) == null ? updatedTickFrequencyMap.lastKey() : updatedTickFrequencyMap.ceilingKey(inputTick);
                tick = Math.abs(inputTick - floorTick) < Math.abs(inputTick - ceilingTick) ? floorTick : ceilingTick;
            }
            outputTickData.add(SMatrix.getOneHotVector(metadata.numberOfEncodedTicks, tickOrder.indexOf(tick)));
        }

        return outputTickData;
    }

    /**
     * Returns MIDI sequence based on data.
     *
     * @param dataKey MIDI key data.
     * @param dataVelocity MIDI velocity data.
     * @param dataTick MIDI tick data.
     * @param resolution resolution.
     * @param asInput if true matrices are provided as input otherwise as output.
     * @param tickScalingConstant tick scaling constant.
     * @return MIDI sequence based on data.
     * @throws InvalidMidiDataException if opening file fails throws exception.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Sequence getSequence(HashMap<Integer, Matrix> dataKey, HashMap<Integer, Matrix> dataVelocity, HashMap<Integer, Matrix> dataTick, int resolution, boolean asInput, double tickScalingConstant) throws InvalidMidiDataException, MatrixException {
        Sequence sequence = new Sequence(Sequence.PPQ, resolution);
        Track track = sequence.createTrack();
        long currentTick = 0;
        boolean firstEntry = true;
        for (Map.Entry<Integer, Matrix> entry : dataKey.entrySet()) {
            int index = entry.getKey();
            Matrix keyMatrix = entry.getValue();
            Matrix velocityMatrix = dataVelocity.get(index);
            Matrix tickMatrix = dataTick.get(index);
            currentTick = addMIDISequenceEntry(keyMatrix, velocityMatrix, tickMatrix, track, currentTick, firstEntry, asInput, tickScalingConstant);
            firstEntry = false;
        }
        return sequence;
    }

    /**
     * Returns MIDI sequence based on data.
     *
     * @param dataKey MIDI key data.
     * @param dataVelocity MIDI velocity data.
     * @param dataTick MIDI tick data.
     * @param asInput if true matrices are provided as input otherwise as output.
     * @param metadata metadata
     * @param tickScalingConstant tick scaling constant.
     * @return MIDI sequence based on data.
     * @throws InvalidMidiDataException if opening file fails throws exception.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Sequence getSequenceAsMatrix(HashMap<Integer, Matrix> dataKey, HashMap<Integer, Matrix> dataVelocity, HashMap<Integer, Matrix> dataTick, boolean asInput, Metadata metadata, double tickScalingConstant) throws InvalidMidiDataException, MatrixException {
        Sequence sequence = new Sequence(metadata.getDivisionType(), metadata.getResolution());
        Track track = sequence.createTrack();
        long currentTick = 0;
        boolean firstEntry = true;
        for (Map.Entry<Integer, Matrix> entry : dataKey.entrySet()) {
            int index = entry.getKey();
            Matrix keyMatrix = entry.getValue();
            Matrix velocityMatrix = dataVelocity.get(index);
            Matrix tickMatrix = dataTick.get(index);
            currentTick = addMIDISequenceEntry(keyMatrix, velocityMatrix, tickMatrix, track, currentTick, firstEntry, asInput, tickScalingConstant);
            firstEntry = false;
        }
        return sequence;
    }

    /**
     * Add entry to MIDI sequence
     *
     * @param keyMatrix key matrix
     * @param velocityMatrix velocity matrix
     * @param tickMatrix tick matrix
     * @param track track
     * @param currentTick current tick
     * @param firstEntry if true this is first entry otherwise false
     * @param asInput if true matrices are provided as input otherwise as output.
     * @param tickScalingConstant tick scaling constant.
     * @return current tick
     * @throws InvalidMidiDataException if opening file fails throws exception.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private long addMIDISequenceEntry(Matrix keyMatrix, Matrix velocityMatrix, Matrix tickMatrix, Track track, long currentTick, boolean firstEntry, boolean asInput, double tickScalingConstant) throws MatrixException, InvalidMidiDataException {
        int keyValue = asInput ? keyMatrix.classify().encodeBitColumnVectorToValue() : keyMatrix.argmax()[0];
        int velocityValue = asInput ? velocityMatrix.classify().encodeBitColumnVectorToValue() : velocityMatrix.argmax()[0];
        if (firstEntry) {
            ShortMessage shortMessage = new ShortMessage();
            shortMessage.setMessage(ShortMessage.PROGRAM_CHANGE, 46, 0);
            MidiEvent midiEvent = new MidiEvent(shortMessage, currentTick);
            track.add(midiEvent);
/*            shortMessage = new ShortMessage();
            shortMessage.setMessage(ShortMessage.CONTROL_CHANGE, 7, 127);
            midiEvent = new MidiEvent(shortMessage, currentTick);
            track.add(midiEvent);
            shortMessage = new ShortMessage();
            shortMessage.setMessage(ShortMessage.CONTROL_CHANGE, 10, 58);
            midiEvent = new MidiEvent(shortMessage, currentTick);
            track.add(midiEvent);
            shortMessage = new ShortMessage();
            shortMessage.setMessage(ShortMessage.CONTROL_CHANGE, 1, 10);
            midiEvent = new MidiEvent(shortMessage, currentTick);
            track.add(midiEvent);
            shortMessage = new ShortMessage();
            shortMessage.setMessage(ShortMessage.CONTROL_CHANGE, 91, 65);
            midiEvent = new MidiEvent(shortMessage, currentTick);
            track.add(midiEvent);
            shortMessage = new ShortMessage();
            shortMessage.setMessage(ShortMessage.CONTROL_CHANGE, 93, 110);
            midiEvent = new MidiEvent(shortMessage, currentTick);
            track.add(midiEvent); */
        }
        ShortMessage shortMessage = new ShortMessage();
        if (metadata.encodeNoteOffs) {
            if (velocityValue == 0) {
                shortMessage.setMessage(ShortMessage.NOTE_OFF, keyValue - 1, 0);
            }
            else {
                shortMessage.setMessage(ShortMessage.NOTE_ON, keyValue - 1, velocityValue - 1);
            }
        }
        else shortMessage.setMessage(ShortMessage.NOTE_ON, keyValue, velocityValue);
        currentTick += metadata.tickValueReverseMapping.get(tickMatrix.argmax()[0]) / tickScalingConstant;
        MidiEvent midiEvent = new MidiEvent(shortMessage, currentTick);
        track.add(midiEvent);
        return currentTick;
    }

    /**
     * Plays MIDI sequence.
     *
     * @param sequence sequence
     * @param playTime play time before stopping.
     * @param wait if true waits given play time before stopping otherwise returns function after starts playing.
     * @throws MidiUnavailableException throws exception if playing fails.
     * @throws InvalidMidiDataException throws exception if playing fails.
     * @return sequencer
     */
    public Sequencer play(Sequence sequence, int playTime, boolean wait) throws MidiUnavailableException, InvalidMidiDataException {
        Sequencer sequencer = MidiSystem.getSequencer();
        sequencer.setSequence(sequence);

        stopPlaying(sequencer);

        sequencer.open();
        sequencer.start();

        if (!wait) return sequencer;

        try {
            TimeUnit.SECONDS.sleep(playTime);
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new RuntimeException(exception);
        }

        sequencer.stop();
        sequencer.close();

        return sequencer;
    }

    /**
     * Stops playing MIDI file.
     *
     * @param sequencer sequencer.
     */
    public void stopPlaying(Sequencer sequencer) {
        if (sequencer.isRunning()) {
            sequencer.stop();
            sequencer.close();
        }
    }

    /**
     * Writes MIDI sequence to file.
     *
     * @param sequence sequence
     * @param fileName file
     * @param version version number of file
     * @throws IOException throw exception is writing to file fails.
     */
    public void writeMIDI(Sequence sequence, String fileName, int version) throws IOException {
        int midiType = MidiSystem.getMidiFileTypes(sequence)[0];
        MidiSystem.write(sequence, midiType, new File(fileName + "_" + version + ".mid"));
    }

}
