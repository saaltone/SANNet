/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.preprocess;

import utils.matrix.*;

import javax.sound.midi.*;
import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Implements functionality reading, encoding, playing and writing out MIDI file.<br>
 *
 */
public class ReadMIDI {

    /**
     * Defines metadata for MIDI
     *
     */
    public static class Metadata {

        /**
         * Resolution for MIDI sequence.
         *
         */
        public final int resolution = 120;

        /**
         * Minimum key value found.
         *
         */
        public int minKeyValue = Integer.MAX_VALUE;

        /**
         * Maximum key value found.
         *
         */
        public int maxKeyValue = Integer.MIN_VALUE;

        /**
         * Output size of key mapping.
         *
         */
        public int keyOutputSize;

        /**
         * Bit vector size needed to encode key for input.
         *
         */
        public int keyBitVectorSize;

        /**
         * Minimum velocity value found.
         *
         */
        public int minVelocityValue = Integer.MAX_VALUE;

        /**
         * Maximum velocity value found.
         *
         */
        public int maxVelocityValue = Integer.MIN_VALUE;

        /**
         * Output size of velocity mapping.
         *
         */
        public int velocityOutputSize;

        /**
         * Bit vector size needed to encode velocity for input.
         *
         */
        public int velocityBitVectorSize;

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
        public final HashMap<Integer, Integer> tickValueMapping = new HashMap<>();

        /**
         * Tick value reverse mapping.
         *
         */
        public final HashMap<Integer, Integer> tickValueReverseMapping = new HashMap<>();

        /**
         * Minimum tick value found.
         *
         */
        public long minTickValue = Integer.MAX_VALUE;

        /**
         * Maximum tick value found.
         *
         */
        public long maxTickValue = Integer.MIN_VALUE;

        /**
         * Constructor for MetaData.
         *
         */
        Metadata() {
            this.maxNumberOfEncodedTicks = 25;
        }

        /**
         * Constructor for MetaData.
         *
         * @param maxNumberOfEncodedTicks maximum number of encoded ticks.
         */
        Metadata(int maxNumberOfEncodedTicks) {
            this.maxNumberOfEncodedTicks = maxNumberOfEncodedTicks;
        }

    }

    /**
     * Metadata for MIDI record.
     *
     */
    private Metadata metadata = new Metadata();

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
    public HashMap<Integer, HashMap<Integer, MMatrix>> readFile(String fileName) throws InvalidMidiDataException, IOException, MatrixException {
        ArrayList<String> fileNames = new ArrayList<>();
        fileNames.add(fileName);
        return readFile(fileNames, 1, false, true, 25);
    }

    /**
     * Reads and encodes MIDI file.
     *
     * @param fileName file name
     * @param maxEncodedTicks maximum number of encoded ticks.
     * @return encoded MIDI file
     * @throws InvalidMidiDataException if opening file fails throws exception.
     * @throws IOException if opening file fails throws exception.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public HashMap<Integer, HashMap<Integer, MMatrix>> readFile(String fileName, int maxEncodedTicks) throws InvalidMidiDataException, IOException, MatrixException {
        metadata = new Metadata(maxEncodedTicks);
        ArrayList<String> fileNames = new ArrayList<>();
        fileNames.add(fileName);
        return readFile(fileNames, 1, false, true, 25);
    }

    /**
     * Reads and encodes MIDI files.
     *
     * @param fileNames MIDI files.
     * @param numberOfInputs number of inputs.
     * @param encodeNoteOffs if true encodes note offs otherwise does not include note offs.
     * @param excludeZeroValuedEntries if true encoded values where key, velocity or tick has zero value otherwise adds all.
     * @param maxEncodedTicks maximum number of encoded ticks.
     * @return encoded inputs and outputs.
     * @throws InvalidMidiDataException if opening file fails throws exception.
     * @throws IOException if opening file fails throws exception.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public HashMap<Integer, HashMap<Integer, MMatrix>> readFile(ArrayList<String> fileNames, int numberOfInputs, boolean encodeNoteOffs, boolean excludeZeroValuedEntries, int maxEncodedTicks) throws InvalidMidiDataException, IOException, MatrixException {
        metadata = new Metadata(maxEncodedTicks);
        int noteOffOffset = encodeNoteOffs ? 1 : 0;

        HashMap<Integer, HashMap<Integer, MMatrix>> result = new HashMap<>();
        result.put(0, new HashMap<>());
        result.put(1, new HashMap<>());
        result.put(2, new HashMap<>());
        result.put(3, new HashMap<>());
        result.put(4, new HashMap<>());
        result.put(5, new HashMap<>());

        ArrayList<Integer> keyDataAsInteger = new ArrayList<>();
        ArrayList<Matrix> keyDataAsMatrix = new ArrayList<>();
        ArrayList<Integer> velocityDataAsInteger = new ArrayList<>();
        ArrayList<Matrix> velocityDataAsMatrix = new ArrayList<>();
        ArrayList<Integer> tickData = new ArrayList<>();

        for (String fileName : fileNames) {
            Sequence sequence = MidiSystem.getSequence(new File(fileName));
            if (sequence.getDivisionType() != Sequence.PPQ) {
                System.out.println("File: '" + fileName + "' omitted because Sequence must have division type of PPQ.");
                continue;
            }
            double resolutionRatio = 1 / ((double)metadata.resolution / (double)sequence.getResolution());
            Track[] tracks = sequence.getTracks();
            for (Track track : tracks) {
                long previousTick = 0;
                int trackSize = track.size();
                for (int trackIndex = 0; trackIndex < trackSize; trackIndex++) {
                    MidiEvent midiEvent = track.get(trackIndex);
                    MidiMessage midiMessage = midiEvent.getMessage();
                    if (midiMessage instanceof ShortMessage shortMessage) {
                        long tick;
                        switch (shortMessage.getCommand()) {
                            case ShortMessage.NOTE_ON -> {
                                int keyValue = shortMessage.getData1();
                                int velocityValue = shortMessage.getData2();
                                tick = midiEvent.getTick();
                                int tickDelta = (int)(resolutionRatio * (double)(tick - previousTick));
                                previousTick = tick;

                                if (!(excludeZeroValuedEntries && (keyValue == 0 || velocityValue == 0 || tickDelta == 0))) {
                                    metadata.minKeyValue = Math.min(metadata.minKeyValue, keyValue);
                                    metadata.maxKeyValue = Math.max(metadata.maxKeyValue, keyValue);
                                    keyDataAsInteger.add(keyValue + noteOffOffset);

                                    metadata.minVelocityValue = Math.min(metadata.minVelocityValue, velocityValue);
                                    metadata.maxVelocityValue = Math.max(metadata.maxVelocityValue, velocityValue);
                                    velocityDataAsInteger.add(velocityValue + noteOffOffset);

                                    metadata.minTickValue = Math.min(metadata.minTickValue, tickDelta);
                                    metadata.maxTickValue = Math.max(metadata.maxTickValue, tickDelta);
                                    tickData.add(tickDelta);
                                }
                            }
                            case ShortMessage.NOTE_OFF -> {
                                tick = midiEvent.getTick();
                                int tickDelta = (int)(resolutionRatio * (double)(tick - previousTick));
                                previousTick = tick;
                                if (encodeNoteOffs) {
                                    keyDataAsInteger.add(0);
                                    velocityDataAsInteger.add(0);
                                    tickData.add(tickDelta);
                                }
                            }
                            default -> {
                            }
                        }
                    }
                }
            }
        }

        metadata.keyOutputSize = noteOffOffset + metadata.maxKeyValue - metadata.minKeyValue + 1;
        metadata.keyBitVectorSize = ComputableMatrix.numberOfBits(metadata.keyOutputSize);
        for (int index = 0; index < keyDataAsInteger.size(); index++) {
            int keyValue = keyDataAsInteger.get(index) - (encodeNoteOffs && keyDataAsInteger.get(index) == 0 ? 0 : metadata.minKeyValue);
            keyDataAsInteger.set(index, keyValue);
            keyDataAsMatrix.add(ComputableMatrix.encodeToBitColumnVector(keyValue, metadata.keyBitVectorSize));
        }

        metadata.velocityOutputSize = noteOffOffset + metadata.maxVelocityValue - metadata.minVelocityValue + 1;
        metadata.velocityBitVectorSize = ComputableMatrix.numberOfBits(metadata.velocityOutputSize);
        for (int index = 0; index < velocityDataAsInteger.size(); index++) {
            int velocityValue = velocityDataAsInteger.get(index) - (encodeNoteOffs && velocityDataAsInteger.get(index) == 0 ? 0 : metadata.minVelocityValue);
            velocityDataAsInteger.set(index, velocityValue);
            velocityDataAsMatrix.add(ComputableMatrix.encodeToBitColumnVector(velocityValue, metadata.velocityBitVectorSize));
        }

        ArrayList<Integer> scaledTickData = new ArrayList<>();
        scaleTickData(tickData, scaledTickData, metadata);

        ArrayDeque<Integer> encodedKeyQueueAsInteger = new ArrayDeque<>();
        ArrayDeque<Matrix> encodedKeyQueueAsMatrix = new ArrayDeque<>();
        ArrayDeque<Integer> encodedVelocityQueueAsInteger = new ArrayDeque<>();
        ArrayDeque<Matrix> encodedVelocityQueueAsMatrix = new ArrayDeque<>();
        ArrayDeque<Integer> encodedTickQueue = new ArrayDeque<>();

        int pos = 0;
        for (int dataIndex = 0; dataIndex < keyDataAsInteger.size(); dataIndex++) {
            encodedKeyQueueAsInteger.addLast(keyDataAsInteger.get(dataIndex));
            encodedKeyQueueAsMatrix.addLast(keyDataAsMatrix.get(dataIndex));
            encodedVelocityQueueAsInteger.addLast(velocityDataAsInteger.get(dataIndex));
            encodedVelocityQueueAsMatrix.addLast(velocityDataAsMatrix.get(dataIndex));
            encodedTickQueue.addLast(scaledTickData.get(dataIndex));
            if (encodedKeyQueueAsInteger.size() >= numberOfInputs + 1) {
                Matrix[] keyInputMatrices = new Matrix[numberOfInputs];
                Matrix keyOutputMatrix = new DMatrix(metadata.keyOutputSize, 1);
                Iterator<Integer> keyIntegerIterator = encodedKeyQueueAsInteger.iterator();
                Iterator<Matrix> keyMatrixIterator = encodedKeyQueueAsMatrix.iterator();

                Matrix[] velocityInputMatrices = new Matrix[numberOfInputs];
                Matrix velocityOutputMatrix = new DMatrix(metadata.velocityOutputSize, 1);
                Iterator<Integer> velocityIntegerIterator = encodedVelocityQueueAsInteger.iterator();
                Iterator<Matrix> velocityMatrixIterator = encodedVelocityQueueAsMatrix.iterator();

                Matrix[] tickInputMatrices = new Matrix[numberOfInputs];
                Matrix tickOutputMatrix = new DMatrix(metadata.numberOfEncodedTicks, 1);
                Iterator<Integer> tickIterator = encodedTickQueue.iterator();

                int inputIndex = 0;
                while (keyIntegerIterator.hasNext()) {
                    int keyValue = keyIntegerIterator.next();
                    Matrix keyMatrix = keyMatrixIterator.next();

                    int velocityValue = velocityIntegerIterator.next();
                    Matrix velocityMatrix = velocityMatrixIterator.next();

                    int tickValue = tickIterator.next();

                    if (inputIndex < numberOfInputs) {
                        keyInputMatrices[inputIndex] = keyMatrix;
                        velocityInputMatrices[inputIndex] = velocityMatrix;
                        tickInputMatrices[inputIndex] = new DMatrix(metadata.numberOfEncodedTicks, 1);
                        tickInputMatrices[inputIndex].setValue(metadata.tickValueMapping.get(tickValue), 0, 1);
                    }
                    else {
                        keyOutputMatrix.setValue(keyValue, 0, 1);
                        velocityOutputMatrix.setValue(velocityValue, 0, 1);
                        tickOutputMatrix.setValue(metadata.tickValueMapping.get(tickValue), 0, 1);
                    }
                    inputIndex++;
                }

                JMatrix joinedKeyInputMatrix = new JMatrix(keyInputMatrices, true);
                JMatrix joinedVelocityInputMatrix = new JMatrix(velocityInputMatrices, true);
                JMatrix joinedTickInputMatrix = new JMatrix(tickInputMatrices, true);

                result.get(0).put(pos, new MMatrix(joinedKeyInputMatrix));
                result.get(1).put(pos, new MMatrix(keyOutputMatrix));

                result.get(2).put(pos, new MMatrix(joinedVelocityInputMatrix));
                result.get(3).put(pos, new MMatrix(velocityOutputMatrix));

                result.get(4).put(pos, new MMatrix(joinedTickInputMatrix));
                result.get(5).put(pos, new MMatrix(tickOutputMatrix));

                pos++;

                encodedKeyQueueAsInteger.removeFirst();
                encodedKeyQueueAsMatrix.removeFirst();

                encodedVelocityQueueAsInteger.removeFirst();
                encodedVelocityQueueAsMatrix.removeFirst();

                encodedTickQueue.removeFirst();
            }

        }

        return result;
    }

    /**
     * Scales tick data between zero and one.
     *
     * @param inputTickData input tick data.
     * @param outputTickData output tick data.
     * @param metadata record data.
     */
    private void scaleTickData(ArrayList<Integer> inputTickData, ArrayList<Integer> outputTickData, Metadata metadata) {

        class FrequencyValue {
            int frequency = 1;
            final int value;

            FrequencyValue(int value) {
                this.value = value;
            }

            void increment() {
                frequency++;
            }
        }

        HashMap<Integer, FrequencyValue> frequencyValueHashMap = new HashMap<>();
        ArrayList<FrequencyValue> frequencyValues = new ArrayList<>();
        ArrayList<FrequencyValue> valueFrequencies = new ArrayList<>();

        for (Integer data : inputTickData) {
            if (frequencyValueHashMap.containsKey(data)) frequencyValueHashMap.get(data).increment();
            else {
                FrequencyValue frequencyValue = new FrequencyValue(data);
                frequencyValueHashMap.put(data, frequencyValue);
                frequencyValues.add(frequencyValue);
                valueFrequencies.add(frequencyValue);
            }
        }
        frequencyValues.sort((o1, o2) -> o1.frequency == o2.frequency ? 0 : o2.frequency - o1.frequency);
        valueFrequencies.sort((o1, o2) -> o1.value == o2.value ? 0 : o2.value - o1.value);

        metadata.numberOfEncodedTicks = Math.min(metadata.maxNumberOfEncodedTicks, frequencyValues.size());

        int count = 0;
        for (FrequencyValue frequencyValue : frequencyValues) {
            metadata.tickValueMapping.put(frequencyValue.value, count);
            metadata.tickValueReverseMapping.put(count, frequencyValue.value);
            if (++count == metadata.numberOfEncodedTicks) break;
        }

        HashMap<Integer, Integer> updatedMapping = new HashMap<>();
        for (int oldIndex = metadata.numberOfEncodedTicks - 1; oldIndex < frequencyValues.size(); oldIndex++) {

            int minDistance = Integer.MAX_VALUE;
            int minIndex = -1;
            for (int newIndex = 0; newIndex < metadata.numberOfEncodedTicks; newIndex++) {
                updatedMapping.put(frequencyValues.get(newIndex).value, frequencyValues.get(newIndex).value);
                int currentDistance = Math.abs(frequencyValues.get(newIndex).value - frequencyValues.get(oldIndex).value);
                if (currentDistance > 0 && currentDistance < minDistance) {
                    minDistance = currentDistance;
                    minIndex = newIndex;
                }
            }

            updatedMapping.put(frequencyValues.get(oldIndex).value, frequencyValues.get(minIndex).value);
        }
        for (Integer data: inputTickData) outputTickData.add(updatedMapping.get(data));

    }

    /**
     * Return division type of MIDI sequence.
     *
     * @param fileName MIDI file name.
     * @return division type of MIDI sequence.
     * @throws InvalidMidiDataException if opening file fails throws exception.
     * @throws IOException if opening file fails throws exception.
     */
    public float getDivisionType(String fileName) throws InvalidMidiDataException, IOException {
        return MidiSystem.getSequence(new File(fileName)).getDivisionType();
    }

    /**
     * Returns resolution of MIDI sequence.
     *
     * @param fileName MIDI file name.
     * @return resolution of MIDI sequence.
     * @throws InvalidMidiDataException if opening file fails throws exception.
     * @throws IOException if sequence creation fails throws exception.
     */
    public int getResolution(String fileName) throws InvalidMidiDataException, IOException {
        return MidiSystem.getSequence(new File(fileName)).getResolution();
    }

    /**
     * Returns MIDI sequence based on data.
     *
     * @param dataKey MIDI key data.
     * @param dataVelocity MIDI velocity data.
     * @param dataTick MIDI tick data.
     * @param resolution resolution.
     * @param asInput if true matrices are provided as input otherwise as output.
     * @param encodeNoteOffs if true encodes note offs otherwise does not include note offs.
     * @param metadata metadata
     * @return MIDI sequence based on data.
     * @throws InvalidMidiDataException if opening file fails throws exception.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Sequence getSequence(HashMap<Integer, MMatrix> dataKey, HashMap<Integer, MMatrix> dataVelocity, HashMap<Integer, MMatrix> dataTick, int resolution, boolean asInput, boolean encodeNoteOffs, Metadata metadata) throws InvalidMidiDataException, MatrixException {
        Sequence sequence = new Sequence(Sequence.PPQ, resolution);
        Track track = sequence.createTrack();
        long currentTick = 0;
        boolean firstEntry = true;
        for (Integer index : dataKey.keySet()) {
            Matrix keyMatrix = dataKey.get(index).get(0);
            Matrix velocityMatrix = dataVelocity.get(index).get(0);
            Matrix tickMatrix = dataTick.get(index).get(0);
            int keyValue = asInput ? keyMatrix.classify().encodeToValue() : keyMatrix.argmax()[0];
            int velocityValue = asInput ? velocityMatrix.classify().encodeToValue() : velocityMatrix.argmax()[0];
            if (firstEntry) {
                ShortMessage shortMessage;
                MidiEvent midiEvent;
                shortMessage = new ShortMessage();
                shortMessage.setMessage(ShortMessage.PROGRAM_CHANGE, 0, 0);
                midiEvent = new MidiEvent(shortMessage, currentTick);
                track.add(midiEvent);
                shortMessage = new ShortMessage();
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
                track.add(midiEvent);
                firstEntry = false;
            }
            ShortMessage shortMessage = new ShortMessage();
            int shortMessageCode = encodeNoteOffs && keyValue == 0 && velocityValue == 0 ? ShortMessage.NOTE_OFF : ShortMessage.NOTE_ON;
            shortMessage.setMessage(shortMessageCode, metadata.minKeyValue + keyValue, metadata.minVelocityValue + velocityValue);
            currentTick += (long) metadata.tickValueReverseMapping.get(tickMatrix.argmax()[0]);
            MidiEvent midiEvent = new MidiEvent(shortMessage, currentTick);
            track.add(midiEvent);
        }
        return sequence;
    }

    /**
     * Plays MIDI sequence.
     *
     * @param sequence sequence
     * @param playTime play time before stopping.
     * @param wait if true waits given play time before stopping otherwise returns function after starts playing.
     * @throws MidiUnavailableException throws exception is playing fails.
     * @throws InvalidMidiDataException throws exception is playing fails.
     * @return sequencer
     */
    public Sequencer play(Sequence sequence, int playTime, boolean wait) throws MidiUnavailableException, InvalidMidiDataException {
        Sequencer sequencer = MidiSystem.getSequencer();
        sequencer.setSequence(sequence);

        stopPlaying(sequencer);

        sequencer.open();
        sequencer.start();

        if (!wait) return sequencer;

        int timeOut = 0;
        try {
            while (sequencer.isRunning() && timeOut < playTime) {
                Thread.sleep(1000);
                timeOut++;
            }
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
