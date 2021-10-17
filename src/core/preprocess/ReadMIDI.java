/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2021 Simo Aaltonen
 */

package core.preprocess;

import utils.matrix.MMatrix;
import utils.matrix.Matrix;
import utils.matrix.SMatrix;

import javax.sound.midi.*;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;

/**
 * Class that reads, encodes, plays and writes out MIDI file.<br>
 *
 */
public class ReadMIDI {

    /**
     * Reads and encodes MIDI file.
     *
     * @param fileName MIDI file.
     * @return encoded inputs and outputs.
     * @throws InvalidMidiDataException if opening file fails throws exception.
     * @throws IOException if opening file fails throws exception.
     */
    public static HashMap<Integer, LinkedHashMap<Integer, MMatrix>> readFile(String fileName) throws InvalidMidiDataException, IOException {
        HashSet<String> fileNames = new HashSet<>();
        fileNames.add(fileName);
        return readFile(fileNames);
    }

    /**
     * Reads and encodes MIDI files.
     *
     * @param fileNames MIDI files.
     * @return encoded inputs and outputs.
     * @throws InvalidMidiDataException if opening file fails throws exception.
     * @throws IOException if opening file fails throws exception.
     */
    public static HashMap<Integer, LinkedHashMap<Integer, MMatrix>> readFile(HashSet<String> fileNames) throws InvalidMidiDataException, IOException {
        HashMap<Integer, LinkedHashMap<Integer, MMatrix>> result = new LinkedHashMap<>();
        result.put(0, new LinkedHashMap<>());
        result.put(1, new LinkedHashMap<>());
        result.put(2, new LinkedHashMap<>());
        result.put(3, new LinkedHashMap<>());
        result.put(4, new LinkedHashMap<>());
        result.put(5, new LinkedHashMap<>());
        for (String fileName : fileNames) {
            Sequence sequence = MidiSystem.getSequence(new File(fileName));
            Track[] tracks = sequence.getTracks();
            for (Track track : tracks) {
                long previousTick = 0;
                int trackSize = track.size();
                for (int trackIndex = 0; trackIndex < trackSize; trackIndex++) {
                    MidiEvent midiEvent = track.get(trackIndex);
                    MidiMessage midiMessage = midiEvent.getMessage();
                    if (midiMessage instanceof ShortMessage shortMessage) {
                        SMatrix noteData1;
                        SMatrix noteData2;
                        SMatrix noteData3;
                        long tick;
                        switch (shortMessage.getCommand()) {
                            case ShortMessage.NOTE_ON -> {
                                noteData1 = new SMatrix(128, 1);
                                noteData1.setValue(shortMessage.getData1(), 0, 1);
                                result.get(0).put(result.get(0).size(), new MMatrix(noteData1));
                                noteData2 = new SMatrix(128, 1);
                                noteData2.setValue(shortMessage.getData2(), 0, 1);
                                result.get(2).put(result.get(2).size(), new MMatrix(noteData2));
                                tick = midiEvent.getTick();
                                noteData3 = new SMatrix(1, 1);
                                noteData3.setValue(0, 0, (double) (tick - previousTick));
                                result.get(4).put(result.get(4).size(), new MMatrix(noteData3));
                                previousTick = tick;
                            }
                            case ShortMessage.NOTE_OFF -> {
                                noteData1 = new SMatrix(128, 1);
                                noteData1.setValue(shortMessage.getData1(), 0, -1);
                                result.get(0).put(result.get(0).size(), new MMatrix(noteData1));
                                noteData2 = new SMatrix(128, 1);
                                noteData2.setValue(shortMessage.getData2(), 0, -1);
                                result.get(2).put(result.get(2).size(), new MMatrix(noteData2));
                                tick = midiEvent.getTick();
                                noteData3 = new SMatrix(1, 1);
                                noteData3.setValue(0, 0, (double) (tick - previousTick));
                                result.get(4).put(result.get(4).size(), new MMatrix(noteData3));
                                previousTick = tick;
                            }
                            default -> {
                            }
                        }
                    }
                }
            }
        }
        for (Integer index : result.get(0).keySet()) {
            if (result.get(0).get(index + 1) != null) {
                result.get(1).put(index, result.get(0).get(index + 1));
            }
            else result.get(0).remove(index);
        }
        for (Integer index : result.get(2).keySet()) {
            if (result.get(2).get(index + 1) != null) {
                result.get(3).put(index, result.get(2).get(index + 1));
            }
            else result.get(2).remove(index);
        }
        for (Integer index : result.get(4).keySet()) {
            if (result.get(4).get(index + 1) != null) {
                result.get(5).put(index, result.get(4).get(index + 1));
            }
            else result.get(4).remove(index);
        }
        return result;
    }

    /**
     * Scales tick data between zero and one.
     *
     * @param inputData input data.
     * @param outputData output data.
     * @param capValue value that sets capped maximum value (useful against data outliers).
     * @return maximum value found in data.
     */
    public static long scaleTickData(LinkedHashMap<Integer, MMatrix> inputData, LinkedHashMap<Integer, MMatrix> outputData, long capValue) {
        double maxValue = Double.NEGATIVE_INFINITY;
        double cappedValue = (double)capValue;
        for (Integer index : inputData.keySet()) {
            maxValue = Math.max(maxValue, inputData.get(index).get(0).getValue(0, 0));
            maxValue = Math.max(maxValue, outputData.get(index).get(0).getValue(0, 0));
            maxValue = Math.min(cappedValue, maxValue);
        }
        HashSet<Matrix> scaledMatrices = new HashSet<>();
        for (Integer index : inputData.keySet()) {
            Matrix inputMatrix = inputData.get(index).get(0);
            if (!scaledMatrices.contains(inputMatrix)) {
                inputMatrix.setValue(0, 0, Math.min(inputMatrix.getValue(0, 0), maxValue));
                inputMatrix.setValue(0, 0, inputMatrix.getValue(0, 0) / maxValue);
                scaledMatrices.add(inputMatrix);
            }
            Matrix outputMatrix = outputData.get(index).get(0);
            if (!scaledMatrices.contains(outputMatrix)) {
                outputMatrix.setValue(0, 0, Math.min(outputMatrix.getValue(0, 0), maxValue));
                outputMatrix.setValue(0, 0, outputMatrix.getValue(0, 0) / maxValue);
                scaledMatrices.add(outputMatrix);
            }
        }
        return (long)maxValue;
    }

    /**
     * Return division type of MIDI sequence.
     *
     * @param fileName MIDI file name.
     * @return division type of MIDI sequence.
     * @throws InvalidMidiDataException if opening file fails throws exception.
     * @throws IOException if opening file fails throws exception.
     */
    public static float getDivisionType(String fileName) throws InvalidMidiDataException, IOException {
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
    public static int getResolution(String fileName) throws InvalidMidiDataException, IOException {
        return MidiSystem.getSequence(new File(fileName)).getResolution();
    }

    /**
     * Returns MIDI sequence based on data.
     *
     * @param dataKey MIDI key data.
     * @param dataVelocity MIDI velocity data.
     * @param dataTick MIDI tick data.
     * @param divisionType division type
     * @param resolution resolution.
     * @param scalingFactor scaling factor.
     * @return MIDI sequence based on data.
     * @throws InvalidMidiDataException if opening file fails throws exception.
     */
    public static Sequence getSequence(LinkedHashMap<Integer, MMatrix> dataKey, LinkedHashMap<Integer, MMatrix> dataVelocity, LinkedHashMap<Integer, MMatrix> dataTick, float divisionType, int resolution, long scalingFactor) throws InvalidMidiDataException {
        Sequence sequence = new Sequence(divisionType, resolution);
        Track track = sequence.createTrack();
        long currentTick = 0;
        boolean firstEntry = true;
        for (Integer index : dataKey.keySet()) {
            Matrix record1 = dataKey.get(index).get(0);
            Matrix record2 = dataVelocity.get(index).get(0);
            Matrix record3 = dataTick.get(index).get(0);
            int maxData1Pos = -1;
            double maxData1Value = Double.NEGATIVE_INFINITY;
            int record1Rows = record1.getRows();
            for (int pos = 0; pos < record1Rows; pos++) {
                double currentValue = record1.getValue(pos, 0);
                if (maxData1Value < currentValue) {
                    maxData1Value = currentValue;
                    maxData1Pos = pos;
                }
            }
            int maxData2Pos = -1;
            double maxData2Value = Double.NEGATIVE_INFINITY;
            int record2Rows = record2.getRows();
            for (int pos = 0; pos < record2Rows; pos++) {
                double currentValue = record2.getValue(pos, 0);
                if (maxData2Value < currentValue) {
                    maxData2Value = currentValue;
                    maxData2Pos = pos;
                }
            }
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
            shortMessage.setMessage((maxData1Value >0 && maxData2Value > 0) ? ShortMessage.NOTE_ON : ShortMessage.NOTE_OFF, Math.max(0, maxData1Pos), Math.max(0, maxData2Pos));
            currentTick += (long)(record3.getValue(0, 0) * (double)scalingFactor);
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
    public static Sequencer play(Sequence sequence, int playTime, boolean wait) throws MidiUnavailableException, InvalidMidiDataException {
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
    public static void stopPlaying(Sequencer sequencer) {
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
    public static void writeMIDI(Sequence sequence, String fileName, int version) throws IOException {
        int midiType = MidiSystem.getMidiFileTypes(sequence)[0];
        MidiSystem.write(sequence, midiType, new File(fileName + "_" + version + ".mid"));
    }

}
