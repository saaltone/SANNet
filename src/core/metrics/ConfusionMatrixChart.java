/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2023 Simo Aaltonen
 */

package core.metrics;

import javax.swing.*;
import java.awt.*;
import java.io.Serial;
import java.io.Serializable;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * Implements confusion matrix chart.
 *
 */
public class ConfusionMatrixChart extends JFrame implements Serializable {

    @Serial
    private static final long serialVersionUID = 7424573839360663216L;

    /**
     * Features classified.
     *
     */
    private TreeSet<Integer> features;

    /**
     * Confusion matrix.
     *
     */
    private TreeMap<Integer, TreeMap<Integer, Integer>> confusion;

    /**
     * If true chart is activated otherwise not.
     *
     */
    private transient boolean isActivated = false;

    /**
     * Accuracy.
     *
     */
    private double classificationAccuracy;

    /**
     * Error rate.
     *
     */
    private double classificationErrorRate;

    /**
     * Precision.
     *
     */
    private double classificationPrecision;

    /**
     * Recall.
     *
     */
    private double classificationRecall;

    /**
     * Specificity.
     *
     */
    private double classificationSpecificity;

    /**
     * F1 score.
     *
     */
    private double classificationF1Score;

    /**
     * Constructor for confusion matrix chart.
     *
     */
    public ConfusionMatrixChart() {
    }

    /**
     * Updates confusion data and draws chart.
     *
     * @param features features.
     * @param confusion confusion matrix data.
     * @param classificationAccuracy classification accuracy
     * @param classificationErrorRate classification error rate
     * @param classificationPrecision classification precision
     * @param classificationRecall classification recall
     * @param classificationSpecificity classification specificity
     * @param classificationF1Score classification F1 score
     */
    public void updateConfusion(TreeSet<Integer> features, TreeMap<Integer, TreeMap<Integer, Integer>> confusion, double classificationAccuracy, double classificationErrorRate, double classificationPrecision, double classificationRecall, double classificationSpecificity, double classificationF1Score) {
        this.features = features;
        this.confusion = confusion;
        this.classificationAccuracy = classificationAccuracy;
        this.classificationErrorRate = classificationErrorRate;
        this.classificationPrecision = classificationPrecision;
        this.classificationRecall = classificationRecall;
        this.classificationSpecificity = classificationSpecificity;
        this.classificationF1Score = classificationF1Score;

        if (!isActivated) activateChart();
        else repaint();

        SwingUtilities.invokeLater(() -> setVisible(true));
    }

    /**
     * Activates chart.
     *
     */
    private void activateChart() {
        setTitle("Confusion Matrix");
        setSize(400, 500);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        JPanel confusionMatrixPanel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            drawConfusionMatrix(g);
            }
        };

        getContentPane().add(confusionMatrixPanel);
        isActivated = true;
    }

    /**
     * Draws confusion matrix chart.
     *
     * @param g graphics.
     */
    private void drawConfusionMatrix(Graphics g) {
        if (features == null || confusion == null) return;

        int numberOfFeatures = features.size();

        int blockWidth = getWidth() / (numberOfFeatures + 2);
        int effectiveBlockHeight = (int)(4 * (double)getHeight() / 5);
        int blockHeight = effectiveBlockHeight / (numberOfFeatures + 2);

        FontMetrics metrics = g.getFontMetrics();

        int maxConfusionValue = 0;
        for (int predictedRow = 0; predictedRow < numberOfFeatures; predictedRow++) {
            for (int actualRow = 0; actualRow < numberOfFeatures; actualRow++) {
                maxConfusionValue = Math.max(maxConfusionValue, getConfusionValue(predictedRow, actualRow));
            }
        }
        for (int predictedRow = 0; predictedRow < numberOfFeatures; predictedRow++) {
            for (int actualRow = 0; actualRow < numberOfFeatures; actualRow++) {
                int currentX = (actualRow + 1) * blockWidth;
                int currentY = (predictedRow + 1) * blockHeight;

                int confusionValue = getConfusionValue(predictedRow, actualRow);

                g.setColor(predictedRow == actualRow ? Color.GREEN : confusionValue != 0 ? new Color(150 + (int)(105 * (1 - (double)confusionValue / (double)maxConfusionValue)), 0, 0) : Color.LIGHT_GRAY);
                g.fillRect(currentX, currentY, blockWidth, blockHeight);

                g.setColor(Color.BLACK);
                g.drawRect(currentX, currentY, blockWidth, blockHeight);

                String confusionValueString = String.valueOf(confusionValue);
                int confusionValueStringX = currentX + blockWidth / 2 - metrics.stringWidth(confusionValueString) / 2;
                int confusionValueStringY = currentY + blockHeight / 2 + metrics.getHeight() / 2;

                g.setColor(predictedRow != actualRow && confusionValue != 0 ? Color.WHITE : Color.BLACK);
                g.drawString(confusionValueString, confusionValueStringX, confusionValueStringY);
            }
        }
        g.setColor(Color.BLACK);
        int statisticsX = getWidth() / 10;
        int statisticsXStep = getWidth() / 2;
        int yStep = (int)((double)getHeight() / 25);

        g.drawString("Accuracy:   " + String.format("%.2f", classificationAccuracy), statisticsX, effectiveBlockHeight);
        g.drawString("Error rate:  " + String.format("%.2f", classificationErrorRate), statisticsX + statisticsXStep, effectiveBlockHeight);
        g.drawString("Precision:   " + String.format("%.2f", classificationPrecision), statisticsX, effectiveBlockHeight + yStep);
        g.drawString("Recall:       " + String.format("%.2f", classificationRecall), statisticsX + statisticsXStep, effectiveBlockHeight + yStep);
        g.drawString("Specificity: " + String.format("%.2f", classificationSpecificity), statisticsX, effectiveBlockHeight + 2 * yStep);
        g.drawString("F1 score:   " + String.format("%.2f", classificationF1Score), statisticsX + statisticsXStep, effectiveBlockHeight + 2 * yStep);
    }

    /**
     * Returns confusion matrix.
     *
     * @return confusion matrix.
     */
    private TreeMap<Integer, TreeMap<Integer, Integer>> getConfusion() {
        return confusion;
    }

    /**
     * Returns specific value in confusion matrix.
     *
     * @param predictedRow predicted row.
     * @param actualRow actual row.
     * @return specific value in confusion matrix.
     */
    private int getConfusionValue(int predictedRow, int actualRow) {
        return getConfusion().get(predictedRow) == null ? 0 : getConfusion().get(predictedRow).getOrDefault(actualRow, 0);
    }

}
