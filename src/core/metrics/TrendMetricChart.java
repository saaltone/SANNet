/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.metrics;

import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import javax.swing.*;
import java.awt.*;
import java.io.Serial;
import java.io.Serializable;
import java.util.Map;
import java.util.TreeMap;

/**
 * Implements trend metric chart.
 *
 */
public class TrendMetricChart extends JFrame implements Serializable {

    @Serial
    private static final long serialVersionUID = 5855370308326266452L;

    /**
     * Chart title.
     *
     */
    private final String chartTitle;

    /**
     * Name of X- axis.
     *
     */
    private final String xAxisName;

    /**
     * Name of Y- axis.
     *
     */
    private final String yAxisName;

    /**
     * Reported errors.
     *
     */
    private final TreeMap<Integer, Matrix> errors = new TreeMap<>();

    /**
     * If true chart is activated otherwise not.
     *
     */
    private transient boolean isActivated = false;

    /**
     * Width of chart in pixels.
     *
     */
    private int width;

    /**
     * Height of chart in pixels.
     *
     */
    private int height;

    /**
     * Position of chart left edge in pixels.
     *
     */
    private int xStart;

    /**
     * Position of chart right edge in pixels.
     *
     */
    private int xEnd;

    /**
     * X step in pixels.
     *
     */
    private int xStep;

    /**
     * Position of chart top edge in pixels.
     *
     */
    private int yStart;

    /**
     * Position of chart bottom edge in pixels.
     *
     */
    private int yEnd;

    /**
     * Y step in pixels.
     *
     */
    private int yStep;

    /**
     * Scaling factor on Y scale in inverse exponents of 10.
     *
     */
    private int factor = 1;

    /**
     * Maximum number of errors show in chart
     *
     */
    private final int maxErrors = 100;

    /**
     * Minimum scaling factor on Y scale in inverse exponents of 10.
     *
     */
    private final int minFactor = 1;

    /**
     * Maximum scaling factor on Y scale in inverse exponents of 10.
     *
     */
    private final int maxFactor = 3;

    /**
     * Maximum steps in Y scale.
     *
     */
    private final int maxYSteps = 10;

    /**
     * x and y offset in pixels.
     *
     */
    private final int xyOffset = 10;

    /**
     * Constructor for trend metric chart
     *
     * @param chartTitle chart title.
     * @param xAxisName X- axis name.
     * @param yAxisName Y- axis name.
     */
    public TrendMetricChart(String chartTitle, String xAxisName, String yAxisName) {
        this.chartTitle = chartTitle;
        this.xAxisName = xAxisName;
        this.yAxisName = yAxisName;
    }

    /**
     * Activates chart in screen.
     *
     */
    private void activateChart() {
        setTitle(chartTitle);
        setSize(800, 400);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        JPanel chartPanel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                try {
                    drawErrorChart(g);
                } catch (MatrixException e) {
                    throw new RuntimeException(e);
                }
            }
        };

        getContentPane().add(chartPanel);

        setVisible(true);
    }

    /**
     * Updates coordinate settings.
     *
     */
    private void updateCoordinateSettings() {
        width = getWidth();
        height = getHeight();

        int xOffset = getWidth() / xyOffset;
        xStart = xOffset;
        xEnd = width - xOffset;
        xStep = xEnd - xStart;

        int yOffset = getHeight() / xyOffset;
        yStart = yOffset;
        yEnd = height - yOffset * 2;
        yStep = yEnd - yStart;
    }

    /**
     * Adds error data.
     *
     * @param iteration    iteration.
     * @param error        error value.
     */
    public void addErrorData(int iteration, double error) {
        if (!isActivated) {
            isActivated = true;
            activateChart();
        }

        SwingUtilities.invokeLater(() -> {
            if (errors.size() > maxErrors - 1) errors.remove(errors.firstKey());
            errors.put(iteration, new DMatrix(error));
            repaint();
        });
    }

    /**
     * Draws error chart
     *
     * @param g graphics.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private void drawErrorChart(Graphics g) throws MatrixException {
        updateCoordinateSettings();

        g.setColor(Color.WHITE);
        g.fillRect(0, 0, width, height);

        int iterations = errors.size();

        drawXAxis(g, iterations);
        drawYAxis(g);
        drawData(g, iterations);
    }

    /**
     * Draws X- axis
     *
     * @param g graphics.
     * @param iterations number of iterations.
     */
    private void drawXAxis(Graphics g, int iterations) {
        FontMetrics metrics = g.getFontMetrics();

        int yOffset = (int)((double)getHeight() / 20);

        g.setColor(Color.BLACK);
        int iteration = 0;
        int iterationStep = errors.size() / 10;
        for (Map.Entry<Integer, Matrix> entry : errors.entrySet()) {
            int x = getX(iteration, iterations);
            if (iterationStep == 0 || iteration % iterationStep == 0) g.drawString(Integer.toString(entry.getKey()), x - metrics.stringWidth(Integer.toString(entry.getKey())) / 2, yEnd + yOffset);
            iteration++;
        }

        g.drawString(xAxisName, xStart + xStep / 2, yEnd + 2 * yOffset);
    }

    /**
     * Draws Y- axis.
     *
     * @param g graphics.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private void drawYAxis(Graphics g) throws MatrixException {
        double maxError = Double.MIN_VALUE;
        for (Map.Entry<Integer, Matrix> entry : errors.entrySet()) maxError = maxError == Double.MIN_VALUE ? entry.getValue().mean() : Math.max(maxError, entry.getValue().mean());

        factor = 0;
        double tempMaxError = maxError;
        while (tempMaxError < 1) {
            tempMaxError *= 10;
            if (++factor >= maxFactor) break;
        }
        factor = Math.max(factor, minFactor);

        int xAxisStep = (int)((double)getWidth() / 160);
        int yAxisStep = (int)((double)getHeight() / 80);

        g.setColor(Color.BLACK);
        for (int i = 0; i <= maxYSteps; i++) {
            double yStep = i / (double)maxYSteps;
            int y = getY(yStep, 1);
            String yString = String.format("%." + factor + "f", yStep / Math.pow(10, factor - 1));

            g.drawString(yString, xStart - 4 * xAxisStep - xAxisStep * factor, y + 1);
            g.drawLine(xStart, y, xEnd, y);
        }

        g.drawString(yAxisName, xStart - 6 * xAxisStep, yStart - 3 * yAxisStep);
        g.drawLine(xStart, yStart, xStart, yEnd);
    }

    /**
     * Draws data
     *
     * @param g graphics
     * @param iterations number of iterations.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private void drawData(Graphics g, int iterations) throws MatrixException {
        g.setColor(Color.BLUE);

        int previousPosition = -1;
        double previousError = -1;

        int position = 0;
        for (Map.Entry<Integer, Matrix> entry : errors.entrySet()) {
            double error = entry.getValue().mean();
            int x = getX(position, iterations);
            int y = getY(error, factor);
            int previousX;
            int previousY;
            if (previousPosition == -1 && previousError == -1) {
                previousX = x;
                previousY = y;
            }
            else {
                previousX = getX(previousPosition, iterations);
                previousY = getY(previousError, factor);
            }
            g.drawLine(previousX, previousY, x, y);

            previousPosition = position;
            previousError = error;

            position++;
        }
    }

    /**
     * Returns X- coordinate.
     *
     * @param position position
     * @param iterations number of iterations.
     * @return X- coordinate.
     */
    private int getX(int position, int iterations) {
        return xStart + (int)(((double)position / (iterations - 1)) * xStep);
    }

    /**
     * Returns Y- coordinate.
     *
     * @param value value.
     * @param yFactor Y- factor.
     * @return Y- coordinate.
     */
    private int getY(double value, int yFactor) {
        return yStart + (int)((1 - value * Math.pow(10, yFactor - 1)) * yStep);
    }

}
