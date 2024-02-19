package core.reinforcement.policy.executablepolicy;

import core.reinforcement.agent.AgentException;
import utils.configurable.DynamicParam;
import utils.configurable.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;
import utils.matrix.UnaryFunction;

import java.util.Random;
import java.util.TreeSet;

/**
 * Implements Ornstein-Uhlenbeck process noise policy.
 * Reference: <a href="https://github.com/xkiwilabs/Multi-Agent-DDPG-using-PTtorch-and-ML-Agents/blob/master/OUNoise.py">...</a>
 *
 */
public class OUNoisePolicy extends AbstractExecutablePolicy {

    /**
     * Parameter name types for Ornstein-Uhlenbeck process policy.
     *     - numberOfActions: number of actions. Default value 1.<br>
     *
     */
    private final static String paramNameTypes = "(numberOfActions:INT)";

    /**
     * Number of actions.
     *
     */
    private int numberOfActions;

    /**
     * Mu.
     *
     */
    private double mu;

    /**
     * Mu matrix.
     *
     */
    private Matrix muMatrix;

    /**
     * Theta parameter.
     *
     */
    private double theta;

    /**
     * Sigma.
     *
     */
    private double sigma;

    /**
     * Sigma matrix.
     *
     */
    private Matrix sigmaMatrix;

    /**
     * Minimum sigma.
     *
     */
    private double minSigma;

    /**
     * Decay period.
     *
     */
    private double sigmaDecay;

    /**
     * State matrix.
     *
     */
    private Matrix state;

    /**
     * Random generator.
     *
     */
    private final Random random = new Random();

    /**
     * Gaussian random function.
     *
     */
    private final UnaryFunction gaussianRandomFunction = new UnaryFunction((value) -> value * random.nextGaussian());

    /**
     * Constructor for OUNoise policy.
     *
     * @param params parameters for OUNoise policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    OUNoisePolicy(String params) throws DynamicParamException, MatrixException {
        super(ExecutablePolicyType.OU_NOISE, params, OUNoisePolicy.paramNameTypes);
    }

    /**
     * Initializes default params.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void initializeDefaultParams() throws MatrixException {
        super.initializeDefaultParams();
        numberOfActions = 1;
        this.mu = 0.0;
        this.muMatrix = new DMatrix(numberOfActions, 1, 1);
        this.muMatrix.initializeToValue(mu);
        this.theta = 0.1;
        this.sigma = 0.5;
        this.sigmaMatrix = new DMatrix(numberOfActions, 1, 1);
        this.sigmaMatrix.initializeToValue(sigma);
        this.minSigma = 0.01;
        this.sigmaDecay = 0.9999;
        reset();
    }

    /**
     * Returns parameters used for OUNoise policy.
     *
     * @return parameters used for OUNoise policy.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + OUNoisePolicy.paramNameTypes;
    }

    /**
     * Sets parameters used for OUNoise policy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - numberOfActions: number of actions. Default value 1.<br>
     *
     * @param params parameters used for OUNoise policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
        if (params.hasParam("numberOfActions")) numberOfActions = params.getValueAsInteger("numberOfActions");
        if (numberOfActions < 1) throw new DynamicParamException("Number of actions cannot be less than 1.");
        else {
            this.muMatrix = new DMatrix(numberOfActions, 1, 1);
            this.muMatrix.initializeToValue(mu);
            this.sigmaMatrix = new DMatrix(numberOfActions, 1, 1);
            this.sigmaMatrix.initializeToValue(sigma);
        }

        try {
            reset();
        }
        catch (MatrixException matrixException) {
            throw new DynamicParamException("Initialization of parameters fail.");
        }
    }

    /**
     * Resets process.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void reset() throws MatrixException {
        state = muMatrix.copy();
        sigma = Math.max(minSigma, sigma * sigmaDecay);
        sigmaMatrix.initializeToValue(sigma);
    }

    /**
     * Samples from Ornstein-Uhlenbeck process.
     *
     * @return sampled result.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public Matrix sample() throws MatrixException {
        return state.add(muMatrix.subtract(state).multiply(theta).add(sigmaMatrix.apply(gaussianRandomFunction)));
    }

    /**
     * Returns action based on policy.
     *
     * @param stateValueSet priority queue containing action values in decreasing order.
     * @return chosen action.
     * @throws AgentException throws exception if policy fails to choose valid action.
     */
    protected int getAction(TreeSet<ActionValueTuple> stateValueSet) throws AgentException {
        int action = -1;
        double maxValue = Double.MIN_VALUE;

        Matrix ouNoise;
        try {
            ouNoise = sample();
        }
        catch (MatrixException matrixException) {
            throw new AgentException("OUNoise policy failed to choose valid action.");
        }

        for (ActionValueTuple actionValueTuple : stateValueSet) {
            double value = actionValueTuple.value() + ouNoise.getValue(actionValueTuple.action(), 0, 0);
            if (maxValue == Double.MIN_VALUE || maxValue < value) {
                maxValue = value;
                action = actionValueTuple.action();
            }
        }
        if (action == -1) throw new AgentException("OUNoise policy failed to choose valid action.");
        else return action;
    }

    /**
     * Increments policy.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void increment() throws MatrixException {
        reset();
    }

}
