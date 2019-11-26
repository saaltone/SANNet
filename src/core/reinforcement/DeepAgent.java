/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement;

import core.NeuralNetwork;
import core.NeuralNetworkException;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;

/**
 * Class that implements agent for reinforcement learning.<br>
 * Agent acts in collaboration receiving information of environment, making decisions on actions taken and receiving rewards for quality of actions taken.<br>
 * Class uses single Q Neural Network or Double Deep Q (doubleQ) Learning method where two neural networks participate in estimating action and it's value.<br>
 * Q neural network (QNN) decides next action given specific state based on highest value unless with probability epsilon random action is taken.<br>
 * if doubleQ learning is applied Target neural network (QNN) estimates maximum value achievable given specific state and action.<br>
 * <br>
 * Reference: https://arxiv.org/pdf/1811.12560.pdf<br>
 *
 */
public class DeepAgent implements Agent, Serializable {

    private static final long serialVersionUID = -2320016939247105291L;

    /**
     * Reference to environment where agent resides in.
     *
     */
    private final Environment environment;

    /**
     * Reference to Q Neural Network.
     *
     */
    private final NeuralNetwork QNN;

    /**
     * Reference to Target Neural Network.
     *
     */
    private NeuralNetwork TNN;

    /**
     * Reference to replay buffer.
     *
     */
    private ReplayBuffer replayBuffer;

    /**
     * Cycle in episode steps for Neural Network training.
     *
     */
    private int trainCycle = 100;

    /**
     * If true uses separate target network for value estimation.
     *
     */
    private boolean doubleQ = true;

    /**
     * Update rate of Target Neural Network.
     *
     */
    private double tau = 0.001;

    /**
     * Gamma (discount) factor for target (TD) value calculation.
     *
     */
    private double gamma = 0.95;

    /**
     * Current epsilon value for epsilon greedy policy defining balance between exploration and exploitation.
     *
     */
    private double epsilon;

    /**
     * Initial epsilon value.
     *
     */
    private double epsilonInitial = 1;

    /**
     * Minimum value for epsilon.
     *
     */
    private double epsilonMin = 0;

    /**
     * Decay rate for epsilon if number of episodes is not used for epsilon decay.
     *
     */
    private double epsilonDecayRate = 0.99;

    /**
     * If true epsilon decays along episode count otherwise decays by epsilon decay rate.
     *
     */
    private boolean epsilonDecayByEpisode = true;

    /**
     * Sample as tuple (s, a, r, s') for current episode step.
     *
     */
    private transient Sample sample;

    /**
     * Total numbers of episodes.
     *
     */
    private int totalEpisodes = 0;

    /**
     * Constructor for agent.
     *
     * @param environment reference to environment where agent resides, takes actions and receives rewards.
     * @param QNN Q Neural Network to estimate agent's moves.
     */
    public DeepAgent(Environment environment, NeuralNetwork QNN) {
        this.environment = environment;
        this.QNN = QNN;
        resetReplayBuffer();
        epsilon = epsilonInitial;
    }

    /**
     * Constructor for agent.
     *
     * @param environment reference to environment where agent resides, takes actions and receives rewards.
     * @param QNN Q Neural Network (QNN) to define agent's policy and possibly value unless separate target network is not used.
     * @param params parameters used for DeepAgent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public DeepAgent(Environment environment, NeuralNetwork QNN, String params) throws DynamicParamException {
        this(environment, QNN);
        setParams(new DynamicParam(params, getParamDefs()));
        epsilon = epsilonInitial;
    }

    /**
     * Returns parameters used for DeepAgent.
     *
     * @return parameters used for DeepAgent.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("trainCycle", DynamicParam.ParamType.INT);
        paramDefs.put("doubleQ", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("tau", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("gamma", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("epsilonInitial", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("epsilonMin", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("epsilonDecayRate", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("epsilonDecayByEpisode", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("replayBufferSize", DynamicParam.ParamType.INT);
        paramDefs.put("alpha", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for DeepAgent.<br>
     * <br>
     * Supported parameters are:<br>
     *     - trainCycle: number of steps after which QNN gets trained. Default value 100.<br>
     *     - doubleQ: If true Double Deep Q Learning i.e. separate QNN and Target Neural Network (TNN) is applied otherwise just single QNN is used. Default value true.<br>
     *     - tau: Update rate of Target Neural Network. Default value 0.001.<br>
     *     - gamma: Discount value for Q learning. Default value 0.95.<br>
     *     - epsilonInitial: Initial epsilon value for greediness / randomness of learning. Default value 1.<br>
     *     - epsilonMin: Lowest value for epsilon. Default value 0.<br>
     *     - epsilonDecay: Decay rate of epsilon. Default value 0.99.<br>
     *     - epsilonDecayByEpisode: If true epsilon decays along episode count otherwise decays by epsilon decay rate. Default value true.<br>
     *     - replayBufferSize: Size of replay buffer. Default value 2000.<br>
     *     - alpha: proportional prioritization factor for samples in replay buffer. Default value 0.6.<br>
     *
     * @param params parameters used for DeepAgent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("trainCycle")) trainCycle = params.getValueAsInteger("trainCycle");
        if (params.hasParam("doubleQ")) doubleQ = params.getValueAsBoolean("doubleQ");
        if (params.hasParam("tau")) tau = params.getValueAsDouble("tau");
        if (params.hasParam("gamma")) gamma = params.getValueAsDouble("gamma");
        if (params.hasParam("epsilonInitial")) epsilonInitial = params.getValueAsDouble("epsilonInitial");
        if (params.hasParam("epsilonMin")) epsilonMin = params.getValueAsDouble("epsilonMin");
        if (params.hasParam("epsilonDecayRate")) epsilonDecayRate = params.getValueAsDouble("epsilonDecayRate");
        if (params.hasParam("epsilonDecayByEpisode")) epsilonDecayByEpisode = params.getValueAsBoolean("epsilonDecayByEpisode");
        if (params.hasParam("replayBufferSize")) replayBuffer.setSize(params.getValueAsInteger("replayBufferSize"));
        if (params.hasParam("alpha")) replayBuffer.setAlpha(params.getValueAsDouble("alpha"));
    }

    /**
     * Returns current epsilon value.
     *
     * @return current epsilon value.
     */
    public double getEpsilon() {
        return epsilon;
    }

    /**
     * Resets replay buffer with default size.
     *
     */
    public void resetReplayBuffer() {
        replayBuffer = new ReplayBuffer();
    }

    /**
     * Resets replay buffer with given size.
     *
     * @param size new size of replay buffer after reset.
     */
    public void resetReplayBuffer(int size) {
        replayBuffer = new ReplayBuffer(size);
    }

    /**
     * Starts agent.
     *
     * @throws NeuralNetworkException throws exception if start of QNN and / or TNN fails.
     * @throws IOException throws exception if copying of QNN as TNN fails.
     * @throws ClassNotFoundException throws exception if copying of QNN as TNN fails.
     */
    public void start() throws NeuralNetworkException, IOException, ClassNotFoundException {
        QNN.start();
        if (doubleQ) {
            TNN = QNN.copy();
            TNN.start();
        }
    }

    /**
     * Stops agent and QNN and TNN.
     *
     */
    public void stop() {
        QNN.stop();
        if (doubleQ) TNN.stop();
    }

    /**
     * Returns action taken based on policy.
     *
     * @return action taken based on policy.
     */
    public int getAction() {
        return sample.action;
    }

    /**
     * Starts new episode. Resets sequence by default.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void newEpisode() throws NeuralNetworkException, MatrixException {
        newEpisode(true);
    }

    /**
     * Starts new episode.
     *
     * @param resetSequence if true resets sequence by setting current sample to null.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void newEpisode(boolean resetSequence) throws NeuralNetworkException, MatrixException {
        totalEpisodes++;
        if (resetSequence) sample = null;
        cycle();
        if (epsilon > epsilonMin) {
            if (epsilonDecayByEpisode) epsilon = epsilonInitial / (double)totalEpisodes;
            else epsilon *= epsilonDecayRate;
        }
    }

    /**
     * Begins new episode step for agent.
     *
     * @param commitPreviousStep if true commits previous step prior starting new step.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void newStep(boolean commitPreviousStep) throws MatrixException {
        Matrix state = environment.getState();
        if (sample != null) sample.nextState = state;
        if (commitPreviousStep) commitStep();
        sample = new Sample(state);
    }

    /**
     * Commits episode step and adds it into replay buffer.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void commitStep() throws MatrixException {
        if (sample != null) replayBuffer.add(sample);
    }

    /**
     * Executes policy taking available action with highest value (exploitation) unless random action is selected (exploration).<br>
     * Chooses random policy by probability epsilon or if forced.<br>
     * Requests environment to execute chosen action.<br>
     *
     * @param alwaysGreedy if true greedy action is always taken. ForceRandomAction flag is omitted.
     * @param forceRandomAction if true forces to take valid random action.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void executePolicy(boolean alwaysGreedy, boolean forceRandomAction) throws NeuralNetworkException, MatrixException {
        sample.action = -1;
        if ((Math.random() < epsilon || forceRandomAction) && !alwaysGreedy) {
            sample.action = environment.requestAction(this);
        }
        else {
            ArrayList<Integer> availableActions = environment.getAvailableActions();
            Matrix values = QNN.predict(sample.state);
            double maxValidValue = Double.NEGATIVE_INFINITY;
            for (int row = 0; row < values.getRows(); row++) {
                double value = values.getValue(row, 0);
                if (availableActions.contains(row)) {
                    if (maxValidValue  < value || maxValidValue == Double.NEGATIVE_INFINITY) {
                        maxValidValue = value;
                        sample.action = row;
                    }
                }
            }
        }
        environment.commitAction(this, sample.action);
    }

    /**
     * Sets immediate reward for episode step after agent has executed policy.
     *
     * @param reward reward
     */
    public void setReward(double reward) {
        sample.reward = reward;
    }

    /**
     * Updates value of state action pair.<br>
     * Depending on choice uses Q Neural Network (QNN) or Target Neural Network (TNN) for target value calculation.<br>
     * Depending on choice either takes max of target state values (QNN only) or in Double Q mode chooses action of target state with maximal value (QNN) and estimates value of this state (TNN).<br>
     * Calculates TD target using immediate reward and target value and updates delta value (priority) in replay buffer.<br>
     *
     * @param targetSample target sample whose value is to be updated.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    private Matrix updateValue(Sample targetSample) throws NeuralNetworkException, MatrixException {
        Matrix values = QNN.predict(targetSample.state);
        double targetValue = 0;
        if (targetSample.nextState != null) {
            if (doubleQ) targetValue = TNN.predict(targetSample.nextState).getValue(QNN.predict(targetSample.nextState).argmax()[0], 0);
            else targetValue = QNN.predict(targetSample.nextState).max();
        }
        targetValue = targetSample.reward + gamma * targetValue;
        replayBuffer.updatePriority(targetSample, targetValue - values.getValue(targetSample.action, 0));
        values.setValue(targetSample.action, 0, targetValue);
        return values;
    }

    /**
     * Cycles agent.<br>
     * Trains QNN with QNN training cycle using samples from replay buffer with updated value for experience replay.<br>
     * Updates TNN if Double Q learning is applied.<br>
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     */
    private void cycle() throws MatrixException, NeuralNetworkException {
        if (doubleQ) TNN.append(QNN, tau);
        if (totalEpisodes % trainCycle == 0) {
            LinkedHashMap<Integer, Matrix> states = new LinkedHashMap<>();
            LinkedHashMap<Integer, Matrix> values = new LinkedHashMap<>();
            HashMap<Integer, Sample> samples = replayBuffer.getSamples(QNN.getTrainingSamplesPerStep());
            for (Integer sampleIndex : samples.keySet()) {
                Sample targetSample = samples.get(sampleIndex);
                Matrix targetValues = updateValue(targetSample);
                replayBuffer.update(targetSample);
                states.put(sampleIndex, targetSample.state);
                values.put(sampleIndex, targetValues);
            }
            QNN.train(states, values);
        }
    }

}
