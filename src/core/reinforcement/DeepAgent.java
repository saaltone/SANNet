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
import utils.Matrix;
import utils.MatrixException;

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
    private Environment environment;

    /**
     * Reference to Q Neural Network.
     *
     */
    private NeuralNetwork QNN;

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
    private int trainCycle = 10;

    /**
     * If true uses separate target network for value estimation.
     *
     */
    private boolean doubleQ = true;

    /**
     * Cycle in episode steps for Target Neural Network update.
     *
     */
    private int updateTNNCycle = 30;

    /**
     * Gamma (discount) factor for target (TD) value calculation.
     *
     */
    private double gamma = 0.85;

    /**
     * Current epsilon value.
     *
     */
    private double epsilon;

    /**
     * Epsilon value for probability of exploration (random action) instead of exploitation.
     *
     */
    private double epsilonInitial = 1;

    /**
     * Minimum value for epsilon.
     *
     */
    private double epsilonMin = 0;

    /**
     * Decay rate for epsilon.
     *
     */
    private double epsilonDecayRate = 0.99;

    /**
     * If true epsilon decays along episode count otherwise decays by epsilon decay rate.
     *
     */
    private boolean epsilonDecayByEpisode = true;

    /**
     * If true invalid actions are stored into agent's replay buffer.
     *
     */
    private boolean storeInvalidActions = false;

    /**
     * Stack to store samples for an episode.
     *
     */
    private transient Stack<Sample> samples;

    /**
     * Tuple of episode step.
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
     * @param QNN Q Neural Network to estimate agent's moves.
     * @param params parameters used for DeepAgent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public DeepAgent(Environment environment, NeuralNetwork QNN, String params) throws DynamicParamException {
        this.environment = environment;
        this.QNN = QNN;
        resetReplayBuffer();
        setParams(new DynamicParam(params, getParamDefs()));
        epsilon = epsilonInitial;
    }

    /**
     * Gets parameters used for DeepAgent.
     *
     * @return parameters used for DeepAgent.
     */
    private HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>();
        paramDefs.put("trainCycle", DynamicParam.ParamType.INT);
        paramDefs.put("doubleQ", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("updateTNNCycle", DynamicParam.ParamType.INT);
        paramDefs.put("learningRate", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("gamma", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("epsilonInitial", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("epsilonMin", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("epsilonDecay", DynamicParam.ParamType.DOUBLE);
        paramDefs.put("epsilonDecayByEpisode", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("storeInvalidActions", DynamicParam.ParamType.BOOLEAN);
        paramDefs.put("replayBufferSize", DynamicParam.ParamType.INT);
        paramDefs.put("alpha", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
    }

    /**
     * Sets parameters used for DeepAgent.<br>
     * <br>
     * Supported parameters are:<br>
     *     - trainCycle: number of episodes after which QNN gets trained. Default value 10.<br>
     *     - doubleQ: If true Double Deep Q Learning (QNN and TNN) is applied otherwise just single QNN is used. Default value 0.95.<br>
     *     - updateTNNCycle: number of episodes after which TNN gets updated. Default value 10.<br>
     *     - epsilonInitial: Initial epsilon value for greediness / randomness of learning. Default value 1.<br>
     *     - epsilonMin: Lowest value for epsilon. Default value 0.<br>
     *     - epsilonDecay: Decay rate of epsilon. Default value 0.99.<br>
     *     - epsilonDecayByEpisode: If true epsilon decays along episode count otherwise decays by epsilon decay rate. Default value true.<br>
     *     - storeInvalidActions: If true If true agent records invalid actions into replay buffer. Default value true.<br>
     *     - replayBufferSize: Size of replay buffer. Default value 2000.<br>
     *     - alpha: proportional prioritization factor for samples in replay buffer. Default value 0.6.<br>
     *
     * @param params parameters used for DeepAgent.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        if (params.hasParam("trainCycle")) trainCycle = params.getValueAsInteger("trainCycle");
        if (params.hasParam("doubleQ")) doubleQ = params.getValueAsBoolean("doubleQ");
        if (params.hasParam("updateTNNCycle")) updateTNNCycle = params.getValueAsInteger("updateTNNCycle");
        if (params.hasParam("gamma")) gamma = params.getValueAsDouble("gamma");
        if (params.hasParam("epsilonInitial")) epsilonInitial = params.getValueAsDouble("epsilonInitial");
        if (params.hasParam("epsilonMin")) epsilonMin = params.getValueAsDouble("epsilonMin");
        if (params.hasParam("epsilonDecayRate")) epsilonDecayRate = params.getValueAsDouble("epsilonDecayRate");
        if (params.hasParam("epsilonDecayByEpisode")) epsilonDecayByEpisode = params.getValueAsBoolean("epsilonDecayByEpisode");
        if (params.hasParam("storeInvalidActions")) storeInvalidActions = params.getValueAsBoolean("storeInvalidActions");
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
     * Starts new episode.
     *
     */
    public void newEpisode() {
        samples = new Stack<>();
    }

    /**
     * Takes next episode step.
     *
     */
    public void nextEpisodeStep() {
        samples.push(sample = new Sample());
    }

    /**
     * Ends episode and stores samples of episode into replay buffer.
     * Cycles QNN and updates TNN neural networks.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws IOException throws exception if cloning of Q Neural Network fails.
     * @throws ClassNotFoundException throws exception if cloning of Q Neural Network fails.
     */
    public void endEpisode() throws MatrixException, NeuralNetworkException, IOException, ClassNotFoundException {
        endEpisode(true);
    }

    /**
     * Ends episode and stores samples of episode into replay buffer.
     * Cycles QNN and updates TNN neural networks.
     *
     * @param updateValue if true updates current state action value otherwise not.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws IOException throws exception if cloning of Q Neural Network fails.
     * @throws ClassNotFoundException throws exception if cloning of Q Neural Network fails.
     */
    public void endEpisode(boolean updateValue) throws MatrixException, NeuralNetworkException, IOException, ClassNotFoundException {
        if (updateValue) updateValue();
        while (!samples.empty()) replayBuffer.add(samples.pop().copy());
        cycle();
        totalEpisodes++;
        if (epsilon > epsilonMin) {
            if (epsilonDecayByEpisode) epsilon = epsilonInitial / (double)totalEpisodes;
            else epsilon *= epsilonDecayRate;
        }
    }

    /**
     * Returns action taken to move from current state to target state.
     *
     * @return action taken.
     */
    public int getAction() {
        return sample.action;
    }

    /**
     * Predict next action by using QNN and taking argmax of predicted values as target action.<br>
     * Predicts random action by epsilon probability (epsilon greedy policy) or if forced.<br>
     * Stores predicted state into target state variable.<br>
     *
     * @param forceRandomAction if true forces to take valid random action.
     * @return returns true if action was successfully committed and executed otherwise returns false.
     * @throws AgentException throws exception if there are no actions available for agent to take or action taken is not in list of available actions.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public boolean act(boolean forceRandomAction) throws AgentException, NeuralNetworkException, MatrixException {
        sample.state = environment.getState();
        sample.values = QNN.predict(sample.state);

        if (Math.random() < epsilon || forceRandomAction) {
            sample.action = environment.requestAction(this);
            sample.validAction = true;
        }
        else {
            sample.action = sample.values.argmax()[0];
            sample.validAction = environment.isValidAction(this, sample.action);
            if (!sample.validAction) {
                if (storeInvalidActions) {
                    updateValue();
                    addToReplayBuffer();
                }
                return false;
            }
        }

        environment.commitAction(this, sample.action);

        return true;
    }

    /**
     * Updates value of state action pair.<br>
     * Depending on choice uses Q Neural Network (QNN) or Target Neural Network (TNN) for target value calculation.<br>
     * Depending on choice either takes max of target state values (QNN only) or chooses action of target state with maximal value (QNN) and estimates value of this state (TNN).<br>
     * Calculates TD target using reward and target value and updates value and stores delta.<br>
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    public void updateValue() throws NeuralNetworkException, MatrixException {
        double value = sample.values.getValue(sample.action, 0);

        sample.targetState = environment.getState();
        double targetValue;
        if (!environment.isTerminalState() && sample.validAction) {
            if (doubleQ) {
                int action = QNN.predict(sample.targetState).argmax()[0];
                targetValue = TNN.predict(sample.targetState).getValue(action, 0);
            }
            else targetValue = QNN.predict(sample.targetState).max();
            sample.terminalState = false;
        }
        else {
            targetValue = 0;
            sample.terminalState = true;
        }

        sample.reward = environment.requestReward(this, sample.validAction);

        double target = sample.reward + gamma * targetValue;
        sample.delta = target - value;
        sample.values.setValue(sample.action, 0, target);
    }

    public void addToReplayBuffer() throws MatrixException {
        replayBuffer.add(sample.copy());
    }

    /**
     * Cycles agent.<br>
     * Trains QNN with QNN training cycle using replay buffer samples for experience replay.<br>
     * Copies QNN to TNN if doubleQ learning is applied.<br>
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws IOException throws exception if copying of QNN as TNN fails.
     * @throws ClassNotFoundException throws exception if copying of QNN as TNN fails.
     */
    private void cycle() throws NeuralNetworkException, IOException, ClassNotFoundException {
        if (totalEpisodes > 0) {
            if (doubleQ && totalEpisodes % updateTNNCycle == 0) {
                TNN.stop();
                TNN = QNN.copy();
                TNN.start();
            }
            if (totalEpisodes % trainCycle == 0) {
                LinkedHashMap<Integer, Matrix> states = new LinkedHashMap<>();
                LinkedHashMap<Integer, Matrix> values = new LinkedHashMap<>();
                replayBuffer.getSamples(QNN.getTrainingSamplesPerStep(), states, values);
                QNN.train(states, values);
            }
        }
    }

}
