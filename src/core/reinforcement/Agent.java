package core.reinforcement;

import core.NeuralNetworkException;
import utils.MatrixException;

import java.io.IOException;

/**
 * Class that implements interface for agent.
 *
 */
public interface Agent {

    /**
     * Starts new episode.
     *
     */
    void newEpisode();

    /**
     * Takes next episode step.
     *
     */
    void nextEpisodeStep();

    /**
     * Ends episode and stores samples of episode into replay buffer.
     * Cycles QNN and updates TNN neural networks.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws IOException throws exception if cloning of Q Neural Network fails.
     * @throws ClassNotFoundException throws exception if cloning of Q Neural Network fails.
     */
    void endEpisode() throws MatrixException, NeuralNetworkException, IOException, ClassNotFoundException;

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
    boolean act(boolean forceRandomAction) throws AgentException, NeuralNetworkException, MatrixException;

    /**
     * Updates value of state action pair.<br>
     * Depending on choice uses Q Neural Network (QNN) or Target Neural Network (TNN) for target value calculation.<br>
     * Depending on choice either takes max of target state values (QNN only) or chooses action of target state with maximal value (QNN) and estimates value of this state (TNN).<br>
     * Calculates TD target using reward and target value and updates value and stores delta.<br>
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     */
    void updateValue() throws NeuralNetworkException, MatrixException;

}
