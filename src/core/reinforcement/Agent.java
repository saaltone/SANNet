/********************************************************
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2019 Simo Aaltonen
 *
 ********************************************************/

package core.reinforcement;

import core.NeuralNetworkException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Interface for agent.
 *
 */
public interface Agent {

    /**
     * Starts new agent step and commits previous step if not yet committed.
     *
     * @param updateValue if true state action value is update prior committing step.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws IOException throws exception if cloning of Q Neural Network fails.
     * @throws ClassNotFoundException throws exception if cloning of Q Neural Network fails.
     */
    void newStep(boolean updateValue) throws MatrixException, NeuralNetworkException, IOException, ClassNotFoundException;

    /**
     * Starts new agent step and commits previous step if not yet committed.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws IOException throws exception if cloning of Q Neural Network fails.
     * @throws ClassNotFoundException throws exception if cloning of Q Neural Network fails.
     */
    void newStep() throws MatrixException, NeuralNetworkException, IOException, ClassNotFoundException;

    /**
     * Commits agent step.
     *
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws IOException throws exception if cloning of Q Neural Network fails.
     * @throws ClassNotFoundException throws exception if cloning of Q Neural Network fails.
     */
    void commitStep() throws MatrixException, NeuralNetworkException, IOException, ClassNotFoundException;

    /**
     * Commits agent step.
     *
     * @param updateValue if true updates current state action value otherwise not.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws IOException throws exception if cloning of Q Neural Network fails.
     * @throws ClassNotFoundException throws exception if cloning of Q Neural Network fails.
     */
    void commitStep(boolean updateValue) throws MatrixException, NeuralNetworkException, IOException, ClassNotFoundException;

    /**
     * Predict next action by using QNN and taking argmax of predicted values as target action.<br>
     * Predicts random action by epsilon probability (epsilon greedy policy) or if forced.<br>
     * Stores predicted state into target state variable.<br>
     *
     * @param alwaysGreedy if true greedy action is always taken. ForceRandomAction flag is omitted.
     * @param forceRandomAction if true forces to take valid random action.
     * @return returns true if action was successfully committed and executed otherwise returns false.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws IOException throws exception if cloning of Q Neural Network fails.
     * @throws ClassNotFoundException throws exception if cloning of Q Neural Network fails.
     */
    boolean act(boolean alwaysGreedy, boolean forceRandomAction) throws NeuralNetworkException, MatrixException, IOException, ClassNotFoundException;

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
