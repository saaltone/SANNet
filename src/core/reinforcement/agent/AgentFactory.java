/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2022 Simo Aaltonen
 */

package core.reinforcement.agent;

import core.network.NeuralNetwork;
import core.network.NeuralNetworkException;
import core.reinforcement.algorithm.*;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.function.NNFunctionEstimator;
import core.reinforcement.function.TabularFunctionEstimator;
import core.reinforcement.memory.Memory;
import core.reinforcement.memory.OnlineMemory;
import core.reinforcement.memory.PriorityMemory;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import utils.configurable.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Factory to create agents.
 *
 */
public class AgentFactory {

    /**
     * Agent algorithm type.
     *
     */
    public enum AgentAlgorithmType {
        /**
         * Tabular Q- Learning
         *
         */
        QN,

        /**
         * Q- Learning
         *
         */
        DQN,

        /**
         * Double Deep Q- Learning
         *
         */
        DDQN,

        /**
         * SARSA
         *
         */
        Sarsa,

        /**
         * Vanilla Actor Critic
         *
         */
        ActorCritic,

        /**
         * Proximal Policy Optimization
         *
         */
        PPO,

        /**
         * Soft Actor Critic Discrete
         *
         */
        SACDiscrete,

        /**
         * REINFORCE
         *
         */
        REINFORCE,

        /**
         * Alpha Zero like MCTS based algorithm
         *
         */
        MCTS
    }

    /**
     * Creates agent.
     *
     * @param agentFunctionEstimator agent estimator for creation of NN function estimators
     * @param agentAlgorithmType agent algorithm.
     * @param environment reference to environment
     * @param inputSize number of inputs for estimator.
     * @param outputSize number of output for estimator.
     * @param singleFunctionEstimator is true single function estimator for policy and value function is assumed otherwise separated estimators.
     * @param applyDueling if true applied dueling layer to non policy gradient network otherwise not.
     * @param executablePolicyType executable policy type.
     * @param params parameters for agent.
     * @return agent.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException throws exception if setting of dynamic parameter fails.
     * @throws IOException throws exception if cloning of neural network fails.
     * @throws ClassNotFoundException throws exception if cloning of neural network fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public static Agent createAgent(AgentFunctionEstimator agentFunctionEstimator, AgentAlgorithmType agentAlgorithmType, Environment environment, int inputSize, int outputSize, boolean singleFunctionEstimator, boolean applyDueling, ExecutablePolicyType executablePolicyType, String params) throws AgentException, DynamicParamException, MatrixException, IOException, ClassNotFoundException, NeuralNetworkException {
        Memory estimatorMemory = usesOnlineMemory(agentAlgorithmType) ? new OnlineMemory() : new PriorityMemory();
        FunctionEstimator policyEstimator;
        FunctionEstimator valueEstimator;
        boolean nnEstimator = switch (agentAlgorithmType) {
            case QN -> false;
            case DQN, DDQN, Sarsa, ActorCritic, PPO, SACDiscrete, REINFORCE, MCTS -> true;
        };
        boolean policyGradient = switch (agentAlgorithmType) {
            case QN, DQN, DDQN, Sarsa -> false;
            case ActorCritic, PPO, SACDiscrete, REINFORCE, MCTS -> true;
        };
        boolean stateValue = switch (agentAlgorithmType) {
            case QN, DQN, DDQN, Sarsa, SACDiscrete, REINFORCE -> false;
            case ActorCritic, PPO, MCTS -> true;
        };
        if (singleFunctionEstimator) {
            // Uses single neural network estimator for both policy and value functions (works for policy gradients).
            NeuralNetwork stateActionValueNN = agentFunctionEstimator.buildNeuralNetwork(inputSize, outputSize);
            policyEstimator = new NNFunctionEstimator(estimatorMemory, stateActionValueNN);
            valueEstimator = new NNFunctionEstimator(estimatorMemory, stateActionValueNN);
        }
        else {
            // Uses separate estimators for value and policy functions.
            policyEstimator = nnEstimator ? new NNFunctionEstimator(estimatorMemory, agentFunctionEstimator.buildNeuralNetwork(inputSize, outputSize, policyGradient, false, applyDueling)) : new TabularFunctionEstimator(estimatorMemory, inputSize, outputSize);
            valueEstimator = nnEstimator ? new NNFunctionEstimator(estimatorMemory, agentFunctionEstimator.buildNeuralNetwork(inputSize, outputSize, false, stateValue, applyDueling)) : new TabularFunctionEstimator(estimatorMemory, inputSize, outputSize);
        }
        return switch (agentAlgorithmType) {
            case QN, DQN -> new DQNLearning(environment, executablePolicyType, valueEstimator, params);
            case DDQN -> new DDQNLearning(environment, executablePolicyType, valueEstimator, params);
            case Sarsa -> new Sarsa(environment, executablePolicyType, valueEstimator, params);
            case ActorCritic -> new ActorCritic(environment, executablePolicyType, policyEstimator, valueEstimator, params);
            case PPO -> new PPO(environment, executablePolicyType, policyEstimator, valueEstimator, params);
            case SACDiscrete -> new SoftActorCriticDiscrete(environment, executablePolicyType, policyEstimator, valueEstimator, new DMatrix(0), params);
            case REINFORCE -> new REINFORCE(environment, executablePolicyType, policyEstimator, params);
            case MCTS -> new MCTSLearning(environment, policyEstimator, valueEstimator, params);
        };
    }

    /**
     * Returns default assumption for memory type.
     *
     * @param agentAlgorithmType agent algorithm type.
     * @return assumption for memory type.
     */
    private static boolean usesOnlineMemory(AgentAlgorithmType agentAlgorithmType) {
        return switch (agentAlgorithmType) {
            case QN, DQN, Sarsa, REINFORCE, ActorCritic, PPO, MCTS -> true;
            case DDQN, SACDiscrete -> false;
        };
    }

    /**
     * Creates agent.
     *
     * @param agentFunctionEstimator agent estimator for creation of NN function estimators
     * @param agentAlgorithmType agent algorithm.
     * @param environment reference to environment
     * @param inputSize number of inputs for estimator.
     * @param outputSize number of output for estimator.
     * @param onlineMemory if true online memory is assumed to be used otherwise priority memory.
     * @param singleFunctionEstimator is true single function estimator for policy and value function is assumed otherwise separated estimators.
     * @param applyDueling if true applied dueling layer to non policy gradient network otherwise not.
     * @param executablePolicyType executable policy type.
     * @param params parameters for agent.
     * @return agent.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws DynamicParamException throws exception if setting of dynamic parameter fails.
     * @throws IOException throws exception if cloning of neural network fails.
     * @throws ClassNotFoundException throws exception if cloning of neural network fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public static Agent createAgent(AgentFunctionEstimator agentFunctionEstimator, AgentAlgorithmType agentAlgorithmType, Environment environment, int inputSize, int outputSize, boolean onlineMemory, boolean singleFunctionEstimator, boolean applyDueling, ExecutablePolicyType executablePolicyType, String params) throws AgentException, DynamicParamException, MatrixException, IOException, ClassNotFoundException, NeuralNetworkException {
        Memory estimatorMemory = onlineMemory ? new OnlineMemory() : new PriorityMemory();
        FunctionEstimator policyEstimator;
        FunctionEstimator valueEstimator;
        boolean nnEstimator = switch (agentAlgorithmType) {
            case QN -> false;
            case DQN, DDQN, Sarsa, ActorCritic, PPO, SACDiscrete, REINFORCE, MCTS -> true;
        };
        boolean policyGradient = switch (agentAlgorithmType) {
            case QN, DQN, DDQN, Sarsa -> false;
            case ActorCritic, PPO, SACDiscrete, REINFORCE, MCTS -> true;
        };
        boolean stateValue = switch (agentAlgorithmType) {
            case QN, DQN, DDQN, Sarsa, SACDiscrete, REINFORCE -> false;
            case ActorCritic, PPO, MCTS -> true;
        };
        if (singleFunctionEstimator) {
            // Uses single neural network estimator for both policy and value functions (works for policy gradients).
            NeuralNetwork stateActionValueNN = agentFunctionEstimator.buildNeuralNetwork(inputSize, outputSize);
            policyEstimator = new NNFunctionEstimator(estimatorMemory, stateActionValueNN);
            valueEstimator = new NNFunctionEstimator(estimatorMemory, stateActionValueNN);
        }
        else {
            // Uses separate estimators for value and policy functions.
            policyEstimator = nnEstimator ? new NNFunctionEstimator(estimatorMemory, agentFunctionEstimator.buildNeuralNetwork(inputSize, outputSize, policyGradient, false, applyDueling)) : new TabularFunctionEstimator(estimatorMemory, inputSize, outputSize);
            valueEstimator = nnEstimator ? new NNFunctionEstimator(estimatorMemory, agentFunctionEstimator.buildNeuralNetwork(inputSize, outputSize, false, stateValue, applyDueling)) : new TabularFunctionEstimator(estimatorMemory, inputSize, outputSize);
        }
        return switch (agentAlgorithmType) {
            case QN, DQN -> new DQNLearning(environment, executablePolicyType, valueEstimator, params);
            case DDQN -> new DDQNLearning(environment, executablePolicyType, valueEstimator, params);
            case Sarsa -> new Sarsa(environment, executablePolicyType, valueEstimator, params);
            case ActorCritic -> new ActorCritic(environment, executablePolicyType, policyEstimator, valueEstimator, params);
            case PPO -> new PPO(environment, executablePolicyType, policyEstimator, valueEstimator, params);
            case SACDiscrete -> new SoftActorCriticDiscrete(environment, executablePolicyType, policyEstimator, valueEstimator, new DMatrix(0), params);
            case REINFORCE -> new REINFORCE(environment, executablePolicyType, policyEstimator, params);
            case MCTS -> new MCTSLearning(environment, policyEstimator, valueEstimator, params);
        };
    }

    /**
     * Returns reference to agent.
     *
     * @param agent reference agent.
     * @return reference to algorithm.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public static Agent createAgent(Agent agent) throws MatrixException, AgentException, IOException, DynamicParamException, ClassNotFoundException {
        return agent.reference();
    }

    /**
     * Returns reference to agent.
     *
     * @param agent reference agent.
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param sharedValueFunctionEstimator if true shared value function estimator is used between value functions otherwise separate value function estimator is used.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to algorithm.
     * @throws IOException throws exception if creation of target value function estimator fails.
     * @throws ClassNotFoundException throws exception if creation of target value function estimator fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws MatrixException throws exception if neural network has less output than actions.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public static Agent createAgent(Agent agent, boolean sharedPolicyFunctionEstimator, boolean sharedValueFunctionEstimator, boolean sharedMemory) throws MatrixException, AgentException, IOException, DynamicParamException, ClassNotFoundException {
        if (agent instanceof AbstractPolicyGradient) return ((AbstractPolicyGradient)agent).reference(sharedPolicyFunctionEstimator, sharedMemory);
        if (agent instanceof AbstractQLearning) return ((AbstractQLearning)agent).reference(sharedValueFunctionEstimator, sharedMemory);
        throw new AgentException("Unknown agent type. Unable to create reference for agent.");
    }

}
