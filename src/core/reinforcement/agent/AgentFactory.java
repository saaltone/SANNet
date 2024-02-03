/*
 * SANNet Neural Network Framework
 * Copyright (C) 2018 - 2024 Simo Aaltonen
 */

package core.reinforcement.agent;

import core.network.NeuralNetworkException;
import core.reinforcement.algorithm.*;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.function.NNFunctionEstimator;
import core.reinforcement.function.TabularFunctionEstimator;
import core.reinforcement.memory.Memory;
import core.reinforcement.memory.OnlineMemory;
import core.reinforcement.memory.PriorityMemory;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import core.reinforcement.policy.updateablepolicy.UpdateableQPolicy;
import core.reinforcement.policy.updateablepolicy.UpdateableSoftQPolicy;
import core.reinforcement.value.QPolicyValueFunction;
import core.reinforcement.value.SoftQValueFunction;
import utils.configurable.DynamicParamException;
import utils.matrix.MatrixException;

import java.io.IOException;

/**
 * Factory to create agents.
 *
 */
public class AgentFactory {

    /**
     * Default constructor for agent factory.
     *
     */
    public AgentFactory() {
    }

    /**
     * Agent algorithm type.
     *
     */
    public enum AgentAlgorithmType {
        /**
         * Tabular Q Learning
         *
         */
        QN,

        /**
         * Q Learning
         *
         */
        DQN,

        /**
         * Double Deep Q Learning
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
         * Deep Deterministic Policy Gradient (DDPG)
         *
         */
        DDPG,

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
     */
    public static Agent createAgent(AgentFunctionEstimator agentFunctionEstimator, AgentAlgorithmType agentAlgorithmType, Environment environment, int inputSize, int outputSize, boolean onlineMemory, boolean singleFunctionEstimator, boolean applyDueling, ExecutablePolicyType executablePolicyType, String params) throws DynamicParamException, MatrixException, IOException, ClassNotFoundException, NeuralNetworkException {
        Memory estimatorMemory = onlineMemory ? new OnlineMemory() : new PriorityMemory();

        FunctionEstimator policyFunctionEstimator;
        FunctionEstimator valueFunctionEstimator;

        boolean nnEstimator = switch (agentAlgorithmType) {
            case QN -> false;
            case DQN, DDQN, Sarsa, ActorCritic, PPO, DDPG, SACDiscrete, REINFORCE, MCTS -> true;
        };
        boolean policyGradient = switch (agentAlgorithmType) {
            case QN, DQN, DDQN, Sarsa -> false;
            case ActorCritic, PPO, DDPG, SACDiscrete, REINFORCE, MCTS -> true;
        };
        boolean stateValue = switch (agentAlgorithmType) {
            case QN, DQN, DDQN, Sarsa, DDPG, SACDiscrete, REINFORCE -> false;
            case ActorCritic, PPO, MCTS -> true;
        };
        boolean hasTargetPolicyEstimator = switch (agentAlgorithmType) {
            case QN, DQN, DDQN, Sarsa, REINFORCE, ActorCritic, PPO, SACDiscrete, MCTS -> false;
            case DDPG -> true;
        };
        boolean hasTargetValueEstimator = switch (agentAlgorithmType) {
            case QN, DQN, Sarsa, REINFORCE, ActorCritic, PPO, MCTS -> false;
            case DDQN, DDPG, SACDiscrete -> true;
        };

        if (singleFunctionEstimator) {
            // Uses single neural network estimator for both policy and value functions (works for policy gradients).
            valueFunctionEstimator = nnEstimator ? new NNFunctionEstimator(agentFunctionEstimator.buildNeuralNetwork(inputSize, policyGradient ? outputSize : -1, stateValue ? 1 : outputSize), hasTargetPolicyEstimator || hasTargetValueEstimator) : new TabularFunctionEstimator(inputSize, outputSize);
            policyFunctionEstimator = policyGradient ? valueFunctionEstimator : null;
        }
        else {
            // Uses separate estimators for value and policy functions.
            valueFunctionEstimator = nnEstimator ? new NNFunctionEstimator(agentFunctionEstimator.buildNeuralNetwork(inputSize, stateValue ? 1 : outputSize, false, applyDueling), hasTargetValueEstimator) : new TabularFunctionEstimator(inputSize, outputSize);
            policyFunctionEstimator = policyGradient ? new NNFunctionEstimator(agentFunctionEstimator.buildNeuralNetwork(inputSize, outputSize, true, false), hasTargetPolicyEstimator) : null;
        }

        return switch (agentAlgorithmType) {
            case QN, DQN -> new DQNLearning(new StateSynchronization(), environment, executablePolicyType, valueFunctionEstimator, estimatorMemory, params);
            case DDQN -> new DDQNLearning(new StateSynchronization(), environment, executablePolicyType, valueFunctionEstimator, estimatorMemory, params);
            case Sarsa -> new Sarsa(new StateSynchronization(), environment, executablePolicyType, valueFunctionEstimator, estimatorMemory, params);
            case ActorCritic -> new ActorCritic(new StateSynchronization(), environment, executablePolicyType, policyFunctionEstimator, valueFunctionEstimator, estimatorMemory, params);
            case PPO -> new PPO(new StateSynchronization(), environment, executablePolicyType, policyFunctionEstimator, valueFunctionEstimator, estimatorMemory, params);
            case DDPG -> new DDPG(new StateSynchronization(), environment, new UpdateableQPolicy(executablePolicyType, policyFunctionEstimator, estimatorMemory, params), new QPolicyValueFunction(valueFunctionEstimator, params), estimatorMemory, params);
            case SACDiscrete -> new SoftActorCriticDiscrete(new StateSynchronization(), environment, new UpdateableSoftQPolicy(executablePolicyType, policyFunctionEstimator, estimatorMemory, params), new SoftQValueFunction(valueFunctionEstimator, params), estimatorMemory, params);
            case REINFORCE -> new REINFORCE(new StateSynchronization(), environment, executablePolicyType, policyFunctionEstimator, estimatorMemory, params);
            case MCTS -> new MCTSLearning(new StateSynchronization(), environment, policyFunctionEstimator, estimatorMemory, params);
        };
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
        if (agent instanceof AbstractPolicyGradient) return ((AbstractPolicyGradient)agent).reference(sharedPolicyFunctionEstimator, sharedValueFunctionEstimator, sharedMemory);
        if (agent instanceof AbstractQLearning) return ((AbstractQLearning)agent).reference(sharedValueFunctionEstimator, sharedMemory);
        throw new AgentException("Unknown agent type. Unable to create reference for agent.");
    }

}
