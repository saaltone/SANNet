package core.reinforcement.memory;

import core.reinforcement.agent.Agent;
import core.reinforcement.agent.AgentException;

import java.util.HashSet;

/**
 * Implements common functions for memory.
 *
 */
public abstract class AbstractMemory implements Memory {

    /**
     * Agents registered for function estimator.
     *
     */
    private final HashSet<Agent> registeredAgents = new HashSet<>();

    /**
     * Agents ready for function estimator update.
     *
     */
    private final HashSet<Agent> completedAgents = new HashSet<>();

    /**
     * Constructor for memory.
     *
     */
    public AbstractMemory() {
    }

    /**
     * Registers agent for abstract function estimator.
     *
     * @param agent agent.
     */
    public void registerAgent(Agent agent) {
        registeredAgents.add(agent);
    }

    /**
     * Notifies that agent is ready to update.
     *
     * @param agent current agent.
     * @return true if all registered agents are ready to update.
     * @throws AgentException throws exception if agent is not registered for function estimator.
     */
    public boolean readyToUpdate(Agent agent) throws AgentException {
        if (!registeredAgents.contains(agent)) throw new AgentException("Agent is not registered for function estimator.");
        completedAgents.add(agent);
        boolean readyToUpdate = completedAgents.containsAll(registeredAgents);
        if (readyToUpdate) completedAgents.clear();
        return readyToUpdate;
    }

}
