package core.reinforcement.policy.updateablepolicy;

import core.network.NeuralNetworkException;
import core.optimization.Adam;
import core.optimization.Optimizer;
import core.reinforcement.agent.AgentException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.policy.Policy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

/**
 * Class that defines UpdateableSoftQPolicy.<br>
 *
 */
public class UpdateableSoftQPolicy extends AbstractUpdateablePolicy {

    /**
     * Parameter name types for UpdateableSoftQPolicy.
     *     - softQAlpha; entropy regularization coefficient. Default value 1.<br>
     *
     */
    private final static String paramNameTypes = "(softQAlpha:DOUBLE)";

    /**
     * Alpha parameter for soft Q value function.
     *
     */
    private double softQAlpha;

    /**
     * Alpha parameter for soft Q value function in matrix form.
     *
     */
    private final Matrix softQAlphaMatrix;

    /**
     * Gradient matrix for alpha soft Q value.
     *
     */
    private final Matrix softQAlphaMatrixGradient = new DMatrix(0);

    /**
     * Cumulative alpha loss.
     *
     */
    private transient double cumulativeAlphaLoss = 0;

    /**
     * Update count for alpha loss.
     *
     */
    private transient int alphaLossCount = 0;

    /**
     * Optimizer for alpha loss.
     *
     */
    private final Optimizer optimizer = new Adam();

    /**
     * Constructor for UpdateableSoftQPolicy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to FunctionEstimator.
     * @param softQAlphaMatrix reference to softQAlphaMatrix.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public UpdateableSoftQPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, Matrix softQAlphaMatrix) throws DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator);
        this.softQAlphaMatrix = softQAlphaMatrix;
        if (!softQAlphaMatrix.isScalar()) throw new AgentException("Soft Q Alpha matrix must be scalar matrix.");
    }

    /**
     * Constructor for UpdateableSoftQPolicy.
     *
     * @param executablePolicyType executable policy type.
     * @param functionEstimator reference to FunctionEstimator.
     * @param softQAlphaMatrix reference to softQAlphaMatrix.
     * @param params parameters for UpdateableSoftQPolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public UpdateableSoftQPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, Matrix softQAlphaMatrix, String params) throws DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator, params);
        this.softQAlphaMatrix = softQAlphaMatrix;
        if (!softQAlphaMatrix.isScalar()) throw new AgentException("Soft Q Alpha matrix must be scalar matrix.");
    }

    /**
     * Initializes default params.
     *
     */
    public void initializeDefaultParams() {
        softQAlpha = 1;
    }

    /**
     * Returns parameters used for UpdateableSoftQPolicy.
     *
     * @return parameters used for UpdateableSoftQPolicy.
     */
    public String getParamDefs() {
        return super.getParamDefs() + ", " + UpdateableSoftQPolicy.paramNameTypes;
    }

    /**
     * Sets parameters used for UpdateableSoftQPolicy.<br>
     * <br>
     * Supported parameters are:<br>
     *     - softQAlpha; entropy regularization coefficient. Default value 1.<br>
     *
     * @param params parameters used for UpdateableSoftQPolicy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     */
    public void setParams(DynamicParam params) throws DynamicParamException {
        super.setParams(params);
        if (params.hasParam("softQAlpha")) softQAlpha = params.getValueAsDouble("softQAlpha");
        softQAlphaMatrix.setValue(0,0, softQAlpha);
    }

    /**
     * Returns reference to policy.
     *
     * @return reference to policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public Policy reference() throws DynamicParamException, AgentException {
        return new UpdateableSoftQPolicy(executablePolicy.getExecutablePolicyType(), functionEstimator, new DMatrix(0), params);
    }

    /**
     * Returns reference to policy.
     *
     * @param sharedPolicyFunctionEstimator if true shared policy function estimator is used otherwise new policy function estimator is created.
     * @param sharedMemory if true shared memory is used between estimators.
     * @return reference to policy.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws NeuralNetworkException throws exception if optimizer is of an unknown type.
     * @throws AgentException throws exception if state action value function is applied to non-updateable policy.
     */
    public Policy reference(boolean sharedPolicyFunctionEstimator, boolean sharedMemory) throws DynamicParamException, NeuralNetworkException, AgentException {
        return new UpdateableSoftQPolicy(executablePolicy.getExecutablePolicyType(), sharedPolicyFunctionEstimator ? functionEstimator : functionEstimator.reference(sharedMemory), new DMatrix(0), params);
    }

    /**
     * Returns policy gradient value for update.
     *
     * @param stateTransition state transition.
     * @return policy gradient value.
     */
    protected double getPolicyValue(StateTransition stateTransition) throws MatrixException, NeuralNetworkException {
        Matrix policyValues = functionEstimator.predict(stateTransition.environmentState.state());
        double policyValue = policyValues.getValue(stateTransition.action, 0);
        double entropyValue = policyValues.entropy(true);
        cumulativeAlphaLoss += policyValue + entropyValue;
        alphaLossCount++;
        return -(valueFunction.getValue(stateTransition) - softQAlpha * Math.log(policyValue));
    }


    /**
     * Updates alpha for value function and updates function estimator.
     *
     * @throws NeuralNetworkException throws exception if neural network operation fails.
     * @throws MatrixException throws exception if matrix operation fails.
     * @throws DynamicParamException throws exception if parameter (params) setting fails.
     * @throws AgentException throws exception if function estimator update fails.
     */
    public void updateFunctionEstimator() throws NeuralNetworkException, MatrixException, DynamicParamException, AgentException {
        super.updateFunctionEstimator();
        // https://raw.githubusercontent.com/BY571/Deep-Reinforcement-Learning-Algorithm-Collection/master/ContinousControl/SAC.ipynb
        // self.target_entropy = -action_size  # -dim(A)
        // alpha_loss = - (self.log_alpha * (log_pis + self.target_entropy).detach()).mean()
        softQAlphaMatrixGradient.setValue(0, 0, softQAlpha * cumulativeAlphaLoss / (double)alphaLossCount);
        optimizer.optimize(softQAlphaMatrix, softQAlphaMatrixGradient);
        softQAlpha = softQAlphaMatrix.getValue(0, 0);
        System.out.println(softQAlpha);
        cumulativeAlphaLoss = 0;
        alphaLossCount = 0;
    }

}
