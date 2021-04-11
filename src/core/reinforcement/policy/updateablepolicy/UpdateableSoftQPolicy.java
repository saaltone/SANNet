package core.reinforcement.policy.updateablepolicy;

import core.NeuralNetworkException;
import core.optimization.Adam;
import core.optimization.Optimizer;
import core.reinforcement.AgentException;
import core.reinforcement.function.FunctionEstimator;
import core.reinforcement.memory.StateTransition;
import core.reinforcement.policy.AbstractUpdateablePolicy;
import core.reinforcement.policy.executablepolicy.ExecutablePolicyType;
import utils.DynamicParam;
import utils.DynamicParamException;
import utils.matrix.DMatrix;
import utils.matrix.Matrix;
import utils.matrix.MatrixException;

import java.util.HashMap;

/**
 * Class that defines UpdateableSoftQPolicy.<br>
 *
 */
public class UpdateableSoftQPolicy extends AbstractUpdateablePolicy {

    /**
     * Alpha parameter for soft Q value function.
     *
     */
    private double softQAlpha = 1;

    /**
     * Alpha parameter for soft Q value function in matrix form.
     *
     */
    private final Matrix softQAlphaMatrix;

    /**
     * Gradient matrix for alpha soft Q value.
     *
     */
    private final Matrix softQAlphaMatrixGradient = new DMatrix(1,1 );

    /**
     * Cumulative alpha loss.
     *
     */
    private double cumulativeAlphaLoss = 0;

    /**
     * Update count for alpha loss.
     *
     */
    private int alphaLossCount = 0;

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
     * @throws AgentException throws exception if creation of executable policy fails.
     */
    public UpdateableSoftQPolicy(ExecutablePolicyType executablePolicyType, FunctionEstimator functionEstimator, Matrix softQAlphaMatrix) throws DynamicParamException, AgentException {
        super(executablePolicyType, functionEstimator);
        this.softQAlphaMatrix = softQAlphaMatrix;
    }

    /**
     * Returns parameters used for UpdateableSoftQPolicy.
     *
     * @return parameters used for UpdateableSoftQPolicy.
     */
    public HashMap<String, DynamicParam.ParamType> getParamDefs() {
        HashMap<String, DynamicParam.ParamType> paramDefs = new HashMap<>(super.getParamDefs());
        paramDefs.put("softQAlpha", DynamicParam.ParamType.DOUBLE);
        return paramDefs;
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
     * Returns policy gradient value for update.
     *
     * @param stateTransition state transition.
     * @return policy gradient value.
     */
    protected double getPolicyValue(StateTransition stateTransition) throws MatrixException, NeuralNetworkException {
        Matrix policyValues = functionEstimator.predict(stateTransition.environmentState.state);
        double target_entropy = -(double)stateTransition.environmentState.availableActions.size();
        cumulativeAlphaLoss += -softQAlpha * (policyValues.getValue(getAction(stateTransition.action), 0) + target_entropy) / target_entropy;
        alphaLossCount++;
        return -(valueFunction.getValue(stateTransition) - softQAlpha * Math.log(policyValues.getValue(getAction(stateTransition.action), 0)));
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
        softQAlphaMatrixGradient.setValue(0, 0, -cumulativeAlphaLoss / alphaLossCount);
        optimizer.optimize(softQAlphaMatrix, softQAlphaMatrixGradient);
        softQAlpha = softQAlphaMatrix.getValue(0, 0);
        cumulativeAlphaLoss = 0;
        alphaLossCount = 0;
    }

}
