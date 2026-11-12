# Oral Defense Notes

## Opening

“My project helps a Kazakhstani sole proprietor decide what to do with an incoming accounting document. The classifier estimates an initial error probability, Bayes updates it using external evidence, and the decision model compares the cost and benefit of three possible actions.”

## Walk through the code

### 1. Supervised classification

- `feature_row()` puts the document type, amount, BIN/IIN validity, ESF delay, and issuer VAT status in a fixed order. The known answer, `has_error`, is the training label. KGD and bank signals enter later.
- `train_models()` uses a stratified split to keep a similar share of errors in training and testing. The preprocessing step turns document types into separate yes/no columns and scales numeric values using the fitted mean and standard deviation.
- Logistic regression adds weighted inputs and uses a sigmoid to produce a probability. The MLP adds a hidden layer of eight units so it can learn combinations of inputs. It uses ReLU in the hidden layer and a sigmoid for the binary output.
- Each model is scored by four-fold cross-validation on the training partition. The pipeline relearns preprocessing inside each fold. Both models are fitted on the full training partition and scored on the held-out test set. The code chooses the model by training cross-validation F1.
- `classify_document()` calls `predict_proba()` and uses the probability for class 1, meaning error. For the receipt, the MLP gives 2.2923%. This is the prior for the next stage.

### 2. Bayesian updating

- `update_risk()` starts with the model's error probability and its complement. The JSON supplies how likely each observed signal is for a clean or erroneous document.
- For each state, multiply the starting probability by the likelihoods of the observed signals. This uses the course assumption that KGD and bank signals are independent once the error state is known. An unknown signal is skipped.
- Add the two weighted values to get the probability of the evidence. Divide the error weight by this total to get the posterior. For the receipt, the lack of a KGD flag and the matching bank amount bring the estimate down to 0.6902%.

### 3. One-step MDP and maximum expected utility

- `select_action()` uses two possible current states. If the posterior is `p`, the weights are `1 - p` for clean and `p` for erroneous.
- For each action and state, take the immediate gain or cost, then add the probability that the document still has an error after the action times the cost of that remaining error. For example, a supplier correction costs 2,000 KZT and leaves 25% of modeled errors unresolved.
- Multiply those state-specific values by the two state weights and add them. The code selects the action with the highest result, even if all three values are negative. Auto-approval's error penalty is already in its immediate cost.

## Live demo cues

| Scenario | Prior | Posterior | Selected action |
|---|---:|---:|---|
| Low-risk receipt | 2.29% | 0.69% | Auto-Approve |
| ESF with an 18-day delay | 54.31% | 44.74% | Request Supplier Correction |
| High-amount AVR with KGD flag and bank mismatch | 11.87% | 81.00% | Escalate to Accountant |

Point to **Input**, **Key calculation / steps**, **Output**, and **Explanation** in each stage. In the ESF example, the missing bank comparison is skipped and the absence of a KGD flag lowers the estimate. The delayed document still has enough estimated risk to warrant a correction.

## Five likely questions

**1. How did you prevent data leakage?**  
I split the data before fitting. The encoder and scaler are inside the pipeline, so each cross-validation fold fits them on its own training rows. The holdout is used for final reporting. The target and document ID are excluded from the features, and KGD is reserved for the Bayesian stage so it is used once.

**2. What is the difference between the prior and posterior?**  
The prior is the classifier's estimate before seeing the KGD or bank evidence. The posterior is the revised estimate after weighting both possible states by the likelihood of those observations and normalizing.

**3. Which activation functions does the MLP use, and why?**  
The hidden layer uses ReLU, which returns zero for a negative input and keeps a positive input. It lets the network learn patterns beyond a single weighted sum. The binary output uses a sigmoid, which converts its score into a number between zero and one.

**4. What happens if you increase the fine for approving an erroneous document?**  
Approval becomes less attractive as risk rises. With the current costs, approval ties supplier correction at about 6.45% posterior risk. If the modeled fine doubles from 50,000 to 100,000 KZT, that crossover falls to about 3.11%, holding other costs fixed. I get these values by setting `1000*(1-p) - fine*p` equal to `-2000 - 0.25*18000*p`. The Bayesian probability stays the same.

**5. Why use the MLP when logistic regression scored better on the test set?**  
The model was selected in advance by average F1 on training cross-validation, where the MLP led 0.673 to 0.606. Logistic regression did better on this particular holdout. There were only four erroneous test documents, so I would collect more labeled examples before choosing a production model.
