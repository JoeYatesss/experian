import xgboost as xgb

# Load the model
model = xgb.Booster()
model.load_model("xgboost.json")

# Get basic model info
print("Model Info:")
print("-" * 50)
print(f"Number of trees: {len(model.get_dump())}")

# Analyze first few decision trees in detail
print("\nDetailed Decision Paths (First 2 Trees):")
print("-" * 50)
trees = model.get_dump(with_stats=True)  # Get trees with statistics
for i, tree in enumerate(trees[:2]):  # Look at first 2 trees
    print(f"\nTree {i+1}:")
    # Split the tree string into lines for better readability
    tree_lines = tree.split('\\n')
    for line in tree_lines:
        # Add indentation based on depth
        depth = line.count('\\t')
        print("    " * depth + line.strip())

def explain_prediction(features, scenario_name):
    """Make a prediction and explain feature contributions"""
    feature_names = ["number_of_open_accounts", "total_credit_limit", "total_balance", "number_of_accounts_in_arrears"]
    dmatrix = xgb.DMatrix([features], feature_names=feature_names)
    pred = model.predict(dmatrix)[0]
    contributions = model.predict(dmatrix, pred_contribs=True)[0]
    
    print(f"\n{scenario_name}")
    print("-" * 50)
    print(f"Input Features:")
    for feat, val in zip(feature_names, features):
        print(f"{feat}: {val:,}")
    
    utilization = features[2] / features[1] * 100 if features[1] > 0 else 0
    print(f"Credit Utilization: {utilization:.1f}%")
    
    print(f"\nFraud Probability: {pred:.3f} ({pred*100:.1f}%)")
    print(f"Risk Level: {'HIGH' if pred > 0.7 else 'MEDIUM' if pred > 0.3 else 'LOW'}")
    
    print("\nFeature Contributions:")
    contributions_with_names = list(zip(feature_names, contributions[:-1]))
    sorted_contributions = sorted(contributions_with_names, key=lambda x: abs(x[1]), reverse=True)
    for feat, contrib in sorted_contributions:
        impact = "+" if contrib > 0 else "-"
        print(f"{feat}: {contrib:.4f} ({impact} risk)")
    print(f"Bias term: {contributions[-1]:.4f}")

print("Key Feature Impact Tests")
print("=" * 80)

# Test 1: Arrears Impact at Different Utilization Levels
base_case = [5, 50000, 10000, 0]  # Base: 20% utilization, no arrears
high_arrears = [5, 50000, 10000, 3]  # Same but with high arrears
high_util = [5, 50000, 45000, 0]  # 90% utilization, no arrears
high_both = [5, 50000, 45000, 3]  # Both high

print("\nTest 1: Impact of Arrears vs Utilization")
for case, desc in [
    (base_case, "Base Case: Low Utilization, No Arrears"),
    (high_arrears, "High Arrears Only"),
    (high_util, "High Utilization Only"),
    (high_both, "Both High")
]:
    explain_prediction(case, desc)

# Test 2: Testing Critical Thresholds
print("\nTest 2: Critical Thresholds")
for arrears in [0, 1, 2, 3]:
    explain_prediction(
        [5, 20000, 10000, arrears],
        f"Testing Arrears Threshold: {arrears}"
    )

# Test 3: Balance Impact Test
print("\nTest 3: Balance Impact")
credit_limit = 50000
for util_pct in [10, 50, 90]:
    balance = int(credit_limit * util_pct / 100)
    explain_prediction(
        [5, credit_limit, balance, 0],
        f"Utilization at {util_pct}%"
    )

def get_all_importance_metrics():
    """Get feature importance using all available metrics"""
    metrics = {}
    
    # Weight (number of times feature appears in trees)
    metrics['weight'] = model.get_score(importance_type="weight")
    
    # Gain (improvement in accuracy brought by feature)
    metrics['gain'] = model.get_score(importance_type="gain")
    
    # Cover (number of samples affected by feature)
    metrics['cover'] = model.get_score(importance_type="cover")
    
    return metrics

def test_feature_impact(feature_idx, test_values):
    """Test how changing one feature affects predictions"""
    feature_names = ["number_of_open_accounts", "total_credit_limit", "total_balance", "number_of_accounts_in_arrears"]
    base_case = [5, 20000, 10000, 0]  # moderate values
    
    results = []
    for val in test_values:
        test_case = base_case.copy()
        test_case[feature_idx] = val
        dmatrix = xgb.DMatrix([test_case], feature_names=feature_names)
        pred = model.predict(dmatrix)[0]
        results.append((val, pred))
    
    return results

print("Detailed Feature Importance Analysis")
print("=" * 80)

# Get importance metrics
importance_metrics = get_all_importance_metrics()

print("\n1. Feature Importance by Different Metrics:")
print("-" * 50)
feature_names = ["number_of_open_accounts", "total_credit_limit", "total_balance", "number_of_accounts_in_arrears"]

for metric, scores in importance_metrics.items():
    print(f"\n{metric.upper()} importance:")
    for feat, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if metric == 'gain':
            print(f"{feat}: {score:.2f}")
        else:
            print(f"{feat}: {score}")

print("\n2. Impact Analysis - Testing Each Feature")
print("-" * 50)

# Test number_of_accounts_in_arrears
print("\nTesting number_of_accounts_in_arrears (supposedly least important):")
arrears_impact = test_feature_impact(3, [0, 1, 2, 3, 4, 5])
for val, pred in arrears_impact:
    print(f"Arrears = {val}: {pred:.3f} ({pred*100:.1f}% fraud probability)")

# Test total_balance
print("\nTesting total_balance (supposedly most important):")
balance_impact = test_feature_impact(2, [1000, 10000, 25000, 50000, 90000])
for val, pred in balance_impact:
    print(f"Balance = {val:,}: {pred:.3f} ({pred*100:.1f}% fraud probability)")

print("\n3. Decision Tree Analysis")
print("-" * 50)
trees = model.get_dump(with_stats=True)
first_tree = trees[0]
print("\nFirst decision in first tree:")
first_decision = first_tree.split('\\n')[0]
print(first_decision)

# Calculate average impact
print("\n4. Average Impact Analysis")
print("-" * 50)
arrears_change = max([pred for _, pred in arrears_impact]) - min([pred for _, pred in arrears_impact])
balance_change = max([pred for _, pred in balance_impact]) - min([pred for _, pred in balance_impact])

print(f"Maximum impact range of number_of_accounts_in_arrears: {arrears_change:.3f} ({arrears_change*100:.1f}%)")
print(f"Maximum impact range of total_balance: {balance_change:.3f} ({balance_change*100:.1f}%)")
