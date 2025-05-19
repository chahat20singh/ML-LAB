import pandas as pd  # for manipulating the csv data
import numpy as np    # for mathematical calculation

# Load dataset
train_data_m = pd.read_csv("/content/PlayTennis.csv")

# Function to calculate total entropy of the dataset
def calc_total_entropy(train_data, label, class_list):
    total_row = train_data.shape[0]
    total_entr = 0
    for c in class_list:
        total_class_count = train_data[train_data[label] == c].shape[0]
        if total_class_count != 0:
            probability = total_class_count / total_row
            total_entr -= probability * np.log2(probability)
    return total_entr

# Function to calculate entropy for a subset of the data
def calc_entropy(feature_value_data, label, class_list):
    class_count = feature_value_data.shape[0]
    entropy = 0
    for c in class_list:
        label_class_count = feature_value_data[feature_value_data[label] == c].shape[0]
        if label_class_count != 0:
            probability_class = label_class_count / class_count
            entropy -= probability_class * np.log2(probability_class)
    return entropy

# Function to calculate information gain for a feature
def calc_info_gain(feature_name, train_data, label, class_list):
    feature_value_list = train_data[feature_name].unique()
    total_row = train_data.shape[0]
    feature_info = 0.0
    for feature_value in feature_value_list:
        feature_value_data = train_data[train_data[feature_name] == feature_value]
        feature_value_count = feature_value_data.shape[0]
        feature_value_entropy = calc_entropy(feature_value_data, label, class_list)
        feature_info += (feature_value_count / total_row) * feature_value_entropy
    return calc_total_entropy(train_data, label, class_list) - feature_info

# Find the most informative feature
def find_most_informative_feature(train_data, label, class_list):
    feature_list = train_data.columns.drop(label)
    max_info_gain = -1
    max_info_feature = None
    for feature in feature_list:
        feature_info_gain = calc_info_gain(feature, train_data, label, class_list)
        if max_info_gain < feature_info_gain:
            max_info_gain = feature_info_gain
            max_info_feature = feature
    return max_info_feature

# Generate subtree for a feature
def generate_sub_tree(feature_name, train_data, label, class_list):
    feature_value_count_dict = train_data[feature_name].value_counts(sort=False)
    tree = {}
    rows_to_remove = []

    for feature_value, count in feature_value_count_dict.items():
        feature_value_data = train_data[train_data[feature_name] == feature_value]
        assigned_to_node = False
        for c in class_list:
            class_count = feature_value_data[feature_value_data[label] == c].shape[0]
            if class_count == count:
                tree[feature_value] = c
                rows_to_remove.append(feature_value_data.index)
                assigned_to_node = True
                break
        if not assigned_to_node:
            tree[feature_value] = "?"
    train_data = train_data.drop(index=np.concatenate(rows_to_remove)) if rows_to_remove else train_data
    return tree, train_data

# Recursive tree-building function
def make_tree(root, prev_feature_value, train_data, label, class_list):
    if train_data.shape[0] != 0:
        max_info_feature = find_most_informative_feature(train_data, label, class_list)
        if max_info_feature is None:
            return
        tree, updated_train_data = generate_sub_tree(max_info_feature, train_data, label, class_list)
        next_root = None
        if prev_feature_value is not None:
            root[prev_feature_value] = {max_info_feature: tree}
            next_root = root[prev_feature_value][max_info_feature]
        else:
            root[max_info_feature] = tree
            next_root = root[max_info_feature]
        for node, branch in list(next_root.items()):
            if branch == "?":
                feature_value_data = updated_train_data[updated_train_data[max_info_feature] == node]
                make_tree(next_root, node, feature_value_data, label, class_list)

# ID3 entry point
def id3(train_data_m, label):
    train_data = train_data_m.copy()
    tree = {}
    class_list = train_data[label].unique()
    make_tree(tree, None, train_data, label, class_list)
    return tree

# Prediction function for a single instance
def predict(tree, instance):
    if not isinstance(tree, dict):
        return tree
    root_node = next(iter(tree))
    feature_value = instance[root_node]
    if feature_value in tree[root_node]:
        return predict(tree[root_node][feature_value], instance)
    else:
        return None

# Evaluate the decision tree
def evaluate(tree, test_data_m, label):
    correct_predict = 0
    wrong_predict = 0
    for index, row in test_data_m.iterrows():
        result = predict(tree, row)
        actual = row[label]
        if result == actual:
            correct_predict += 1
        else:
            wrong_predict += 1
    total = correct_predict + wrong_predict
    accuracy = correct_predict / total if total != 0 else 0
    return accuracy

# Build and evaluate tree
tree = id3(train_data_m, 'Play Tennis')
test_data_m = pd.read_csv("/content/PlayTennis.csv")
accuracy = evaluate(tree, test_data_m, 'Play Tennis')
print("Accuracy:", accuracy)
print(tree)
     
