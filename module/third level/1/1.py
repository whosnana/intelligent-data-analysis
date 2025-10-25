import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report

data = pd.DataFrame({
    'brand': [
        'LuxeSkin', 'BioCare', 'PureGlow', 'Natura', 'AromaLine',
        'Velvet', 'BeautyLab', 'EcoDerm', 'SkinPro', 'HerbalMix',
        'MagicTouch', 'Glamify', 'DermSoft', 'VitalSkin', 'RoseMist'
    ],
    'ad_spend': [12000, 8000, 15000, 5000, 10000, 18000, 9000, 7000, 20000, 4000, 16000, 14000, 11000, 6000, 10000],
    'units_sold': [3400, 2700, 4200, 1800, 3000, 5100, 2900, 2300, 5400, 1600, 4900, 4500, 3200, 2100, 2800],
    'revenue': [85000, 60000, 95000, 40000, 72000, 110000, 64000, 50000, 125000, 37000, 102000, 96000, 78000, 52000, 68000],
    'profit_margin': [0.32, 0.28, 0.35, 0.22, 0.30, 0.37, 0.29, 0.25, 0.39, 0.21, 0.36, 0.33, 0.31, 0.26, 0.30],
    'roi_class': ['medium', 'low', 'high', 'low', 'medium', 'high', 'medium', 'low', 'high', 'low', 'high', 'high', 'medium', 'low', 'medium']
})

feature_cols = ['ad_spend', 'units_sold', 'revenue', 'profit_margin']
target_col = 'roi_class'

X = data[feature_cols]
y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

tree = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=None,
    random_state=42
)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))

importance = pd.Series(tree.feature_importances_, index=feature_cols)
print(importance.sort_values(ascending=False))

rules = export_text(tree, feature_names=feature_cols)
print(rules)
