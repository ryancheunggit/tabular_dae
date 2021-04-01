import os
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from tabular_dae.model import DAE


def test_dae():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'test_data', 'titanic.csv'))
    y = df['Survived']
    df.drop('Survived', axis=1, inplace=True)

    dae = DAE()
    dae.fit(df)

    features = dae.transform(df)
    classifier = RidgeClassifierCV(cv=5).fit(features, y)
    assert classifier.best_score_ > .78