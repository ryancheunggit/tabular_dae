import os
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from tabular_dae.model import DAE, load


def test_dae():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'test_data', 'titanic.csv'))
    y = df['Survived']
    df.drop('Survived', axis=1, inplace=True)

    dae = DAE()
    dae.fit(df)

    features = dae.transform(df)
    classifier = RidgeClassifierCV(cv=5).fit(features, y)
    assert classifier.best_score_ > .78

    # saving
    dump_path = os.path.join(os.path.dirname(__file__), 'test_data', 'dump.pkl')
    dae.save(dump_path)

    # load
    dae2 = load(dump_path)

    # still works?
    features2 = dae2.transform(df)
    assert (features2 == features).all()