# check.py
import os

checks = {
    'models/valve_model.pkl'      : os.path.exists('models/valve_model.pkl'),
    'data/processed/X_train.csv'  : os.path.exists('data/processed/X_train.csv'),
    'data/processed/X_test.csv'   : os.path.exists('data/processed/X_test.csv'),
    'src/api/app.py'              : os.path.exists('src/api/app.py'),
    'src/data/load_data.py'       : os.path.exists('src/data/load_data.py'),
    'src/data/preprocess.py'      : os.path.exists('src/data/preprocess.py'),
    'src/__init__.py'             : os.path.exists('src/__init__.py'),
    'src/data/__init__.py'        : os.path.exists('src/data/__init__.py'),
    'src/api/__init__.py'         : os.path.exists('src/api/__init__.py'),
}

for path, ok in checks.items():
    status = 'OK      ' if ok else 'MANQUANT'
    print(f'{status}  {path}')