----------------------
-- Manually install using pip
----------------------

-- Environment

-- OS    : Win64
-- Python: 3.7.4
-- pip   : 19.0.3 


-- In advance, download whl files into "%PYTHON_HOME%\venv\Scripts\whl"

%PYTHON_HOME%\venv\Scripts>pip install whl/numpy-1.18.1-cp37-cp37m-win_amd64.whl
Processing %PYTHON_HOME%\venv\scripts\whl\numpy-1.18.1-cp37-cp37m-win_amd64.whl
Installing collected packages: numpy
Successfully installed numpy-1.18.1

%PYTHON_HOME%\venv\Scripts>

cd %PYTHON_HOME%\venv\Scripts
pip install whl/six-1.14.0-py2.py3-none-any.whl

cd %PYTHON_HOME%\venv\Scripts
pip install whl/python_dateutil-2.8.1-py2.py3-none-any.whl

cd %PYTHON_HOME%\venv\Scripts
pip install whl/pyparsing-2.4.6-py2.py3-none-any.whl

cd %PYTHON_HOME%\venv\Scripts
pip install whl/cycler-0.10.0-py2.py3-none-any.whl


cd %PYTHON_HOME%\venv\Scripts
pip install whl/kiwisolver-1.1.0-cp37-none-win_amd64.whl

cd %PYTHON_HOME%\venv\Scripts
pip install whl/matplotlib-3.1.3-cp37-cp37m-win_amd64.whl
