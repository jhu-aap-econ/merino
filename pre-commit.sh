jupytext --set-formats notebooks//ipynb,markdown//md,scripts//py notebooks/*.ipynb
ruff check --fix --select ALL notebooks/*.ipynb
ruff format notebooks/*.ipynb
jupytext --set-formats notebooks//ipynb,markdown//md,scripts//py notebooks/*.ipynb
