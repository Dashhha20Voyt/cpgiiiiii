# Сборка блокнотов из исходников

Блокноты в `notebooks/` собраны из «плоских» Python-скриптов с разметкой ячеек:

- `# %%` — кодовая ячейка;
- `# %% [markdown]` — текстовая ячейка (строки с префиксом `# `);
- `# %% WRITEFILE x.py` — ячейка `%%writefile {CODE}/x.py`, в которую подставляется текущий модуль из `src/`.

Пересборка (из каталога `dev/`, модули берутся из `../src`):

```bash
cp ../src/*.py .
python3 build_nb.py p1_body.py ../notebooks/01_regional_clocks_real_data.ipynb p1_title.md
python3 build_nb.py p2_body.py ../notebooks/02_coherent_regions_real_data.ipynb p2_title.md
rm core.py coherence.py data_build.py hic_extract.py
```

Модули в `src/` идентичны тем, что записываются ячейками `%%writefile` в блокнотах; отдельно их можно импортировать
для повторного использования (`sys.path.insert(0, "src")`).
