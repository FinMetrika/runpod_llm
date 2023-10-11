# DATACLASS

# Upgrade pip, setuptools, wheel to the latest versions
#pip install --upgrade pip setuptools wheel

# Upgrade all the installed packages
#pip freeze --local | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U

# Install requirements needed for the fine-tuning
python -m pip install -r requirements.txt

# Run the model
python3 src/main.py
    #proba $PROBA \

# ARGPARSE
# export FILE_NAMES='["keksDejan1.csv", "keksIta.csv"]'
# python3 src/main.py \
#     --file_names "$FILE_NAMES"