# RL_Aufgabe3

## Aufgabenverteilung 

- Beide Gruppenmitglieder haben den Code über VSCode und der Extension LiveShare gleichzeitig an dem Code gearbeitet 
- Die Dokumentation wurde dabei auch zusammen über Overleaf verfasst 

## Installationsanweisungen

### Mit pip


1. Installieren Sie die erforderlichen Pakete mit:

```bash
pip install -r requirements.txt
```

### Mit Conda

1. Stellen Sie sicher, dass Sie Anaconda oder Miniconda auf Ihrem Computer installiert haben.

2. Erstellen Sie eine neue Conda-Umgebung und installieren Sie die erforderlichen Pakete mit:

```bash
conda env create -f environment_droplet.yml
```

3. Aktivieren Sie die neu erstellte Conda-Umgebung mit:

```bash
conda activate rl
```
4. Damit es auf einem Nootebook funktioniert:
```bash
conda install -n rl ipykernel --update-deps --force-reinstall
```