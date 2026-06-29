# Demo rápido: modelo sintético para prueba

Este demo crea etiquetas sintéticas a partir de las imágenes en `acne_vulgaris/`, entrena un modelo multi-label ligero y expone un endpoint Flask `/predict` donde puedes subir una foto y recibir probabilidades para `prediabetes`, `insulin_resistance` y `hypertension`.

Advertencia: Esto es un prototipo con etiquetas sintéticas para probar el flujo end-to-end. No es clínicamente válido.

Pasos rápidos:

1. Generar etiquetas sintéticas:

```bash
python generate_labels.py --img-dir ..\acne_vulgaris --out labels.csv --seed 42
```

2. Instalar dependencias:

```bash
python -m pip install -r requirements.txt
```

3. Entrenar (rápido):

```bash
python train.py --csv labels.csv --img-dir ..\acne_vulgaris --out-dir . --epochs 3 --batch-size 8
```

4. Ejecutar servidor:

```bash
python app.py
```

5. Probar inferencia:

```bash
curl -F "file=@C:/ruta/a/foto.jpg" http://127.0.0.1:5000/predict
```
