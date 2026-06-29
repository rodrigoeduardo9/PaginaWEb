# Datasets públicos recomendados y cómo acceder rápido

Notas rápidas: muchos datasets médicos requieren registro y uso responsable. A continuación hay recursos útiles para imágenes relacionadas con diabetes y condiciones relacionadas.

- EyePACS / Diabetic Retinopathy (Kaggle): https://www.kaggle.com/c/diabetic-retinopathy-detection — Gran dataset de fundus/retina; buen recurso para detectar diabetes a partir de retina (requiere Kaggle account).
- Messidor: http://www.adcis.net/en/Download-Third-Party/Messidor.html — Dataset de imágenes de retina (algunos enlaces requieren registro).
- NIH Chest X-ray / CheXpert (Stanford): https://stanfordmlgroup.github.io/competitions/chexpert/ — imágenes de tórax (no diabetes, pero ejemplo de datasets clínicos abiertos).
- UK Biobank: enorme repositorio con imágenes y metadatos (acceso restringido bajo aplicación) — https://www.ukbiobank.ac.uk/ (no es descarga directa).
- MIMIC / PhysioNet: https://physionet.org/ — repositorios clínicos con datos (acceso y uso controlados).
- Kaggle general (buscar): https://www.kaggle.com/datasets?q=diabetes+images — revisar resultados; algunos contienen imágenes y metadatos útiles.

Recomendación rápida para re-entrenamiento serio (pasos):

1. Identifica datasets que contengan imágenes relevantes y etiquetas clínicas para las condiciones objetivo.
2. Regístrate y descarga los datasets autorizados (Kaggle, Messidor, PhysioNet, etc.).
3. Usa `import_dataset.py` para normalizar la estructura a `filename,prediabetes,insulin_resistance,hypertension`.
4. Si el dataset trae metadatos/labels, mapea esas columnas al formato esperado.
5. Unifica imágenes y CSVs en una sola carpeta y ejecuta `train.py` con el CSV resultante.

Si quieres, puedo intentar descargar automáticamente datasets públicos que no requieran autenticación (por ejemplo, Messidor si el enlace directo lo permite), pero para Kaggle necesitaré que generes una API token y me lo indiques (no pegas secretos aquí — te diré cómo hacerlo localmente).
