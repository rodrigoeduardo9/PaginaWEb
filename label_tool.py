"""Herramienta rápida de etiquetado en consola.
Muestra cada imagen (abre con el visualizador por defecto) y pide 3 etiquetas 0/1.
Escribe `labels_real.csv` en el directorio actual con columnas: filename,prediabetes,insulin_resistance,hypertension
"""
import os
import csv
from PIL import Image


def main(img_dir, out='labels_real.csv'):
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    files.sort()
    if os.path.exists(out):
        print(out, 'already exists — it will be appended to. Remove to start fresh.')

    with open(out, 'a', newline='') as f:
        writer = csv.writer(f)
        if os.path.getsize(out) == 0:
            writer.writerow(['filename','prediabetes','insulin_resistance','hypertension'])
        for fn in files:
            print('Image:', fn)
            path = os.path.join(img_dir, fn)
            try:
                img = Image.open(path)
                img.show()
            except Exception as e:
                print('Cannot open image:', e)
            # Prompt labels
            vals = []
            for label in ['prediabetes','insulin_resistance','hypertension']:
                while True:
                    v = input(f'{label} (0/1, s=skip, q=quit): ').strip().lower()
                    if v in ('0','1'):
                        vals.append(int(v))
                        break
                    if v == 's':
                        vals = None
                        break
                    if v == 'q':
                        print('Quitting; saved progress to', out)
                        return
                    print('Invalid input')
                if vals is None:
                    break
            if vals is None:
                print('Skipped', fn)
                continue
            writer.writerow([fn]+vals)
            print('Saved labels for', fn)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python label_tool.py PATH_TO_IMAGE_DIR')
    else:
        main(sys.argv[1])
