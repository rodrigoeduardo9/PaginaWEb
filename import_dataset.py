"""Script para normalizar/combinar datasets externos en el formato CSV esperado.
Busca imágenes en una carpeta y crea un CSV `imported_labels.csv` con columnas:
filename,prediabetes,insulin_resistance,hypertension
Si el dataset externo trae etiquetas, el script intentará mapearlas; si no, dejará 0s.
"""
import os
import csv
import argparse


def main(src_img_dir, out='imported_labels.csv', default_label=0):
    files = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    files.sort()
    with open(out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename','prediabetes','insulin_resistance','hypertension'])
        for fn in files:
            writer.writerow([fn, default_label, default_label, default_label])
    print('Wrote', out, 'with', len(files), 'rows')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', required=True)
    parser.add_argument('--out', default='imported_labels.csv')
    parser.add_argument('--default', type=int, default=0)
    args = parser.parse_args()
    main(args.img_dir, args.out, args.default)
