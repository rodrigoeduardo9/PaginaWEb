# Proyecto Diabetes AI

Aplicación web simple para registrar usuarios y capturar datos de salud con un diagnóstico básico de riesgo.

## Características

- Registro de usuarios
- Formulario de salud con índices corporales y datos de glucosa
- Lista de usuarios registrados
- Lista de registros de salud
- Diagnóstico de riesgo AI básico en función de glucosa, IMC y hábitos alimentarios

## Requisitos

- Python 3.9+
- MySQL 8.0+
- Base de datos `bd_salud` creada con `schema.sql`

## Instalación

1. Crear un entorno virtual:

```bash
python -m venv venv
venv\Scripts\activate
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Configurar variables de entorno (opcional):

- `DB_HOST` (predeterminado: `localhost`)
- `DB_USER` (predeterminado: `root`)
- `DB_PASSWORD` (predeterminado: `''`)
- `DB_NAME` (predeterminado: `bd_salud`)
- `FLASK_SECRET`

4. Iniciar la aplicación:

```bash
python app.py
```

5. Abrir en el navegador:

`http://127.0.0.1:5000`
