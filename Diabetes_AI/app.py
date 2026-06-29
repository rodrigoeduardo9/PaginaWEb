import os
import sqlite3
import json
import gc
from functools import wraps
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import mysql.connector

import torch
from PIL import Image

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET', 'proyecto_diabetes_ai_secret')

DB_TYPE = os.getenv('DB_TYPE', 'auto').lower()
DB_PORT = int(os.getenv('DB_PORT', 3306))
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'bd_salud'),
    'port': DB_PORT,
}
SQLITE_DB_PATH = os.path.join(os.path.dirname(__file__), 'bd_salud.sqlite3')
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

ACANTOSIS_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'modelo_acantosis_traced.pt')
ACANTOSIS_CLASS_PATH = os.path.join(os.path.dirname(__file__), 'class_names.json')
_acantosis_model = None
_acantosis_class_names = []

try:
    if os.path.exists(ACANTOSIS_CLASS_PATH):
        _acantosis_class_names = json.load(open(ACANTOSIS_CLASS_PATH))
except Exception as e:
    print('Error loading class names:', e)


def get_acantosis_model():
    global _acantosis_model
    if _acantosis_model is None and os.path.exists(ACANTOSIS_MODEL_PATH):
        try:
            _acantosis_model = torch.jit.load(ACANTOSIS_MODEL_PATH)
            _acantosis_model.eval()
        except Exception as e:
            print('Error loading acantosis model:', e)
            _acantosis_model = None
    return _acantosis_model


def preprocess_skin_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    x = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(x).unsqueeze(0).float()


def generar_recomendaciones_piel(clase, confianza):
    recs = {
        'Acanthosis_Nigricans': (
            "🔴 ACANTOSIS NIGRICANS DETECTADA: Esta condici\u00f3n est\u00e1 fuertemente asociada con resistencia a la insulina. "
            "Consulta a un endocrin\u00f3logo lo antes posible. Reduce el consumo de az\u00facares y carbohidratos refinados. "
            "Realiza ejercicio f\u00edsico regular. Un diagn\u00f3stico temprano puede prevenir el desarrollo de diabetes tipo 2."
        ),
        'CRP': (
            "\ud83d\udfe1 MARCADOR INFLAMATORIO (CRP) ELEVADO: La prote\u00edna C reactiva elevada indica inflamaci\u00f3n sist\u00e9mica. "
            "Podr\u00eda estar asociada a riesgo cardiovascular, diabetes o infecciones. "
            "Se recomienda una dieta antiinflamatoria (omega-3, frutas, verduras), reducir el estr\u00e9s y consultar a un m\u00e9dico general."
        ),
        'Healthy': (
            "\ud83d\udfe2 PIEL SALUDABLE: No se detectan anormalidades significativas. "
            "Contin\u00faa con tus h\u00e1bitos de cuidado personal, protecci\u00f3n solar y chequeos m\u00e9dicos peri\u00f3dicos."
        ),
        'TFFD': (
            "\ud83d\udfe1 MARCADOR TFFD DETECTADO: Este hallazgo puede estar relacionado con factores metab\u00f3licos. "
            "Se recomienda realizar un chequeo m\u00e9dico completo, incluyendo perfil gluc\u00e9mico y lip\u00eddico. "
            "Mant\u00e9n una dieta equilibrada y actividad f\u00edsica regular."
        ),
    }
    nivel = 'Alto' if confianza >= 0.6 else 'Moderado' if confianza >= 0.3 else 'Bajo'
    base = recs.get(clase, f"Resultado: {clase} con {confianza:.0%} de confianza. Consulta a tu m\u00e9dico.")
    return f"{base}\n\n\u2728 Confianza del diagn\u00f3stico: {confianza:.1%} | Nivel de riesgo: {nivel}"


def is_sqlite_connection(conn):
    return isinstance(conn, sqlite3.Connection)


def sqlite_connection():
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def db_placeholder(conn):
    return '?' if is_sqlite_connection(conn) else '%s'


def ensure_sqlite_user_schema(conn):
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(usuarios)")
    columns = [row['name'] for row in cursor.fetchall()]
    if 'password' not in columns:
        cursor.execute('ALTER TABLE usuarios ADD COLUMN password TEXT')
        conn.commit()
    cursor.close()


def init_sqlite_db():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    conn = sqlite_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS usuarios (
            id_usuario INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre_completo TEXT NOT NULL,
            correo TEXT NOT NULL UNIQUE,
            password TEXT,
            fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )'''
    )
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS registros_salud (
            id_registro INTEGER PRIMARY KEY AUTOINCREMENT,
            id_usuario INTEGER NOT NULL,
            fecha_formulario TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            peso_kg REAL NOT NULL,
            altura_cm REAL NOT NULL,
            edad INTEGER NOT NULL,
            nivel_glucosa REAL NOT NULL,
            ojos_rojos TEXT NOT NULL,
            consumo_azucares INTEGER NOT NULL,
            consumo_harinas INTEGER NOT NULL,
            imc REAL NOT NULL,
            FOREIGN KEY (id_usuario) REFERENCES usuarios (id_usuario) ON DELETE CASCADE
        )'''
    )
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS analisis_piel (
            id_analisis INTEGER PRIMARY KEY AUTOINCREMENT,
            id_usuario INTEGER NOT NULL,
            filename TEXT NOT NULL,
            clase_detectada TEXT,
            confianza REAL,
            recomendaciones TEXT,
            fecha_analisis TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (id_usuario) REFERENCES usuarios (id_usuario) ON DELETE CASCADE
        )'''
    )
    conn.commit()
    ensure_sqlite_user_schema(conn)
    conn.close()


def ensure_mysql_user_schema(conn):
    cursor = conn.cursor()
    try:
        cursor.execute("SHOW TABLES LIKE 'usuarios'")
        if cursor.fetchone():
            cursor.execute("SHOW COLUMNS FROM usuarios LIKE 'password'")
            if not cursor.fetchone():
                cursor.execute('ALTER TABLE usuarios ADD COLUMN password VARCHAR(255) NULL')
                conn.commit()
    except mysql.connector.Error:
        pass
    finally:
        cursor.close()


def get_db_connection():
    if DB_TYPE == 'sqlite':
        init_sqlite_db()
        return sqlite_connection()
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        ensure_mysql_user_schema(conn)
        return conn
    except mysql.connector.Error as err:
        if DB_TYPE == 'mysql':
            raise
        print('MySQL no disponible, usando SQLite de respaldo:', err)
        init_sqlite_db()
        return sqlite_connection()


def get_current_user():
    if 'user_id' not in session:
        return None
    return {
        'id': session.get('user_id'),
        'nombre': session.get('user_name'),
        'correo': session.get('user_email'),
    }


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if 'user_id' not in session:
            flash('Por favor inicia sesi\u00f3n para acceder a esta secci\u00f3n.', 'info')
            return redirect(url_for('login'))
        return view(*args, **kwargs)
    return wrapped_view


@app.context_processor
def inject_user():
    return dict(current_user=get_current_user())


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health')
def health():
    return {'status': 'ok', 'model_loaded': get_acantosis_model() is not None}, 200


@app.route('/')
def index():
    if get_current_user():
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/registrar', methods=['GET', 'POST'])
def registrar():
    if request.method == 'POST':
        nombre = request.form['nombre_completo']
        correo = request.form['correo']
        password = request.form['password']
        if not password:
            flash('Ingresa una contrase\u00f1a.', 'error')
            return render_template('register.html', title='Registro de usuario')
        hashed_password = generate_password_hash(password)
        conn = get_db_connection()
        cursor = conn.cursor() if is_sqlite_connection(conn) else conn.cursor()
        placeholder = db_placeholder(conn)
        try:
            cursor.execute(
                f'INSERT INTO usuarios (nombre_completo, correo, password) VALUES ({placeholder}, {placeholder}, {placeholder})',
                (nombre, correo, hashed_password),
            )
            conn.commit()
        except (sqlite3.IntegrityError, mysql.connector.Error):
            conn.rollback()
            flash('El correo ya est\u00e1 registrado, usa otro correo.', 'error')
            cursor.close()
            conn.close()
            return render_template('register.html', title='Registro de usuario')
        cursor.close()
        conn.close()
        flash('Usuario registrado correctamente. Ahora inicia sesi\u00f3n.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Registro de usuario')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        correo = request.form['correo']
        password = request.form['password']
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True) if not is_sqlite_connection(conn) else conn.cursor()
        placeholder = db_placeholder(conn)
        cursor.execute(
            f'SELECT id_usuario, nombre_completo, correo, password FROM usuarios WHERE correo = {placeholder}',
            (correo,),
        )
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        if not user:
            flash('Correo o contrase\u00f1a incorrectos.', 'error')
            return render_template('login.html', title='Iniciar sesi\u00f3n')
        stored_password = user['password']
        if not stored_password or not check_password_hash(stored_password, password):
            flash('Correo o contrase\u00f1a incorrectos.', 'error')
            return render_template('login.html', title='Iniciar sesi\u00f3n')
        session['user_id'] = user['id_usuario']
        session['user_name'] = user['nombre_completo']
        session['user_email'] = user['correo']
        flash(f'Bienvenido {user["nombre_completo"]}', 'success')
        return redirect(url_for('index'))
    return render_template('login.html', title='Iniciar sesi\u00f3n')


@app.route('/logout')
def logout():
    session.clear()
    flash('Has cerrado sesi\u00f3n.', 'success')
    return redirect(url_for('index'))


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    current_user = get_current_user()

    if request.method == 'POST':
        form_type = request.form.get('form_type')

        if form_type == 'health':
            registro = {
                'id_usuario': current_user['id'],
                'peso_kg': request.form['peso_kg'],
                'altura_cm': request.form['altura_cm'],
                'edad': request.form['edad'],
                'nivel_glucosa': request.form['nivel_glucosa'],
                'ojos_rojos': request.form['ojos_rojos'],
                'consumo_azucares': request.form['consumo_azucares'],
                'consumo_harinas': request.form['consumo_harinas'],
            }
            registro['imc'] = round(float(registro['peso_kg']) / ((float(registro['altura_cm']) / 100) ** 2), 2)
            conn = get_db_connection()
            cursor = conn.cursor()
            sql = '''INSERT INTO registros_salud
                (id_usuario, peso_kg, altura_cm, edad, nivel_glucosa, ojos_rojos, consumo_azucares, consumo_harinas, imc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)'''
            vals = (
                registro['id_usuario'], registro['peso_kg'], registro['altura_cm'],
                registro['edad'], registro['nivel_glucosa'], registro['ojos_rojos'],
                registro['consumo_azucares'], registro['consumo_harinas'], registro['imc'],
            )
            if not is_sqlite_connection(conn):
                sql = '''INSERT INTO registros_salud
                    (id_usuario, peso_kg, altura_cm, edad, nivel_glucosa, ojos_rojos, consumo_azucares, consumo_harinas)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)'''
                vals = (
                    registro['id_usuario'], registro['peso_kg'], registro['altura_cm'],
                    registro['edad'], registro['nivel_glucosa'], registro['ojos_rojos'],
                    registro['consumo_azucares'], registro['consumo_harinas'],
                )
            cursor.execute(sql, vals)
            conn.commit()
            cursor.close()
            conn.close()
            flash('Datos de salud guardados correctamente.', 'success')
            return redirect(url_for('dashboard'))

        if form_type == 'upload_skin':
            if 'imagen_piel' not in request.files:
                flash('Selecciona una imagen para cargar.', 'error')
                return redirect(url_for('dashboard'))
            imagen = request.files['imagen_piel']
            if imagen.filename == '':
                flash('Selecciona un archivo antes de enviar.', 'error')
                return redirect(url_for('dashboard'))
            if not allowed_file(imagen.filename):
                flash('Solo se permiten im\u00e1genes JPG, JPEG, PNG o BMP.', 'error')
                return redirect(url_for('dashboard'))
            filename = secure_filename(imagen.filename)
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            imagen.save(os.path.join(UPLOAD_FOLDER, filename))

            model = get_acantosis_model()
            if model is not None:
                try:
                    x = preprocess_skin_image(os.path.join(UPLOAD_FOLDER, filename))
                    with torch.no_grad():
                        logits = model(x)
                        probs = torch.softmax(logits, dim=1).numpy()[0]
                    idx = int(np.argmax(probs))
                    clase = _acantosis_class_names[idx] if idx < len(_acantosis_class_names) else f'Class_{idx}'
                    confianza = float(probs[idx])
                    rec_piel = generar_recomendaciones_piel(clase, confianza)

                    conn = get_db_connection()
                    cursor = conn.cursor()
                    placeholder = db_placeholder(conn)
                    cursor.execute(
                        f'INSERT INTO analisis_piel (id_usuario, filename, clase_detectada, confianza, recomendaciones) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})',
                        (current_user['id'], filename, clase, confianza, rec_piel)
                    )
                    conn.commit()
                    cursor.close()
                    conn.close()
                    flash(f'An\u00e1lisis completado - {clase}: {confianza:.1%}', 'success')
                    gc.collect()
                except Exception as e:
                    flash(f'Error al analizar la imagen: {e}', 'error')
                    gc.collect()
            else:
                flash('No se encontr\u00f3 el modelo (modelo_acantosis.keras).', 'info')
            return redirect(url_for('dashboard'))

        if form_type == 'settings':
            current_password = request.form['current_password']
            new_password = request.form['new_password']
            confirm_password = request.form['confirm_password']
            if not current_password or not new_password or not confirm_password:
                flash('Completa todos los campos de contrase\u00f1a.', 'error')
                return redirect(url_for('dashboard'))
            if new_password != confirm_password:
                flash('La nueva contrase\u00f1a y la confirmaci\u00f3n no coinciden.', 'error')
                return redirect(url_for('dashboard'))
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True) if not is_sqlite_connection(conn) else conn.cursor()
            placeholder = db_placeholder(conn)
            cursor.execute(
                f'SELECT password FROM usuarios WHERE id_usuario = {placeholder}',
                (current_user['id'],),
            )
            user = cursor.fetchone()
            stored_password = user['password']
            if not stored_password or not check_password_hash(stored_password, current_password):
                flash('Contrase\u00f1a actual incorrecta.', 'error')
                cursor.close()
                conn.close()
                return redirect(url_for('dashboard'))
            hashed_password = generate_password_hash(new_password)
            cursor.execute(
                f'UPDATE usuarios SET password = {placeholder} WHERE id_usuario = {placeholder}',
                (hashed_password, current_user['id']),
            )
            conn.commit()
            cursor.close()
            conn.close()
            flash('Contrase\u00f1a actualizada correctamente.', 'success')
            return redirect(url_for('dashboard'))

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True) if not is_sqlite_connection(conn) else conn.cursor()
    placeholder = db_placeholder(conn)
    cursor.execute(
        f'SELECT * FROM analisis_piel WHERE id_usuario = {placeholder} ORDER BY fecha_analisis DESC LIMIT 5',
        (current_user['id'],)
    )
    piel_list = cursor.fetchall()
    if is_sqlite_connection(conn):
        piel_list = [dict(row) for row in piel_list]
    ultimo_recomendacion = piel_list[0].get('recomendaciones') if piel_list else None
    ultimo_analisis = piel_list[0] if piel_list else None
    cursor.close()
    conn.close()

    return render_template(
        'dashboard.html', title='Dashboard',
        current_user=current_user,
        piel_list=piel_list,
        ultimo_recomendacion=ultimo_recomendacion,
        ultimo_analisis=ultimo_analisis,
    )


# Preload model at startup to avoid timeout on first request
try:
    get_acantosis_model()
    app.logger.info('Acantosis model loaded at startup')
except Exception as e:
    app.logger.error('Error preloading acantosis model: %s', e)


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
