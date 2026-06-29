import os
import sqlite3
import tempfile
from functools import wraps
import torch
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import mysql.connector
from utils import get_model, preprocess_image

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

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pth')
MODEL_LABELS = ["prediabetes", "insulin_resistance", "hypertension"]
_device = None
_model = None


def get_ai_model():
    global _model, _device
    if _model is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if os.path.exists(MODEL_PATH):
            _model = get_model(num_outputs=len(MODEL_LABELS), weights_path=MODEL_PATH, device=_device)
        else:
            _model = None
    return _model


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
        '''CREATE TABLE IF NOT EXISTS analisis_ia (
            id_analisis INTEGER PRIMARY KEY AUTOINCREMENT,
            id_usuario INTEGER NOT NULL,
            filename TEXT NOT NULL,
            prediabetes REAL,
            insulin_resistance REAL,
            hypertension REAL,
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
            flash('Por favor inicia sesión para acceder a esta sección.', 'info')
            return redirect(url_for('login'))
        return view(*args, **kwargs)
    return wrapped_view


@app.context_processor
def inject_user():
    return dict(current_user=get_current_user())


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def calcular_riesgo(registro):
    nivel_glucosa = float(registro['nivel_glucosa'])
    imc = float(registro['imc'])
    azucar = int(registro['consumo_azucares'])
    harinas = int(registro['consumo_harinas'])

    if nivel_glucosa >= 140 or imc >= 30 or azucar >= 4 or harinas >= 4:
        return 'Alto'
    if nivel_glucosa >= 120 or imc >= 25 or azucar >= 3 or harinas >= 3:
        return 'Moderado'
    return 'Bajo'


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
            flash('Ingresa una contraseña.', 'error')
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
            flash('El correo ya está registrado, usa otro correo.', 'error')
            cursor.close()
            conn.close()
            return render_template('register.html', title='Registro de usuario')
        cursor.close()
        conn.close()
        flash('Usuario registrado correctamente. Ahora inicia sesión.', 'success')
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
            flash('Correo o contraseña incorrectos.', 'error')
            return render_template('login.html', title='Iniciar sesión')

        stored_password = user['password']
        if not stored_password or not check_password_hash(stored_password, password):
            flash('Correo o contraseña incorrectos.', 'error')
            return render_template('login.html', title='Iniciar sesión')

        session['user_id'] = user['id_usuario']
        session['user_name'] = user['nombre_completo']
        session['user_email'] = user['correo']
        flash(f'Bienvenido {user["nombre_completo"]}', 'success')
        return redirect(url_for('index'))

    return render_template('login.html', title='Iniciar sesión')


@app.route('/logout')
def logout():
    session.clear()
    flash('Has cerrado sesión.', 'success')
    return redirect(url_for('index'))


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    current_user = get_current_user()
    uploaded_filename = None

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
            if is_sqlite_connection(conn):
                cursor.execute(
                    '''INSERT INTO registros_salud
                    (id_usuario, peso_kg, altura_cm, edad, nivel_glucosa, ojos_rojos, consumo_azucares, consumo_harinas, imc)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (
                        registro['id_usuario'],
                        registro['peso_kg'],
                        registro['altura_cm'],
                        registro['edad'],
                        registro['nivel_glucosa'],
                        registro['ojos_rojos'],
                        registro['consumo_azucares'],
                        registro['consumo_harinas'],
                        registro['imc'],
                    )
                )
            else:
                cursor.execute(
                    '''INSERT INTO registros_salud
                    (id_usuario, peso_kg, altura_cm, edad, nivel_glucosa, ojos_rojos, consumo_azucares, consumo_harinas)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)''',
                    (
                        registro['id_usuario'],
                        registro['peso_kg'],
                        registro['altura_cm'],
                        registro['edad'],
                        registro['nivel_glucosa'],
                        registro['ojos_rojos'],
                        registro['consumo_azucares'],
                        registro['consumo_harinas'],
                    )
                )
            conn.commit()
            cursor.close()
            conn.close()
            flash('Datos de salud guardados correctamente. El dashboard de IA estará disponible pronto.', 'success')
            return redirect(url_for('dashboard'))

        if form_type == 'upload':
            if 'imagen' not in request.files:
                flash('Selecciona una imagen para cargar.', 'error')
                return redirect(url_for('dashboard'))

            imagen = request.files['imagen']
            if imagen.filename == '':
                flash('Selecciona un archivo antes de enviar.', 'error')
                return redirect(url_for('dashboard'))

            if not allowed_file(imagen.filename):
                flash('Solo se permiten imágenes JPG, JPEG, PNG o BMP.', 'error')
                return redirect(url_for('dashboard'))

            filename = secure_filename(imagen.filename)
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            imagen.save(os.path.join(UPLOAD_FOLDER, filename))
            uploaded_filename = filename

            model = get_ai_model()
            if model is not None:
                try:
                    x = preprocess_image(os.path.join(UPLOAD_FOLDER, filename)).to(_device)
                    with torch.no_grad():
                        logits = model(x)[0]
                        probs = torch.sigmoid(logits).cpu().numpy().tolist()
                    ia_result = dict(zip(MODEL_LABELS, probs))

                    conn = get_db_connection()
                    cursor = conn.cursor()
                    placeholder = db_placeholder(conn)
                    cursor.execute(
                        f'INSERT INTO analisis_ia (id_usuario, filename, prediabetes, insulin_resistance, hypertension) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})',
                        (current_user['id'], filename, ia_result['prediabetes'], ia_result['insulin_resistance'], ia_result['hypertension'])
                    )
                    conn.commit()
                    cursor.close()
                    conn.close()

                    flash(f'Imagen analizada - Prediabetes: {ia_result["prediabetes"]:.1%}, Resistencia insulina: {ia_result["insulin_resistance"]:.1%}, Hipertensión: {ia_result["hypertension"]:.1%}', 'success')
                except Exception as e:
                    flash(f'Error al analizar la imagen: {e}', 'error')
            else:
                flash('Imagen cargada, pero no se encontró el modelo IA (model.pth).', 'info')
            return redirect(url_for('dashboard'))

        if form_type == 'settings':
            current_password = request.form['current_password']
            new_password = request.form['new_password']
            confirm_password = request.form['confirm_password']

            if not current_password or not new_password or not confirm_password:
                flash('Completa todos los campos de contraseña.', 'error')
                return redirect(url_for('dashboard'))

            if new_password != confirm_password:
                flash('La nueva contraseña y la confirmación no coinciden.', 'error')
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
                flash('Contraseña actual incorrecta.', 'error')
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
            flash('Contraseña actualizada correctamente.', 'success')
            return redirect(url_for('dashboard'))

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True) if not is_sqlite_connection(conn) else conn.cursor()
    placeholder = db_placeholder(conn)
    cursor.execute(
        f'SELECT * FROM analisis_ia WHERE id_usuario = {placeholder} ORDER BY fecha_analisis DESC LIMIT 5',
        (current_user['id'],)
    )
    analisis_list = cursor.fetchall()
    if is_sqlite_connection(conn):
        analisis_list = [dict(row) for row in analisis_list]
    cursor.close()
    conn.close()

    return render_template('dashboard.html', title='Dashboard', current_user=current_user, analisis_list=analisis_list)


@app.route('/upload')
@login_required
def upload():
    return redirect(url_for('dashboard'))


@app.route('/salud')
@login_required
def salud():
    return redirect(url_for('dashboard'))


@app.route('/usuarios')
@login_required
def usuarios():
    return redirect(url_for('dashboard'))


@app.route('/registros')
@login_required
def registros():
    return redirect(url_for('dashboard'))


@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'file missing'}), 400
    f = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    model = get_ai_model()
    if model is None:
        os.unlink(tmp_path)
        return jsonify({'error': 'model.pth not found', 'labels': MODEL_LABELS, 'probabilities': np.random.rand(len(MODEL_LABELS)).tolist(), 'note': 'Placeholder'}), 200

    try:
        x = preprocess_image(tmp_path).to(_device)
        with torch.no_grad():
            logits = model(x)[0]
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
        os.unlink(tmp_path)
        return jsonify({'labels': MODEL_LABELS, 'probabilities': probs, 'note': 'Model predictions'})
    except Exception as e:
        os.unlink(tmp_path)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
