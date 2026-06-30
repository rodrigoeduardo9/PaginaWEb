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
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()


def generar_recomendacion_combinada(clase, confianza, salud=None):
    rec_piel = generar_recomendaciones_piel(clase, confianza)
    if not salud:
        return rec_piel, None

    peso = float(salud['peso_kg'])
    altura = float(salud['altura_cm'])
    imc = round(peso / ((altura / 100) ** 2), 2)
    glucosa = float(salud['nivel_glucosa'])
    edad = int(salud['edad'])
    azucares = int(salud['consumo_azucares'])
    harinas = int(salud['consumo_harinas'])
    ojos_rojos = salud['ojos_rojos']

    riesgo_imagen = 2 if clase == 'Acanthosis_Nigricans' else 1 if clase in ('CRP', 'TFFD') else 0
    riesgo_salud = 0
    if imc >= 30:
        riesgo_salud += 2
    elif imc >= 25:
        riesgo_salud += 1
    if glucosa >= 126:
        riesgo_salud += 2
    elif glucosa >= 100:
        riesgo_salud += 1
    if edad >= 45:
        riesgo_salud += 1
    if ojos_rojos.lower() == 's\u00ed':
        riesgo_salud += 1
    if (azucares + harinas) >= 8:
        riesgo_salud += 1

    salud_recs = []
    riesgos = 0

    if imc >= 30:
        salud_recs.append("🔴 Tu IMC indica obesidad. Esto aumenta el riesgo de resistencia a la insulina y diabetes.")
        riesgos += 2
    elif imc >= 25:
        salud_recs.append("🟡 Tienes sobrepeso (IMC {:.1f}). Controla tu alimentaci\u00f3n y haz ejercicio.".format(imc))
        riesgos += 1
    else:
        salud_recs.append("🟢 Tu IMC ({:.1f}) est\u00e1 en un rango saludable.".format(imc))

    if glucosa >= 126:
        salud_recs.append("🔴 Glucosa elevada ({:.0f} mg/dL). Podr\u00eda indicar diabetes. Consulta a un m\u00e9dico.".format(glucosa))
        riesgos += 2
    elif glucosa >= 100:
        salud_recs.append("🟡 Glucosa ligeramente elevada ({:.0f} mg/dL). Podr\u00eda ser prediabetes.".format(glucosa))
        riesgos += 1
    else:
        salud_recs.append("🟢 Nivel de glucosa normal ({:.0f} mg/dL).".format(glucosa))

    if edad >= 45:
        salud_recs.append("🔴 Edad ({}) a\u00f1os: factor de riesgo para diabetes tipo 2.".format(edad))
        riesgos += 1
    elif edad >= 35:
        salud_recs.append("🟡 Edad ({}) a\u00f1os: riesgo moderado, haz chequeos peri\u00f3dicos.".format(edad))
        riesgos += 0

    if ojos_rojos.lower() == 's\u00ed':
        salud_recs.append("🔴 Ojos rojos: posible signo de fatiga visual o problemas de presi\u00f3n.")
        riesgos += 1

    consumo_total = azucares + harinas
    if consumo_total >= 8:
        salud_recs.append("🔴 Alto consumo de az\u00facares y harinas ({}/10). Reduce su ingesta.".format(consumo_total))
        riesgos += 1
    elif consumo_total >= 5:
        salud_recs.append("🟡 Consumo moderado de az\u00facares y harinas ({}/10).)".format(consumo_total))

    if clase == 'Acanthosis_Nigricans':
        riesgos += 2
    elif clase == 'CRP' or clase == 'TFFD':
        riesgos += 1

    if riesgo_salud <= 1 and riesgo_imagen >= 2:
        salud_recs.append(
            "\n⚠️ **POSIBLE DISCREPANCIA:** La imagen sugiere un marcador de riesgo metab\u00f3lico, "
            "pero tus datos de salud registrados son normales o de bajo riesgo. "
            "Esto podr\u00eda indicar que el an\u00e1lisis de piel est\u00e1 detectando se\u00f1ales tempranas "
            "que a\u00fan no se reflejan en tus m\u00e9tricas actuales. "
            "Te recomendamos monitoreo regular y una consulta m\u00e9dica para estudios m\u00e1s espec\u00edficos."
        )
    elif riesgo_salud >= 4 and riesgo_imagen == 0:
        salud_recs.append(
            "\n⚠️ **POSIBLE DISCREPANCIA:** Tu piel se ve saludable seg\u00fan el an\u00e1lisis, "
            "pero tus datos de salud indican un riesgo metab\u00f3lico significativo. "
            "Es posible que tengas una condici\u00f3n en etapa temprana que a\u00fan no se manifiesta en la piel. "
            "Te recomendamos consultar a un m\u00e9dico para estudios adicionales "
            "(curva de tolerancia a la glucosa, perfil lip\u00eddico)."
        )

    if riesgos >= 5:
        salud_recs.append("\n🔴 RIESGO GLOBAL ALTO: Combinaci\u00f3n de m\u00faltiples factores. Busca atenci\u00f3n m\u00e9dica pronto.")
    elif riesgos >= 3:
        salud_recs.append("\n🟡 RIESGO GLOBAL MODERADO: Varios factores detectados. Toma medidas preventivas.")
    else:
        salud_recs.append("\n🟢 RIESGO GLOBAL BAJO: Contin\u00faa con tus h\u00e1bitos saludables.")

    rec_combined = rec_piel + "\n\n---\n\n📋 **Datos de salud complementarios:**\n" + "\n".join(salud_recs)
    return rec_combined, {'imc': imc, 'riesgo_global': riesgos}


def generar_recomendaciones_piel(clase, confianza):
    recs = {
        'Acanthosis_Nigricans': (
            "🔴 ACANTOSIS NIGRICANS DETECTADA: Esta condici\u00f3n est\u00e1 fuertemente asociada con resistencia a la insulina. "
            "Consulta a un endocrin\u00f3logo lo antes posible. Reduce el consumo de az\u00facares y carbohidratos refinados. "
            "Realiza ejercicio f\u00edsico regular. Un diagn\u00f3stico temprano puede prevenir el desarrollo de diabetes tipo 2."
        ),
        'CRP': (
            "\U0001F7E1 MARCADOR INFLAMATORIO (CRP) ELEVADO: La prote\u00edna C reactiva elevada indica inflamaci\u00f3n sist\u00e9mica. "
            "Podr\u00eda estar asociada a riesgo cardiovascular, diabetes o infecciones. "
            "Se recomienda una dieta antiinflamatoria (omega-3, frutas, verduras), reducir el estr\u00e9s y consultar a un m\u00e9dico general."
        ),
        'Healthy': (
            "\U0001F7E2 PIEL SALUDABLE: No se detectan anormalidades significativas. "
            "Contin\u00faa con tus h\u00e1bitos de cuidado personal, protecci\u00f3n solar y chequeos m\u00e9dicos peri\u00f3dicos."
        ),
        'TFFD': (
            "\U0001F7E1 MARCADOR TFFD DETECTADO: Este hallazgo puede estar relacionado con factores metab\u00f3licos. "
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

                    conn = get_db_connection()
                    cursor = conn.cursor(dictionary=True) if not is_sqlite_connection(conn) else conn.cursor()
                    placeholder = db_placeholder(conn)
                    cursor.execute(
                        f'SELECT * FROM registros_salud WHERE id_usuario = {placeholder} ORDER BY fecha_formulario DESC LIMIT 1',
                        (current_user['id'],)
                    )
                    ultima_salud = cursor.fetchone()
                    if is_sqlite_connection(conn) and ultima_salud:
                        ultima_salud = dict(ultima_salud)
                    rec_final, salud_resumen = generar_recomendacion_combinada(clase, confianza, ultima_salud)

                    cursor.execute(
                        f'INSERT INTO analisis_piel (id_usuario, filename, clase_detectada, confianza, recomendaciones) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})',
                        (current_user['id'], filename, clase, confianza, rec_final)
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

    cursor.execute(
        f'SELECT * FROM registros_salud WHERE id_usuario = {placeholder} ORDER BY fecha_formulario DESC LIMIT 1',
        (current_user['id'],)
    )
    ultima_salud = cursor.fetchone()
    if is_sqlite_connection(conn) and ultima_salud:
        ultima_salud = dict(ultima_salud)

    cursor.close()
    conn.close()

    return render_template(
        'dashboard.html', title='Dashboard',
        current_user=current_user,
        piel_list=piel_list,
        ultimo_recomendacion=ultimo_recomendacion,
        ultimo_analisis=ultimo_analisis,
        ultima_salud=ultima_salud,
    )


@app.route('/configuracion', methods=['GET', 'POST'])
@login_required
def configuracion():
    current_user = get_current_user()
    if request.method == 'POST':
        current_password = request.form['current_password']
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']
        if not current_password or not new_password or not confirm_password:
            flash('Completa todos los campos de contrase\u00f1a.', 'error')
            return redirect(url_for('configuracion'))
        if new_password != confirm_password:
            flash('La nueva contrase\u00f1a y la confirmaci\u00f3n no coinciden.', 'error')
            return redirect(url_for('configuracion'))
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
            return redirect(url_for('configuracion'))
        hashed_password = generate_password_hash(new_password)
        cursor.execute(
            f'UPDATE usuarios SET password = {placeholder} WHERE id_usuario = {placeholder}',
            (hashed_password, current_user['id']),
        )
        conn.commit()
        cursor.close()
        conn.close()
        flash('Contrase\u00f1a actualizada correctamente.', 'success')
        return redirect(url_for('configuracion'))
    return render_template('settings.html', title='Configuraci\u00f3n de cuenta', current_user=current_user)


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
