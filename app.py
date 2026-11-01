import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Predicción de deserción estudiantil",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelo_desercion.pkl")

# Los desplegables muestran descripciones legibles, pero devuelven al modelo
# los códigos numéricos originales del dataset UCI.
CATEGORY_LABELS = {
    "Marital status": {
        1: "Soltero/a", 2: "Casado/a", 3: "Viudo/a", 4: "Divorciado/a",
        5: "Unión de hecho", 6: "Separado/a legalmente",
    },
    "Application mode": {
        1: "Fase 1 – cupo general", 2: "Admisión por normativa 612/93",
        5: "Fase 1 – cupo especial (Azores)", 7: "Titular de otro estudio superior",
        10: "Admisión por normativa 854-B/99", 15: "Estudiante internacional (pregrado)",
        16: "Fase 1 – cupo especial (Madeira)", 17: "Fase 2 – cupo general",
        18: "Fase 3 – cupo general", 26: "Admisión por plan diferente (normativa 533-A/99 b2)",
        27: "Admisión desde otra institución (normativa 533-A/99 b3)",
        39: "Admisión para mayores de 23 años", 42: "Traslado de institución",
        43: "Cambio de programa", 44: "Diploma de especialización tecnológica",
        51: "Cambio de institución o programa", 53: "Diploma de ciclo corto",
        57: "Cambio de institución o programa (internacional)",
    },
    "Application order": {
        0: "Primera preferencia", 1: "Segunda preferencia", 2: "Tercera preferencia", 3: "Cuarta preferencia",
        4: "Quinta preferencia", 5: "Sexta preferencia", 6: "Séptima preferencia", 9: "Décima preferencia (última)",
    },
    "Course": {
        33: "Tecnologías de producción de biocombustibles",
        171: "Diseño de animación y multimedia",
        8014: "Trabajo social (jornada nocturna)", 9003: "Agronomía",
        9070: "Diseño de comunicación", 9085: "Enfermería veterinaria",
        9119: "Ingeniería informática", 9130: "Equinocultura",
        9147: "Administración", 9238: "Trabajo social", 9254: "Turismo",
        9500: "Enfermería", 9556: "Higiene oral",
        9670: "Gestión de publicidad y marketing", 9773: "Periodismo y comunicación",
        9853: "Educación básica", 9991: "Administración (jornada nocturna)",
    },
    "Daytime/evening attendance": {1: "Diurna", 0: "Nocturna"},
    "Previous qualification": {
        1: "Educación secundaria", 2: "Educación superior – licenciatura",
        3: "Educación superior – título universitario", 4: "Educación superior – maestría",
        5: "Educación superior – doctorado", 6: "Estudios previos de educación superior",
        9: "12.º año de escolaridad (incompleto)", 10: "11.º año de escolaridad (incompleto)",
        12: "Otro – 11.º año de escolaridad", 14: "10.º año de escolaridad",
        15: "10.º año de escolaridad (incompleto)",
        19: "Educación básica, 3.er ciclo (9.º–11.º) o equivalente",
        38: "Educación básica, 2.º ciclo (6.º–8.º) o equivalente",
        39: "Curso de especialización tecnológica",
        40: "Educación superior – título de 1.er ciclo",
        42: "Curso técnico superior profesional",
        43: "Educación superior – máster de 2.º ciclo",
    },
    "Nacionality": {
        1: "Portuguesa", 2: "Alemana", 6: "Española", 11: "Italiana",
        13: "Neerlandesa", 14: "Inglesa", 17: "Lituana", 21: "Angoleña",
        22: "Caboverdiana", 24: "Guineana", 25: "Mozambiqueña", 26: "Santomense",
        32: "Turca", 41: "Brasileña", 62: "Rumena", 100: "Moldava",
        101: "Mexicana", 103: "Ucraniana", 105: "Rusa", 108: "Cubana", 109: "Colombiana",
    },
}

# Códigos de nivel educativo de madre y padre según el diccionario UCI.
PARENT_EDUCATION = {
    1: "Educación secundaria (12.º año)",
    2: "Educación superior – licenciatura",
    3: "Educación superior – título universitario",
    4: "Educación superior – maestría",
    5: "Educación superior – doctorado",
    6: "Estudios previos de educación superior",
    9: "12.º año de escolaridad (incompleto)",
    10: "11.º año de escolaridad (incompleto)",
    11: "7.º año (plan antiguo)",
    12: "Otro – 11.º año de escolaridad",
    13: "2.º año del curso complementario de secundaria",
    14: "10.º año de escolaridad",
    18: "Curso general de comercio",
    19: "Educación básica, 3.er ciclo (9.º–11.º) o equivalente",
    20: "Curso complementario de secundaria",
    22: "Curso técnico-profesional",
    25: "Curso complementario de secundaria (no finalizado)",
    26: "7.º año de escolaridad",
    27: "2.º ciclo del curso general de secundaria",
    29: "9.º año de escolaridad (incompleto)",
    30: "8.º año de escolaridad",
    31: "Curso general de administración y comercio",
    33: "Contabilidad y administración complementaria",
    34: "Desconocido",
    35: "No sabe leer ni escribir",
    36: "Puede leer, pero no completó el 4.º año",
    37: "Educación básica, 1.er ciclo (4.º–5.º) o equivalente",
    38: "Educación básica, 2.º ciclo (6.º–8.º) o equivalente",
    39: "Curso de especialización tecnológica",
    40: "Educación superior – título de 1.er ciclo",
    41: "Curso de estudios superiores especializados",
    42: "Curso técnico superior profesional",
    43: "Educación superior – máster de 2.º ciclo",
    44: "Educación superior – doctorado de 3.er ciclo",
}
CATEGORY_LABELS["Mother's qualification"] = PARENT_EDUCATION
CATEGORY_LABELS["Father's qualification"] = PARENT_EDUCATION

# Categorías ocupacionales definidas por UCI. Las etiquetas se simplifican
# en español para que el formulario sea entendible sin alterar los códigos.
OCCUPATION_LABELS = {
    0: "Estudiante", 1: "Directivos y responsables de organismos públicos",
    2: "Especialistas científicos e intelectuales",
    3: "Técnicos y profesionales de nivel medio",
    4: "Personal administrativo", 5: "Servicios personales, seguridad y ventas",
    6: "Agricultura, pesca y actividad forestal cualificada",
    7: "Industria, construcción y oficios cualificados",
    8: "Operadores de instalaciones, máquinas y montaje",
    9: "Trabajadores no cualificados", 10: "Fuerzas armadas",
    90: "Otra situación", 99: "Sin información",
    101: "Oficiales de las fuerzas armadas", 102: "Suboficiales de las fuerzas armadas",
    103: "Otro personal de las fuerzas armadas",
    112: "Directores de servicios administrativos y comerciales",
    114: "Directores de hoteles, restaurantes, comercio y otros servicios",
    121: "Especialistas en ciencias físicas, matemáticas e ingeniería",
    122: "Profesionales de la salud", 123: "Docentes",
    124: "Especialistas en finanzas, contabilidad y administración",
    125: "Especialistas en tecnologías de información y comunicación",
    131: "Técnicos de ciencias e ingeniería", 132: "Técnicos de salud",
    134: "Técnicos de servicios jurídicos, sociales, deportivos y culturales",
    135: "Técnicos de tecnologías de información y comunicación",
    141: "Empleados de oficina, secretariado y procesamiento de datos",
    143: "Operadores de contabilidad, finanzas y registros",
    144: "Personal de apoyo administrativo",
    151: "Trabajadores de servicios personales", 152: "Vendedores",
    153: "Trabajadores de cuidado personal", 154: "Personal de protección y seguridad",
    161: "Agricultores y trabajadores cualificados orientados al mercado",
    163: "Agricultores, ganaderos y pescadores de subsistencia",
    171: "Trabajadores cualificados de construcción (excepto electricistas)",
    172: "Trabajadores cualificados de metalurgia y afines",
    173: "Impresión, instrumentos de precisión, joyería y artesanía",
    174: "Trabajadores cualificados de electricidad y electrónica",
    175: "Procesamiento de alimentos, madera, ropa y otras manufacturas",
    181: "Operadores de instalaciones y máquinas fijas",
    182: "Ensambladores", 183: "Conductores y operadores de equipos móviles",
    191: "Personal de limpieza", 192: "Trabajadores no cualificados de agricultura y pesca",
    193: "Trabajadores no cualificados de minería, construcción, manufactura y transporte",
    194: "Ayudantes de preparación de alimentos",
    195: "Vendedores ambulantes y prestadores de servicios en la calle",
}
CATEGORY_LABELS["Mother's occupation"] = OCCUPATION_LABELS
CATEGORY_LABELS["Father's occupation"] = OCCUPATION_LABELS

# Variables binarias: se muestra Sí/No y se devuelve 1/0.
BINARY_LABELS = {
    "Displaced": {0: "No", 1: "Sí"},
    "Educational special needs": {0: "No", 1: "Sí"},
    "Debtor": {0: "No", 1: "Sí"},
    "Tuition fees up to date": {0: "No", 1: "Sí"},
    "Gender": {0: "Mujer", 1: "Hombre"},
    "Scholarship holder": {0: "No", 1: "Sí"},
    "International": {0: "No", 1: "Sí"},
}
CATEGORY_LABELS.update(BINARY_LABELS)

# Limita cada desplegable a los códigos que realmente aparecen en la columna
# correspondiente del archivo de entrenamiento. Evita ofrecer opciones inválidas.
ALLOWED_CODES = {
    "Marital status": [1, 2, 3, 4, 5, 6],
    "Application mode": [1, 2, 5, 7, 10, 15, 16, 17, 18, 26, 27, 39, 42, 43, 44, 51, 53, 57],
    "Application order": [0, 1, 2, 3, 4, 5, 6, 9],
    "Course": [33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254, 9500, 9556, 9670, 9773, 9853, 9991],
    "Daytime/evening attendance": [0, 1],
    "Previous qualification": [1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 19, 38, 39, 40, 42, 43],
    "Nacionality": [1, 2, 6, 11, 13, 14, 17, 21, 22, 24, 25, 26, 32, 41, 62, 100, 101, 103, 105, 108, 109],
    "Mother's qualification": [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 18, 19, 22, 26, 27, 29, 30, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
    "Father's qualification": [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 19, 20, 22, 25, 26, 27, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
    "Mother's occupation": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90, 99, 122, 123, 125, 131, 132, 134, 141, 143, 144, 151, 152, 153, 171, 173, 175, 191, 192, 193, 194],
    "Father's occupation": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90, 99, 101, 102, 103, 112, 114, 121, 122, 123, 124, 131, 132, 134, 135, 141, 143, 144, 151, 152, 153, 154, 161, 163, 171, 172, 174, 175, 181, 182, 183, 192, 193, 194, 195],
    "Displaced": [0, 1], "Educational special needs": [0, 1],
    "Debtor": [0, 1], "Tuition fees up to date": [0, 1], "Gender": [0, 1],
    "Scholarship holder": [0, 1], "International": [0, 1],
}

DEFAULTS = {
    "Marital status": 1,
    "Application mode": 1,
    "Application order": 1,
    "Course": 9500,
    "Daytime/evening attendance": 1,
    "Previous qualification": 1,
    "Previous qualification (grade)": 133.1,
    "Nacionality": 1,
    "Mother's qualification": 1,
    "Father's qualification": 37,
    "Mother's occupation": 9,
    "Father's occupation": 9,
    "Admission grade": 126.1,
    "Displaced": 1,
    "Educational special needs": 0,
    "Debtor": 0,
    "Tuition fees up to date": 1,
    "Gender": 0,
    "Scholarship holder": 0,
    "Age at enrollment": 18,
    "International": 0,
    "Curricular units 1st sem (credited)": 0,
    "Curricular units 1st sem (enrolled)": 6,
    "Curricular units 1st sem (evaluations)": 8,
    "Curricular units 1st sem (approved)": 6,
    "Curricular units 1st sem (grade)": 12.29,
    "Curricular units 1st sem (without evaluations)": 0,
    "Unemployment rate": 11.1,
    "Inflation rate": 1.4,
    "GDP": 0.32,
}

LABELS = {
    "Marital status": "Estado civil",
    "Application mode": "Vía de admisión",
    "Application order": "Orden de preferencia del programa",
    "Course": "Programa académico",
    "Daytime/evening attendance": "Jornada de estudio",
    "Previous qualification": "Nivel de estudios previo",
    "Previous qualification (grade)": "Calificación de estudios previos",
    "Nacionality": "Nacionalidad",
    "Mother's qualification": "Nivel educativo de la madre",
    "Father's qualification": "Nivel educativo del padre",
    "Mother's occupation": "Ocupación de la madre",
    "Father's occupation": "Ocupación del padre",
    "Admission grade": "Calificación de admisión",
    "Displaced": "¿Es estudiante desplazado/a?",
    "Educational special needs": "¿Tiene necesidades educativas especiales?",
    "Debtor": "¿Tiene deudas pendientes?",
    "Tuition fees up to date": "¿Está al día con el pago de matrícula?",
    "Gender": "Género",
    "Scholarship holder": "¿Tiene beca?",
    "Age at enrollment": "Edad al ingresar",
    "International": "¿Es estudiante internacional?",
    "Curricular units 1st sem (credited)": "Asignaturas reconocidas en el primer semestre",
    "Curricular units 1st sem (enrolled)": "Asignaturas matriculadas en el primer semestre",
    "Curricular units 1st sem (evaluations)": "Evaluaciones realizadas en el primer semestre",
    "Curricular units 1st sem (approved)": "Asignaturas aprobadas en el primer semestre",
    "Curricular units 1st sem (grade)": "Promedio del primer semestre (escala 0–20)",
    "Curricular units 1st sem (without evaluations)": "Asignaturas sin evaluación en el primer semestre",
    "Unemployment rate": "Tasa de desempleo (%)",
    "Inflation rate": "Tasa de inflación (%)",
    "GDP": "Indicador del PIB",
}

# Explicaciones específicas para cada campo. Se muestran en el icono de ayuda (?)
# junto al control correspondiente.
FIELD_HELP = {
    "Marital status": "Estado civil de la persona al momento de solicitar el ingreso. Selecciona la situación que figure en el registro del estudiante.",
    "Application mode": "Vía administrativa por la que ingresó a la institución (fase, cupo, traslado, cambio de programa, etc.). Las categorías corresponden al sistema portugués del dataset UCI; no necesariamente equivalen a los procesos de admisión de una institución colombiana.",
    "Application order": "Posición del programa dentro de las preferencias del estudiante: 0 representa la primera opción y 9 la última. El modelo utiliza el código original.",
    "Course": "Programa o carrera en la que se matriculó el estudiante, según el catálogo de la institución que originó el dataset.",
    "Daytime/evening attendance": "Jornada en la que estudia: diurna o nocturna. Selecciona la jornada registrada para el estudiante.",
    "Previous qualification": "Nivel o tipo de formación académica que el estudiante había completado o cursado antes de ingresar al programa actual.",
    "Previous qualification (grade)": "Calificación asociada a los estudios previos. El dataset utiliza una escala de 0 a 200; registra la nota en esa misma escala.",
    "Nacionality": "Nacionalidad registrada en la matrícula. Las opciones disponibles son las categorías presentes en el dataset original.",
    "Mother's qualification": "Nivel educativo más alto registrado para la madre. Si el dato no se conoce, utiliza la opción de desconocido cuando esté disponible.",
    "Father's qualification": "Nivel educativo más alto registrado para el padre. Si el dato no se conoce, utiliza la opción de desconocido cuando esté disponible.",
    "Mother's occupation": "Categoría ocupacional de la madre, no necesariamente el nombre exacto de su cargo. Elige la categoría que mejor corresponda a su actividad registrada.",
    "Father's occupation": "Categoría ocupacional del padre, no necesariamente el nombre exacto de su cargo. Elige la categoría que mejor corresponda a su actividad registrada.",
    "Admission grade": "Puntuación o calificación obtenida por el estudiante en el proceso de admisión. En el dataset se expresa en una escala de 0 a 200.",
    "Displaced": "Indica si el estudiante aparece registrado en condición de desplazamiento en la información de origen.",
    "Educational special needs": "Indica si el estudiante tiene necesidades educativas especiales registradas.",
    "Debtor": "Indica si el estudiante figura con obligaciones o deudas pendientes en el registro institucional.",
    "Tuition fees up to date": "Indica si los pagos de matrícula se encuentran al día en el registro considerado.",
    "Gender": "Género registrado en el dataset. Se muestran las opciones en palabras y la aplicación envía el código correspondiente al modelo.",
    "Scholarship holder": "Indica si el estudiante es beneficiario de una beca.",
    "Age at enrollment": "Edad del estudiante, en años, cuando ingresó al programa académico.",
    "International": "Indica si el estudiante está registrado como estudiante internacional en la institución de origen.",
    "Curricular units 1st sem (credited)": "Cantidad de asignaturas o unidades curriculares del primer semestre que fueron reconocidas por estudios anteriores o equivalencias.",
    "Curricular units 1st sem (enrolled)": "Número de asignaturas o unidades curriculares en las que el estudiante se matriculó durante el primer semestre.",
    "Curricular units 1st sem (evaluations)": "Cantidad de evaluaciones registradas para el estudiante en las asignaturas del primer semestre.",
    "Curricular units 1st sem (approved)": "Número de asignaturas del primer semestre que el estudiante aprobó.",
    "Curricular units 1st sem (grade)": "Promedio de calificaciones obtenido en el primer semestre. El dataset utiliza una escala de 0 a 20.",
    "Curricular units 1st sem (without evaluations)": "Número de asignaturas matriculadas en el primer semestre para las que no se registró ninguna evaluación.",
    "Unemployment rate": "Tasa de desempleo del entorno económico asociado al periodo del registro. No es un dato individual del estudiante. El valor de referencia se toma de la mediana histórica del dataset UCI.",
    "Inflation rate": "Tasa de inflación del entorno económico asociado al periodo del registro. No es un dato individual del estudiante. El valor de referencia se toma de la mediana histórica del dataset UCI.",
    "GDP": "Indicador del producto interno bruto (PIB) asociado al periodo del registro. Puede ser positivo o negativo. No es un dato individual del estudiante; el valor de referencia se toma de la mediana histórica del dataset UCI.",
}

# Medianas históricas de las tres variables macroeconómicas en el archivo UCI
# de 4.424 registros: desempleo 11.1, inflación 1.4 y PIB 0.32.
MACRO_REFERENCE = {
    "Unemployment rate": 11.1,
    "Inflation rate": 1.4,
    "GDP": 0.32,
}

NUMERIC_RANGES = {
    "Previous qualification (grade)": (0.0, 200.0, 0.1),
    "Admission grade": (0.0, 200.0, 0.1),
    "Age at enrollment": (17, 70, 1),
    "Curricular units 1st sem (credited)": (0, 20, 1),
    "Curricular units 1st sem (enrolled)": (0, 26, 1),
    "Curricular units 1st sem (evaluations)": (0, 45, 1),
    "Curricular units 1st sem (approved)": (0, 26, 1),
    "Curricular units 1st sem (grade)": (0.0, 20.0, 0.01),
    "Curricular units 1st sem (without evaluations)": (0, 12, 1),
    "Unemployment rate": (7.0, 17.0, 0.1),
    "Inflation rate": (-1.0, 4.0, 0.1),
    "GDP": (-5.0, 4.0, 0.01),
}

# Ajustes visuales: mejor legibilidad, ancho máximo y títulos claros.
st.markdown(
    """
    <style>
      .block-container {max-width: 1320px; padding-top: 1.7rem; padding-bottom: 2.5rem;}
      h1 {font-size: 2rem !important;}
      h2 {font-size: 1.35rem !important;}
      h3 {font-size: 1.08rem !important;}
      div[data-testid="stWidgetLabel"] p {font-size: 0.96rem; font-weight: 600;}
      div[data-testid="stCaptionContainer"] p {font-size: 0.88rem;}
      .intro {font-size: 1rem; line-height: 1.55; color: #c9ced8;}
      .small-note {font-size: .86rem; color: #aeb6c4;}
    </style>
    """,
    unsafe_allow_html=True,
)


def render_feature(feature: str):
    label = LABELS.get(feature, feature)
    default = DEFAULTS[feature]

    if feature in CATEGORY_LABELS:
        options = ALLOWED_CODES[feature]
        # El control muestra la etiqueta descriptiva; el valor que devuelve es el código original.
        return st.selectbox(
            label,
            options=options,
            index=options.index(default) if default in options else 0,
            format_func=lambda code: CATEGORY_LABELS[feature][code],
            key=f"field_{feature}",
            help=FIELD_HELP[feature] + " La selección se convierte automáticamente al código numérico original del dataset; no necesitas conocer ese código.",
        )

    minimum, maximum, step = NUMERIC_RANGES[feature]
    return st.number_input(
        label,
        min_value=minimum,
        max_value=maximum,
        value=default,
        step=step,
        key=f"field_{feature}",
        help=FIELD_HELP[feature] + " Introduce el valor correspondiente al estudiante; respeta la escala indicada.",
    )


st.title("🎓 Predicción de deserción y éxito académico")
st.markdown(
    '<p class="intro">Herramienta académica que utiliza un modelo de Machine Learning para estimar '
    'si un perfil estudiantil se asocia con <b>deserción</b>, <b>continuidad de matrícula</b> o '
    '<b>graduación</b>.</p>',
    unsafe_allow_html=True,
)
st.info(
    "El modelo necesita 30 variables: 27 datos del estudiante y 3 indicadores del contexto económico. "
    "Todos los campos deben tener un valor para calcular la predicción. Los valores iniciales son "
    "referencias y deben revisarse o reemplazarse por la información real del estudiante. "
    "Los desplegables muestran descripciones en español y convierten cada selección al código original del modelo."
)

if not os.path.exists(MODEL_PATH):
    st.error("No se encontró modelo_desercion.pkl. Ubica el archivo en la misma carpeta que app.py.")
    st.stop()

try:
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    model_name = artifact.get("model_name", "Modelo entrenado")
except Exception as exc:
    st.error(
        "No fue posible cargar el modelo. Comprueba que modelo_desercion.pkl sea el archivo "
        "reentrenado con Scikit-learn 1.9.1 y que la versión instalada coincida."
    )
    st.exception(exc)
    st.stop()

if len(feature_columns) != 30:
    st.error(
        f"El artefacto contiene {len(feature_columns)} variables; la aplicación espera las 30 variables del notebook."
    )
    st.stop()

st.subheader("Información del estudiante")
st.caption("Revisa los campos de cada sección. Los 27 campos del estudiante están precargados con valores de referencia; reemplázalos por los datos reales antes de calcular la estimación. Los 3 indicadores económicos se administran en la sección 5. No dejes valores de ejemplo sin revisar.")

with st.form("student_prediction_form"):
    values = {}

    with st.expander("1. Datos de admisión y trayectoria académica", expanded=True):
        col1, col2 = st.columns(2, gap="large")
        group_1 = [
            "Marital status", "Application mode", "Application order", "Course",
            "Daytime/evening attendance", "Previous qualification",
            "Previous qualification (grade)", "Nacionality", "Admission grade",
            "Age at enrollment",
        ]
        for idx, feature in enumerate(group_1):
            with (col1 if idx % 2 == 0 else col2):
                values[feature] = render_feature(feature)

    with st.expander("2. Información familiar", expanded=True):
        col1, col2 = st.columns(2, gap="large")
        group_2 = [
            "Mother's qualification", "Father's qualification",
            "Mother's occupation", "Father's occupation",
        ]
        for idx, feature in enumerate(group_2):
            with (col1 if idx % 2 == 0 else col2):
                values[feature] = render_feature(feature)

    with st.expander("3. Situación personal y socioeconómica", expanded=True):
        col1, col2 = st.columns(2, gap="large")
        group_3 = [
            "Gender", "Displaced", "Educational special needs", "International",
            "Debtor", "Tuition fees up to date", "Scholarship holder",
        ]
        for idx, feature in enumerate(group_3):
            with (col1 if idx % 2 == 0 else col2):
                values[feature] = render_feature(feature)

    with st.expander("4. Desempeño académico del primer semestre", expanded=True):
        col1, col2 = st.columns(2, gap="large")
        group_4 = [
            "Curricular units 1st sem (credited)",
            "Curricular units 1st sem (enrolled)",
            "Curricular units 1st sem (evaluations)",
            "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (grade)",
            "Curricular units 1st sem (without evaluations)",
        ]
        for idx, feature in enumerate(group_4):
            with (col1 if idx % 2 == 0 else col2):
                values[feature] = render_feature(feature)

    reviewed = st.checkbox(
        "Confirmo que revisé los datos y que representan al estudiante que deseo evaluar.",
        value=False,
        help="Esta confirmación evita calcular una estimación dejando por error los valores de ejemplo. No es una variable adicional del modelo."
    )
    submitted = st.form_submit_button(
        "🔎 Calcular predicción", type="primary", use_container_width=True
    )

# Las variables macroeconómicas no describen al estudiante. Por defecto se usan
# las medianas del dataset histórico; opcionalmente pueden personalizarse.
with st.expander("5. Contexto económico del periodo (3 variables del modelo)", expanded=False):
    st.write(
        "Estos indicadores describen el entorno económico del periodo, no al estudiante. "
        "Por defecto se aplican las medianas históricas del dataset UCI. El conjunto procede "
        "de una institución de Portugal; estos valores no son indicadores actuales de Colombia."
    )
    customize_macro = st.checkbox(
        "Tengo los indicadores exactos del periodo y deseo personalizarlos",
        value=False,
        help="Activa esta opción solo si cuentas con las cifras correspondientes al mismo contexto y periodo que deseas representar."
    )
    if customize_macro:
        macro_col1, macro_col2, macro_col3 = st.columns(3)
        with macro_col1:
            values["Unemployment rate"] = st.number_input(
                "Tasa de desempleo (%)", min_value=7.0, max_value=17.0,
                value=MACRO_REFERENCE["Unemployment rate"], step=0.1,
                help=FIELD_HELP["Unemployment rate"]
            )
        with macro_col2:
            values["Inflation rate"] = st.number_input(
                "Tasa de inflación (%)", min_value=-1.0, max_value=4.0,
                value=MACRO_REFERENCE["Inflation rate"], step=0.1,
                help=FIELD_HELP["Inflation rate"]
            )
        with macro_col3:
            values["GDP"] = st.number_input(
                "Indicador del PIB", min_value=-5.0, max_value=4.0,
                value=MACRO_REFERENCE["GDP"], step=0.01,
                help=FIELD_HELP["GDP"]
            )
    else:
        macro_col1, macro_col2, macro_col3 = st.columns(3)
        macro_col1.metric("Desempleo", f'{MACRO_REFERENCE["Unemployment rate"]:.1f} %')
        macro_col2.metric("Inflación", f'{MACRO_REFERENCE["Inflation rate"]:.1f} %')
        macro_col3.metric("Indicador del PIB", f'{MACRO_REFERENCE["GDP"]:.2f}')
        st.caption("Se aplicarán estos valores de referencia al modelo. Puedes personalizarlos si conoces los indicadores del periodo.")
    for macro_feature, macro_value in MACRO_REFERENCE.items():
        if macro_feature not in values:
            values[macro_feature] = macro_value

if submitted:
    if not reviewed:
        st.warning("Antes de calcular la predicción, confirma que revisaste los valores del estudiante. Los valores iniciales son solo referencias.")
        st.stop()
    missing_features = [column for column in feature_columns if column not in values]
    if missing_features:
        st.error("Faltan variables necesarias para el modelo: " + ", ".join(missing_features))
        st.stop()

    # Mantiene estrictamente el orden y los nombres de variables guardados en el artefacto.
    input_data = pd.DataFrame(
        [[values[column] for column in feature_columns]],
        columns=feature_columns,
    )

    try:
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        class_probabilities = dict(zip(model.classes_, probabilities))

        friendly_names = {
            "Dropout": "Deserción",
            "Enrolled": "Matriculado/a",
            "Graduate": "Graduado/a",
        }
        predicted_label = friendly_names.get(prediction, str(prediction))

        st.divider()
        st.subheader("Resultado de la estimación")
        st.metric("Categoría estimada por el modelo", predicted_label)
        st.caption(f"Modelo utilizado: {model_name}")

        if prediction == "Dropout":
            st.warning(
                "El modelo asocia este perfil con la categoría de deserción. "
                "Puede ser una señal para revisar necesidades de acompañamiento académico."
            )
        elif prediction == "Enrolled":
            st.info("El modelo asocia este perfil con la categoría de continuidad de matrícula.")
        else:
            st.success("El modelo asocia este perfil con la categoría de graduación.")

        probability_rows = [
            {
                "Resultado": friendly_names.get(label, str(label)),
                "Probabilidad (%)": round(float(probability) * 100, 2),
            }
            for label, probability in class_probabilities.items()
        ]
        probability_df = pd.DataFrame(probability_rows).sort_values(
            "Probabilidad (%)", ascending=False
        )
        st.write("**Probabilidades estimadas por categoría**")
        st.bar_chart(probability_df.set_index("Resultado")["Probabilidad (%)"])
        st.dataframe(probability_df, hide_index=True, use_container_width=True)

        st.caption(
            "Importante: es una predicción académica experimental basada en registros históricos de una institución de Portugal. "
            "No representa una certeza ni debe utilizarse como único criterio para tomar decisiones sobre estudiantes. "
            "Su aplicación en otra institución o país requiere validación con datos locales."
        )
    except Exception as exc:
        st.error("Ocurrió un error al realizar la predicción. Revisa que el modelo corresponda al notebook actualizado.")
        st.exception(exc)

st.divider()
st.caption("Proyecto académico – Especialización en Inteligencia Artificial | CUN")
