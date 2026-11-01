ACA 1 - PREDICCION DE DESERCION ESTUDIANTIL
Aplicacion Streamlit con explicaciones por campo

CONTENIDO
- app.py: interfaz Streamlit en espanol.
- modelo_desercion.pkl: modelo entrenado en Google Colab con scikit-learn 1.9.1.
- requirements.txt: dependencias del proyecto.

CAMBIOS DE ESTA VERSION
- Cada uno de los 30 campos tiene una explicacion especifica en el icono de ayuda (?).
- Los desplegables muestran descripciones en espanol y conservan el codigo original esperado por el modelo.
- Se explica que el modelo requiere 30 variables: 27 datos del estudiante y 3 indicadores macroeconomicos.
- Los indicadores macroeconomicos usan inicialmente las medianas del dataset historico (desempleo 11.1, inflacion 1.4, PIB 0.32). Pueden personalizarse si se conocen los valores del periodo pertinente.
- Se advierte que el dataset procede de una institucion de Portugal; los indicadores no son cifras actuales de Colombia.
- Antes de calcular, el usuario debe confirmar que reviso los valores de referencia.

REQUISITOS
Python 3.14.4 (u otra version compatible con las dependencias instaladas)
Scikit-learn 1.9.1

EJECUCION EN WINDOWS POWERSHELL
1. Abrir PowerShell en la carpeta del proyecto.
2. Activar el entorno virtual existente:
   .\venv\Scripts\Activate.ps1
3. Ejecutar la aplicacion:
   python -m streamlit run app.py

Si aun no existe el entorno virtual:
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   python -m streamlit run app.py

NOTA ACADEMICA
Es una herramienta experimental basada en datos historicos de una institucion portuguesa. No representa una certeza ni debe utilizarse como unico criterio para tomar decisiones sobre estudiantes. Su aplicacion en otra institucion o pais requiere validacion con datos locales.
