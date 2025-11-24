#!/bin/bash
# ============================================================
# Ejecutar el Slicer 3D usando el entorno virtual (Linux)
# ============================================================

VENV_DIR="venv"
SCRIPT="GUITFG.py"

echo "=== Iniciando el Slicer 3D ==="

#  Comprobar que existe el entorno virtual
if[ ! -d "$VENV_DIR" ]; then
    echo " No se encontró el entorno virtual '$VENV_DIR'."
    echo "   Ejecuta primero el script de instalación:"
    echo "   ./setup_env.sh"
    exit 1
fi

#  Activar entorno virtual
echo " Activando entorno virtual..."
source "$VENV_DIR/bin/activate"

#  Comprobar que existe el script principal
if [ ! -f "$SCRIPT" ]; then
    echo " No se encontró el archivo $SCRIPT."
    echo "   Asegúrate de que GUITFG.py está en esta carpeta."
    deactivate
    exit 1
fi

#  Ejecutar el programa
echo " Ejecutando aplicación..."
python3 "$SCRIPT"

echo " "
echo " Ejecución finalizada."
echo "Para volver a usar la aplicación:  ./run_slicer.sh"

# Desactivar entorno virtual (opcional pero limpio)
deactivate
