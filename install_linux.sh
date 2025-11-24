#!/bin/bash
# ============================================================
# Script de configuración automática del entorno virtual para el Slicer 3D
# Usa un archivo requirements.txt existente
# ============================================================

VENV_DIR="venv"
REQ_FILE="requirements.txt"

echo "=== Configuración automática del entorno virtual para el Slicer 3D ==="

#Comprobar si Python está instalado
if ! command -v python3 &> /dev/null; then
    echo " Python3 no está instalado. Instálalo antes de continuar."
    exit 1
fi

# Comprobar si existe requirements.txt
if [ ! -f "$REQ_FILE" ]; then
    echo " No se encontró el archivo $REQ_FILE."
    echo "   Por favor, crea uno con 'pip freeze > requirements.txt' o instala manualmente."
    exit 1
fi

#Crear entorno virtual si no existe
if [ ! -d "$VENV_DIR" ]; then
    echo " Creando entorno virtual en: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    echo " El entorno virtual ya existe, se reutilizará."
fi

#  Activar el entorno virtual
echo " Activando entorno virtual..."
source "$VENV_DIR/bin/activate"

# Actualizar pip
echo "  Actualizando pip..."
pip install --upgrade pip

#Instalar dependencias desde requirements.txt
echo " Instalando dependencias desde $REQ_FILE..."
pip install -r "$REQ_FILE"

#Mensaje final
echo "Entorno configurado correctamente."
echo "Para activarlo manualmente en el futuro, ejecuta:"
echo "   source venv/bin/activate"
echo "============================================================"