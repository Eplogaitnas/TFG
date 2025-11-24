@echo off
REM ============================================================
REM Script de configuración automática del entorno virtual (Windows)
REM Usa un archivo requirements.txt existente
REM ============================================================

set VENV_DIR=venv
set REQ_FILE=requirements.txt

echo === Configuración automática del entorno virtual para el Slicer 3D ===

REM Comprobar que Python está instalado
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo  Python no está instalado o no está en el PATH.
    echo Instálalo desde https://www.python.org/ y vuelve a ejecutar este script.
    pause
    exit /b
)

REM Comprobar que existe requirements.txt
if not exist "%REQ_FILE%" (
    echo  No se encontró el archivo %REQ_FILE%.
    echo Crea uno con "pip freeze > requirements.txt" o instala manualmente.
    pause
    exit /b
)

REM Crear entorno virtual si no existe
if not exist "%VENV_DIR%" (
    echo  Creando entorno virtual en: %VENV_DIR%
    python -m venv "%VENV_DIR%"
) else (
    echo  El entorno virtual ya existe, se reutilizará.
)

REM Activar entorno virtual
echo  Activando entorno virtual...
call "%VENV_DIR%\Scripts\activate"

REM Actualizar pip
echo   Actualizando pip...
python -m pip install --upgrade pip

REM Instalar dependencias desde requirements.txt
echo  Instalando dependencias desde %REQ_FILE%...
pip install -r "%REQ_FILE%"

REM  Mensaje final
echo ============================================================
echo  Entorno configurado correctamente.
echo Para activarlo manualmente en el futuro, ejecuta:
echo     venv\Scripts\activate
echo ============================================================

pause
