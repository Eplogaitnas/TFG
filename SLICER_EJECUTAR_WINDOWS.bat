@echo off
REM ============================================================
REM Ejecutar el Slicer 3D usando el entorno virtual (Windows)
REM ============================================================

set VENV_DIR=venv
set SCRIPT=GUITFG.py

echo === Iniciando el Slicer 3D ===

REM Comprobar que existe el entorno virtual
if not exist "%VENV_DIR%" (
    echo  No se encontró el entorno virtual "%VENV_DIR%".
    echo Ejecuta primero "setup_env.bat" para crearlo e instalar dependencias.
    pause
    exit /b
)

REM Activar el entorno virtual
echo  Activando entorno virtual...
call "%VENV_DIR%\Scripts\activate"

REM Comprobar que el script existe
if not exist "%SCRIPT%" (
    echo  No se encontró el archivo %SCRIPT%.
    echo Asegúrate de que el archivo GUITFG.py está en la carpeta del proyecto.
    pause
    exit /b
)

REM Ejecutar el programa
echo  Ejecutando aplicación...
python "%SCRIPT%"

echo.
echo  Ejecución finalizada.
pause
