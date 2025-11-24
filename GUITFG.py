from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QTableWidget, QTableWidgetItem, QDialog, QFormLayout, QLineEdit, QMessageBox,
    QComboBox, QGroupBox, QOpenGLWidget, QCheckBox, QTabWidget, QDialogButtonBox, QSlider, QInputDialog,
    QToolButton
)
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QFont
import json
import os
import numpy as np
import shutil
import cv2
from OpenGL.GL import (
    glClear, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    glUseProgram, glGetUniformLocation, glUniformMatrix4fv, glUniform3f,
    GL_FALSE, glViewport, glDisable, glEnable, GL_DEPTH_TEST, glPolygonMode, GL_FRONT_AND_BACK, GL_LINE, GL_FILL
)
from Slicer_TFG import Slicer as SlicerOptimizado, SlicerNormal, SlicerSettings, Material
from gl_utils import load_shader, GLModel

class LayerOptionsDialog(QDialog):
    def __init__(self, slicer_settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Opciones de capa")
        layout = QFormLayout()

        self.layer_thickness_input = QLineEdit(str(slicer_settings.layer_thickness))
        self.first_layer_thickness_input = QLineEdit(str(slicer_settings.first_layer_thickness))

        layout.addRow("Grosor de capa:", self.layer_thickness_input)
        layout.addRow("Grosor de primera capa:", self.first_layer_thickness_input)

        # Botones Aceptar / Cancelar
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)
        self.slicer_settings = slicer_settings

    def accept(self):
        try:
            self.slicer_settings.layer_thickness = float(self.layer_thickness_input.text())
            self.slicer_settings.first_layer_thickness = float(self.first_layer_thickness_input.text())
            super().accept()
        except ValueError:
            QMessageBox.warning(self, "Error", "Introduce valores numéricos válidos.")

class Preview3DWidget(QOpenGLWidget):
    def __init__(self, materials=None, parent=None):
        super().__init__(parent)
        self.models = [None] * 5
        self.last_pos = None
        self.rot_x = 90
        self.rot_y = 0
        # escala por ejes: [sx, sy, sz]
        self.scale = [0.250, 0.250, 0.250]
        self.object_color = (1., 0., 0.)
        self.materials = materials if materials is not None else []

    def set_scale(self, sx, sy=None, sz=None):
        """
        set_scale(s) -> uniforme
        set_scale(sx, sy, sz) -> por ejes
        """
        if sy is None and sz is None:
            # uniforme
            self.scale = [float(sx)] * 3
        else:
            self.scale = [float(sx), float(sy), float(sz)]
        self.update()

    def set_color(self, r, g, b):
        self.object_color = (r, g, b)
        self.update()

    def load_model(self, filename, slot=0):
        """Carga un modelo en el slot indicado (0-4)."""
        if 0 <= slot < 5:
            self.models[slot] = GLModel(filename)
            self.models[slot].filename = filename
            self.update()

    def clear_models(self):
        self.models = [None] * 5
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_pos is not None:
            dx = event.x() - self.last_pos.x()
            dy = event.y() - self.last_pos.y()
            self.rot_x += dy
            self.rot_y += dx
            self.last_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        self.last_pos = None

    def paintGL(self):
        if not self.shader_program or self.shader_program == 0:
            print("Error: shader_program no válido")
            return
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        try:
            glUseProgram(self.shader_program)
        except Exception as e:
            print(f"Error al activar el shader: {e}")
            return

        # soporta escala por ejes
        try:
            sx, sy, sz = self.scale
        except Exception:
            sx = sy = sz = float(self.scale)
        s = np.array([
            [sx, 0,  0,  0],
            [0,  sy, 0,  0],
            [0,  0,  sz, 0],
            [0,  0,  0,  1]
        ], dtype=np.float32)
        rx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(np.radians(self.rot_x)), -np.sin(np.radians(self.rot_x)), 0],
            [0, np.sin(np.radians(self.rot_x)), np.cos(np.radians(self.rot_x)), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        ry = np.array([
            [np.cos(np.radians(self.rot_y)), 0, np.sin(np.radians(self.rot_y)), 0],
            [0, 1, 0, 0],
            [-np.sin(np.radians(self.rot_y)), 0, np.cos(np.radians(self.rot_y)), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        view = np.identity(4, dtype=np.float32)
        projection = np.identity(4, dtype=np.float32)

        model_loc = glGetUniformLocation(self.shader_program, "model")
        view_loc = glGetUniformLocation(self.shader_program, "view")
        proj_loc = glGetUniformLocation(self.shader_program, "projection")
        light_loc = glGetUniformLocation(self.shader_program, "lightPos")
        viewpos_loc = glGetUniformLocation(self.shader_program, "viewPos")
        color_loc = glGetUniformLocation(self.shader_program, "objectColor")

        # asegurar contigüidad/dtype y comprobar loc != -1
        if view_loc != -1:
            glUniformMatrix4fv(view_loc, 1, GL_FALSE, np.ascontiguousarray(view, dtype=np.float32))
        if proj_loc != -1:
            glUniformMatrix4fv(proj_loc, 1, GL_FALSE, np.ascontiguousarray(projection, dtype=np.float32))
        glUniform3f(light_loc, 0, 0, -0.5)
        glUniform3f(viewpos_loc, 0, 0, 0)

        # Dibuja todos los modelos cargados
        for idx, model_obj in enumerate(self.models):
            if model_obj:
                min_z = 0.0
                try:
                    verts = np.array(model_obj.vertices)
                    min_z = np.min(verts[:, 2])
                except Exception:
                    min_z = 0.0

                model = s @ rx @ ry
                model[2, 3] = -min_z if min_z < 0 else 0.0
                if model_loc != -1:
                    glUniformMatrix4fv(model_loc, 1, GL_FALSE, np.ascontiguousarray(model, dtype=np.float32))

                # Usa el color del material correspondiente
                if idx < len(self.materials):
                    mat_color = self.materials[idx].color
                else:
                    mat_color = "Rojo"

                color_map = {
                    "Rojo": (1.0, 0.0, 0.0),
                    "Verde": (0.0, 1.0, 0.0),
                    "Azul": (0.0, 0.0, 1.0),
                    "Amarillo": (1.0, 1.0, 0.0),
                    "Blanco": (1.0, 1.0, 1.0)
                }
                rgb = color_map.get(mat_color, (1.0, 0.0, 0.0))
                glUniform3f(color_loc, *rgb)
                model_obj.paint()

                # --- Dibuja la malla encima ---
                glDisable(GL_DEPTH_TEST)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glUniform3f(color_loc, 0.0, 0.0, 0.0)  # Negro para la malla
                if hasattr(model_obj, "paint_wireframe"):
                    model_obj.paint_wireframe()
                else:
                    model_obj.paint()
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                glEnable(GL_DEPTH_TEST)

        glUseProgram(0)

    def initializeGL(self):
        from OpenGL.GL import glClearColor, glEnable, GL_DEPTH_TEST
        glClearColor(0.2, 0.2, 0.2, 1)
        glEnable(GL_DEPTH_TEST)
        self.shader_program = load_shader("shaderTFG.vert", "shaderTFG.frag")
        if not self.shader_program or self.shader_program == 0:
            print("Error: Shader program no válido")

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)


class SlicerThread(QThread):
    def __init__(self, slicer):
        super().__init__()
        self.slicer = slicer

    def run(self):
        self.slicer.run()  # Ejecuta en el hilo principal


class MainWindow(QMainWindow):
    def __init__(self, slicer_settings):
        super().__init__()
        self.setWindowTitle("Bioprinter Slicer")
        self.slicer_settings = slicer_settings

        central = QWidget()
        main_layout = QHBoxLayout()

        # Panel previsualización
        preview_panel = QVBoxLayout()
        self.preview_3d = Preview3DWidget(materials=self.slicer_settings.materials)
        self.preview_3d.setMinimumHeight(400)
        preview_panel.addWidget(self.preview_3d)
        # valor de referencia para cambios "Uniforme" (multiplicador relativo)
        self.last_uniform_value = sum(self.preview_3d.scale) / 3.0

        self.load_buttons = []
        for i in range(5):
            h = QHBoxLayout()
            btn = QPushButton(f"Cargar Material en slot {i+1}")
            btn.clicked.connect(lambda _, slot=i: self.load_stl(slot))
            h.addWidget(btn)
            self.load_buttons.append(btn)

            del_btn = QPushButton("Borrar")
            del_btn.clicked.connect(lambda _, slot=i: self.on_delete_model(slot))
            h.addWidget(del_btn)

            preview_panel.addLayout(h)

        preview_widget = QWidget()
        preview_widget.setLayout(preview_panel)
        preview_widget.setMinimumWidth(400)
        preview_widget.setMaximumWidth(800)
        main_layout.addWidget(preview_widget, stretch=3)

        # Panel de opciones y materiales
        materials_panel = QVBoxLayout()

        options_group = QGroupBox("Opciones")
        options_layout = QVBoxLayout()
        self.options_button = QPushButton("Ajustes de capa")
        self.options_button.clicked.connect(self.open_layer_options)
        options_layout.addWidget(self.options_button)

        #  Checkbox para elegir modo optimizado
        self.optimized_checkbox = QCheckBox("Usar slicing optimizado")
        self.optimized_checkbox.stateChanged.connect(self.on_checkbox_toggled)
        # coloca el checkbox y el botón "Separar componentes" en la misma fila,
        # de forma que el botón quede a la derecha del checkbox.
        h_opt = QHBoxLayout()
        h_opt.addWidget(self.optimized_checkbox)

        # Cambiado a QCheckBox para coherencia UI
        self.split_components_cb = QCheckBox("Separar componentes")
        self.split_components_cb.setChecked(False)
        self.split_components_cb.toggled.connect(self.on_toggle_split_components)
        h_opt.addWidget(self.split_components_cb)

        options_layout.addLayout(h_opt)

        # --- Transformaciones: Escalado ---
        transform_group = QGroupBox("Transformaciones")
        transform_layout = QHBoxLayout()

        self.scale_axis_combo = QComboBox()
        self.scale_axis_combo.addItems(["X", "Y", "Z", "Uniforme"])
        try:
            # preview_3d.scale es [sx, sy, sz]
            current_axis_value = self.preview_3d.scale[0]
        except Exception:
            current_axis_value = 1.0
        self.scale_axis_combo.setCurrentIndex(3)  # por defecto Uniforme

        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setMinimum(1)    # 0.01
        self.scale_slider.setMaximum(500)  # 5.00
        self.scale_slider.setValue(int(current_axis_value * 100))
        self.scale_slider.setTickInterval(10)
        self.scale_slider.setTickPosition(QSlider.TicksBelow)

        self.scale_input = QLineEdit(str(current_axis_value))
        self.scale_input.setMaximumWidth(60)

        self.scale_lock_btn = QToolButton()
        self.scale_lock_btn.setCheckable(True)
        self.scale_lock_btn.setText("Restaurar🔓")  # unlocked icon/text
        self.scale_lock_btn.setToolTip("Bloquear escala entre ejes")

        # Conexiones
        self.scale_slider.valueChanged.connect(self.update_scale_from_slider)
        self.scale_input.editingFinished.connect(self.update_scale_from_input)
        self.scale_axis_combo.currentIndexChanged.connect(self.on_scale_axis_changed)
        self.scale_lock_btn.toggled.connect(self.on_scale_lock_toggled)

        transform_layout.addWidget(QLabel("Eje:"))
        transform_layout.addWidget(self.scale_axis_combo)
        transform_layout.addWidget(QLabel("Escala:"))
        transform_layout.addWidget(self.scale_slider)
        transform_layout.addWidget(self.scale_input)
        transform_layout.addWidget(self.scale_lock_btn)

        transform_group.setLayout(transform_layout)
        options_layout.addWidget(transform_group)

        options_group.setLayout(options_layout)
        materials_panel.addWidget(options_group)

        material_group = QGroupBox("Materiales")
        material_group.setMinimumHeight(450)
        material_layout = QVBoxLayout()

        # Reemplaza el selector por pestañas
        self.material_tabs = QTabWidget()
        self.material_tabs.setTabPosition(QTabWidget.North)
        self.material_tabs.currentChanged.connect(self.update_material_info)

        # Título grande para cada campo en cada material
        label_font = QFont()
        label_font.setPointSize(14)
        label_font.setBold(True)

        # Crea una pestaña por cada material
        self.material_edit_widgets = []
        for mat in self.slicer_settings.materials:
            tab = QWidget()
            tab_layout = QVBoxLayout()

            name_label = QLabel("Nombre de material:")
            name_label.setFont(label_font)
            tab_layout.addWidget(name_label)
            name_input = QLineEdit(mat.nombre)
            name_input.setMinimumHeight(32)
            name_input.setFont(QFont("", 12))
            tab_layout.addWidget(name_input)

            expo_label = QLabel("Exposición:")
            expo_label.setFont(label_font)
            tab_layout.addWidget(expo_label)
            expo_input = QLineEdit(str(mat.tiempo_exposicion))
            expo_input.setMinimumHeight(32)
            expo_input.setFont(QFont("", 12))
            tab_layout.addWidget(expo_input)

            cura_label = QLabel("Cura:")
            cura_label.setFont(label_font)
            tab_layout.addWidget(cura_label)
            cura_input = QLineEdit(str(mat.tiempo_cura))
            cura_input.setMinimumHeight(32)
            cura_input.setFont(QFont("", 12))
            tab_layout.addWidget(cura_input)

            # --- Campo color NO modificable ---
            color_label = QLabel("Color:")
            color_label.setFont(label_font)
            tab_layout.addWidget(color_label)
            color_display = QLabel(str(mat.color))  # Usa QLabel en vez de QLineEdit
            color_display.setMinimumHeight(32)
            color_display.setFont(QFont("", 12))
            tab_layout.addWidget(color_display)

            tab.setLayout(tab_layout)
            self.material_tabs.addTab(tab, mat.nombre)
            self.material_edit_widgets.append((name_input, expo_input, cura_input, color_display))

        material_layout.addWidget(self.material_tabs)

        # Botón para guardar cambios
        self.save_material_button = QPushButton("Guardar cambios")
        self.save_material_button.clicked.connect(self.save_material_changes)
        material_layout.addWidget(self.save_material_button)

        # Botón para guardar solo el material seleccionado
        self.save_material_json_button = QPushButton("Guardar material seleccionado en JSON")
        self.save_material_json_button.clicked.connect(self.save_material_to_json)
        material_layout.addWidget(self.save_material_json_button)

        # Botón para cargar solo en el material seleccionado
        self.load_material_json_button = QPushButton("Cargar material en seleccionado desde JSON")
        self.load_material_json_button.clicked.connect(self.load_material_from_json)
        material_layout.addWidget(self.load_material_json_button)

        material_group.setLayout(material_layout)
        materials_panel.addWidget(material_group)

        self.slice_button = QPushButton("Slice")
        self.slice_button.clicked.connect(self.slice)
        materials_panel.addWidget(self.slice_button)

        materials_widget = QWidget()
        materials_widget.setLayout(materials_panel)
        main_layout.addWidget(materials_widget, stretch=5)

        central.setLayout(main_layout)
        self.setCentralWidget(central)

        self.material_modified = [False for _ in self.slicer_settings.materials]
        self.update_material_info(0)
        self.material_tabs.currentChanged.connect(self.on_tab_changed)

        # (removido duplicado, se usa self.split_components_cb junto al checkbox de optimizado)

    def update_scale_from_slider(self, value):
        """Actualiza la escala según el slider; modifica solo el eje seleccionado a menos que sea 'Uniforme' o esté bloqueado.
        Si está 'Uniforme' y NO está bloqueado, ajusta cada eje proporcionalmente respecto a last_uniform_value."""
        s = value / 100.0
        axis = self.get_selected_axis_index()  # 0=X,1=Y,2=Z, None=Uniforme
        # obtiene valores actuales de escala
        try:
            sx, sy, sz = list(self.preview_3d.scale)
        except Exception:
            try:
                v = float(self.preview_3d.scale)
            except Exception:
                v = 1.0
            sx = sy = sz = v

        if axis is None:
            # Uniforme selected
            if self.scale_lock_btn.isChecked():
                # bloqueo -> imponer valor a los tres ejes
                sx = sy = sz = s
                self.last_uniform_value = s
            else:
                # sin bloqueo -> ajustar proporcionalmente respecto a last_uniform_value
                # evita división por cero
                prev = self.last_uniform_value if getattr(self, "last_uniform_value", 0.0) > 0 else (sx + sy + sz) / 3.0
                if prev <= 0:
                    prev = 1.0
                ratio = s / prev
                sx *= ratio
                sy *= ratio
                sz *= ratio
                # actualiza referencia uniforme al nuevo valor
                self.last_uniform_value = s
        else:
            # modifica solo el eje seleccionado
            if axis == 0:
                sx = s
            elif axis == 1:
                sy = s
            elif axis == 2:
                sz = s
            # si está bloqueado, propaga a los demás ejes
            if self.scale_lock_btn.isChecked():
                sx = sy = sz = [sx, sy, sz][axis]

        # aplica la escala al preview
        try:
            self.preview_3d.set_scale(sx, sy, sz)
        except Exception:
            self.preview_3d.scale = [sx, sy, sz]
            self.preview_3d.update()

        # muestra en el input el valor del eje seleccionado (o uniforme)
        display_val = sx if axis == 0 else (sy if axis == 1 else (sz if axis == 2 else s))
        self.scale_input.setText(f"{display_val:.2f}")
        self.preview_3d.update()

    def update_scale_from_input(self):
        """Lee el input de texto y aplica escala (solo eje seleccionado salvo Uniforme/bloqueo).
        Si Uniforme y no bloqueado, aplica cambio proporcional a los ejes respecto a last_uniform_value."""
        try:
            entered = float(self.scale_input.text())
        except ValueError:
            # restaura el valor correcto desde el slider
            v = self.scale_slider.value() / 100.0
            self.scale_input.setText(f"{v:.2f}")
            return

        s_clamped = max(0.01, min(entered, 5.0))
        self.scale_slider.setValue(int(s_clamped * 100))

        axis = self.get_selected_axis_index()
        try:
            sx, sy, sz = list(self.preview_3d.scale)
        except Exception:
            try:
                v = float(self.preview_3d.scale)
            except Exception:
                v = 1.0
            sx = sy = sz = v

        if axis is None:
            if self.scale_lock_btn.isChecked():
                sx = sy = sz = s_clamped
                self.last_uniform_value = s_clamped
            else:
                prev = self.last_uniform_value if getattr(self, "last_uniform_value", 0.0) > 0 else (sx + sy + sz) / 3.0
                if prev <= 0:
                    prev = 1.0
                ratio = s_clamped / prev
                sx *= ratio
                sy *= ratio
                sz *= ratio
                self.last_uniform_value = s_clamped
        else:
            if axis == 0:
                sx = s_clamped
            elif axis == 1:
                sy = s_clamped
            elif axis == 2:
                sz = s_clamped
            if self.scale_lock_btn.isChecked():
                sx = sy = sz = [sx, sy, sz][axis]

        try:
            self.preview_3d.set_scale(sx, sy, sz)
        except Exception:
            self.preview_3d.scale = [sx, sy, sz]
            self.preview_3d.update()

        self.preview_3d.update()
        self.scale_input.setText(f"{s_clamped:.2f}")

    def get_selected_axis_index(self):
        """Devuelve 0=X,1=Y,2=Z, None=Uniforme."""
        idx = self.scale_axis_combo.currentIndex()
        if idx in (0, 1, 2):
            return idx
        return None

    def on_scale_axis_changed(self, _idx):
        """Sincroniza el slider/input con el valor actual del eje seleccionado.
        Si se selecciona 'Uniforme', muestra la referencia last_uniform_value (no resetea ejes)."""
        axis = self.get_selected_axis_index()
        try:
            sx, sy, sz = list(self.preview_3d.scale)
        except Exception:
            try:
                v = float(self.preview_3d.scale)
            except Exception:
                v = 1.0
            sx = sy = sz = v

        if axis is None:
            # Uniforme: muestra la referencia (no impone cambio)
            val = getattr(self, "last_uniform_value", (sx + sy + sz) / 3.0)
        else:
            val = [sx, sy, sz][axis]

        # actualiza controles sin disparar señales
        self.scale_slider.blockSignals(True)
        self.scale_input.blockSignals(True)
        self.scale_slider.setValue(int(val * 100))
        self.scale_input.setText(f"{val:.2f}")
        self.scale_slider.blockSignals(False)
        self.scale_input.blockSignals(False)

    def on_scale_lock_toggled(self, locked):
        """Si se activa bloqueo, sincroniza todos los ejes al valor del eje seleccionado (o al slider si Uniforme)."""
        self.scale_lock_btn.setText("Restaurado 🔒" if locked else "Restaurar 🔓")
        try:
            sx, sy, sz = list(self.preview_3d.scale)
        except Exception:
            try:
                v = float(self.preview_3d.scale)
            except Exception:
                v = 1.0
            sx = sy = sz = v

        axis = self.get_selected_axis_index()
        if locked:
            if axis is None:
                # uniforme -> usa valor del slider
                s = self.scale_slider.value() / 100.0
            else:
                s = [sx, sy, sz][axis]
            try:
                self.preview_3d.set_scale(s, s, s)
            except Exception:
                self.preview_3d.scale = [s, s, s]
            # actualiza controles
            self.scale_slider.setValue(int(s * 100))
            self.scale_input.setText(f"{s:.2f}")
        # si se desbloquea, no cambiamos escalas existentes; solo actualizamos icono/estado
        self.preview_3d.update()

    def on_checkbox_toggled(self, state):
        if self.optimized_checkbox.isChecked():
            QMessageBox.information(self, "Modo de slicing", "Has activado el modo OPTIMIZADO.\n\n Atención: no todas las impresoras son compatibles con este algoritmo. "
            "Verifica la compatibilidad de tu impresora antes de continuar. (Debe poder decodificar las imagenes resultantes)")
        else:
            QMessageBox.information(self, "Modo de slicing", "Se ha activado el modo NORMAL.")
    
    def open_layer_options(self):
        dialog = LayerOptionsDialog(self.slicer_settings, self)
        if dialog.exec_() == QDialog.Accepted:
            QMessageBox.information(self, "Opciones", "Valores de capa actualizados.")

    def load_stl(self, slot=0):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, f"Cargar archivo STL para modelo {slot+1}", "", "Archivos STL (*.stl);;Todos los archivos (*)", options=options)
        if file_name:
            self.preview_3d.load_model(file_name, slot=slot)

    def on_delete_model(self, slot):
        if 0 <= slot < len(self.preview_3d.models):
            self.preview_3d.models[slot] = None
            self.preview_3d.update()

    def slice(self):
        # Obtén todos los modelos cargados
        stl_filenames = []
        for model_obj in self.preview_3d.models:
            if model_obj:
                try:
                    stl_filenames.append(model_obj.filename)
                except AttributeError:
                    continue

        if not stl_filenames:
            QMessageBox.warning(self, "Slice", "Primero carga al menos un modelo STL.")
            return

        # --- Pide el nombre del archivo al usuario ---
        nombre, ok = QInputDialog.getText(self, "Nombre de proyecto", "¿Cómo deseas llamar al archivo?")
        if not ok or not nombre.strip():
            QMessageBox.warning(self, "Slice", "Debes introducir un nombre para el proyecto.")
            return
        nombre = nombre.strip()

        # --- Crea la carpeta metadatos/<nombre> y metadatos/<nombre>/imagenes ---
        base_dir = os.path.join(os.getcwd(), "metadatos")
        proyecto_dir = os.path.join(base_dir, nombre)
        imagenes_dir = os.path.join(proyecto_dir, "imagenes")
        os.makedirs(imagenes_dir, exist_ok=True)

        # --- Ruta para el JSON ---
        metadata_path = os.path.join(proyecto_dir, f"{nombre}.json")

        # --- Asegura que model_transforms tiene suficientes elementos ---
        num_models = len(stl_filenames)
        if len(self.slicer_settings.model_transforms) < num_models:
            from Slicer_TFG import ModelTransform
            self.slicer_settings.model_transforms += [
                ModelTransform() for _ in range(num_models - len(self.slicer_settings.model_transforms))
            ]
        elif len(self.slicer_settings.model_transforms) > num_models:
            self.slicer_settings.model_transforms = self.slicer_settings.model_transforms[:num_models]

        # --- Ejecuta el slicer ---
        # Asegura que la opción de separar componentes se transmite al Slicer
        try:
            self.slicer_settings.split_components = bool(self.split_components_cb.isChecked())
        except Exception:
            self.slicer_settings.split_components = False

        if self.optimized_checkbox.isChecked():
            slicer = SlicerOptimizado(stl_filenames, metadata_path, self.slicer_settings)
        else:
            slicer = SlicerNormal(stl_filenames, metadata_path, self.slicer_settings)

        slicer.finished.connect(lambda: QMessageBox.information(self, "Slice", "Modelo(s) cortado(s) con éxito."))
        slicer.progress.connect(lambda p: self.statusBar().showMessage(f"Progreso slicing: {p}%"))
        slicer.run()
        QMessageBox.information(self, "Slice", "Modelo(s) cortado(s) con éxito.")

        # --- Mueve las imágenes generadas (PNG) a metadatos/<nombre>/imagenes ---
        # Busca imágenes en la carpeta del proyecto
        for fname in os.listdir(proyecto_dir):
            if fname.endswith(".png"):
                shutil.move(os.path.join(proyecto_dir, fname), os.path.join(imagenes_dir, fname))

        # Si la casilla "Separar componentes" está activada, ejecutar split automático en la carpeta de imágenes
        try:
            if getattr(self, "split_components_cb", None) and self.split_components_cb.isChecked():
                self.split_components_in_dir(imagenes_dir)
        except Exception:
            pass

        # Elimina los modelos STL de la previsualización
        self.preview_3d.clear_models()
        self.preview_3d.set_scale(0.250)
        self.preview_3d.rot_x = 90
        self.preview_3d.rot_y = 0
        self.preview_3d.set_color(1.0, 0.0, 0.0)
        self.preview_3d.update()

        # --- Reinicia la aplicación manteniendo materiales y configuración ---
        size = self.size()
        pos = self.pos()
        new_window = MainWindow(self.slicer_settings)
        new_window.resize(size)
        new_window.move(pos)
        new_window.show()
        self.close()

    def mark_material_modified(self, index):
        if 0 <= index < len(self.material_modified):
            self.material_modified[index] = True

    def save_material_changes(self):
        index = self.material_tabs.currentIndex()
        if index < 0:
            QMessageBox.warning(self, "Error", "No hay material seleccionado.")
            return

        name_input, expo_input, cura_input, _ = self.material_edit_widgets[index]
        new_name = name_input.text()
        mat = self.slicer_settings.materials[index]
        try:
            mat.tiempo_exposicion = float(expo_input.text())
            mat.tiempo_cura = float(cura_input.text())
        except ValueError:
            QMessageBox.warning(self, "Error", "Exposición y Cura deben ser valores numéricos.")
            return

        duplicate_count = sum(1 for i, m in enumerate(self.slicer_settings.materials)
                              if (m.nombre == new_name and i != index))
        if duplicate_count > 0:
            QMessageBox.information(self, "Aviso", f"Ya existe otro material con el nombre '{new_name}'. Considera cambiarlo para evitar confusiones.")

        mat.nombre = new_name
        self.material_tabs.setTabText(index, new_name)
        self.update_material_info(index)
        self.material_modified[index] = False  # Se ha guardado
        QMessageBox.information(self, "Material", f"Se han guardado los cambios en {mat.nombre}.")

    def update_material_info(self, index):
        if index < 0 or index >= len(self.material_edit_widgets):
            return
        mat = self.slicer_settings.materials[index]
        name_input, expo_input, cura_input, color_display = self.material_edit_widgets[index]
        name_input.setText(str(mat.nombre))
        expo_input.setText(str(mat.tiempo_exposicion))
        cura_input.setText(str(mat.tiempo_cura))
        color_display.setText(str(mat.color))  # Actualiza el color
        # Marca como no modificado al refrescar
        self.material_modified[index] = False

        # Conecta los cambios para marcar como modificado
        name_input.textChanged.connect(lambda: self.mark_material_modified(index))
        expo_input.textChanged.connect(lambda: self.mark_material_modified(index))
        cura_input.textChanged.connect(lambda: self.mark_material_modified(index))

    def on_tab_changed(self, new_index):
        old_index = getattr(self, "_last_tab_index", 0)
        if 0 <= old_index < len(self.material_modified) and self.material_modified[old_index]:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Aviso")
            msg_box.setText("Hay cambios sin guardar en el material actual.\n¿Quieres cambiar de pestaña igualmente? \n Los cambios se perderán.")
            aceptar_btn = msg_box.addButton("Aceptar", QMessageBox.YesRole)
            cancelar_btn = msg_box.addButton("Cancelar", QMessageBox.NoRole)
            msg_box.setDefaultButton(aceptar_btn)
            msg_box.exec_()
            if msg_box.clickedButton() == cancelar_btn:
                self.material_tabs.blockSignals(True)
                self.material_tabs.setCurrentIndex(old_index)
                self.material_tabs.blockSignals(False)
                return
        self._last_tab_index = new_index

    def save_material_to_json(self):
        index = self.material_tabs.currentIndex()
        if index < 0:
            QMessageBox.warning(self, "Error", "No hay material seleccionado.")
            return
        mat = self.slicer_settings.materials[index]

        materiales_dir = os.path.join(os.getcwd(), "materiales")
        if not os.path.exists(materiales_dir):
            os.makedirs(materiales_dir)

        file_name = os.path.join(materiales_dir, f"{mat.nombre}.material")
        data = {
            "nombre": mat.nombre,
            "tiempo_exposicion": mat.tiempo_exposicion,
            "tiempo_cura": mat.tiempo_cura
            # color intentionally not saved
        }
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        QMessageBox.information(self, "Material", f"Material guardado en {file_name}.")

    def load_material_from_json(self):
        index = self.material_tabs.currentIndex()
        if index < 0:
            QMessageBox.warning(self, "Error", "No hay material seleccionado.")
            return

        materiales_dir = os.path.join(os.getcwd(), "materiales")
        dialog = MaterialSelectDialog(materiales_dir, self)
        if dialog.exec_() == QDialog.Accepted and dialog.selected_file:
            file_name = os.path.join(materiales_dir, dialog.selected_file)
            try:
                with open(file_name, "r", encoding="utf-8") as f:
                    mat_data = json.load(f)
                new_name = mat_data.get("nombre", "")
                duplicate_count = sum(1 for i, m in enumerate(self.slicer_settings.materials)
                                      if (m.nombre == new_name and i != index))
                if duplicate_count > 0:
                    QMessageBox.information(self, "Aviso", f"Ya existe otro material con el nombre '{new_name}'. Considera cambiarlo para evitar confusiones.")
                mat = self.slicer_settings.materials[index]
                mat.nombre = new_name
                mat.tiempo_exposicion = mat_data.get("tiempo_exposicion", mat.tiempo_exposicion)
                mat.tiempo_cura = mat_data.get("tiempo_cura", mat.tiempo_cura)
                mat.color = mat_data.get("color", getattr(mat, "color", ""))
                self.material_tabs.setTabText(index, new_name)
                self.update_material_info(index)
                QMessageBox.information(self, "Material", "Material cargado y actualizado correctamente.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"No se pudo cargar el archivo: {e}")

    def on_toggle_split_components(self, checked):
        # assume self.slicer_settings existe y es la misma instancia que pasas a Slicer
        try:
            self.slicer_settings.split_components = bool(checked)
        except Exception:
            pass

    def split_components_in_dir(self, images_dir):
        """Separa cada MASK_*.png en componentes conectados y organiza en carpetas 'pre' y 'post'.

        Comportamiento:
        - Si una máscara se divide en >1 componente: el PNG original se mueve a pre/ y
          cada componente individual se guarda en post/ como *_compN.png.
        - Si una máscara NO se divide (0 o 1 componente): el PNG original se mueve directamente a post/
          (no se copia en pre).
        """
        if not os.path.isdir(images_dir):
            QMessageBox.warning(self, "Split components", f"No existe la carpeta: {images_dir}")
            return

        pre_dir = os.path.join(images_dir, "pre")
        post_dir = os.path.join(images_dir, "post")
        os.makedirs(pre_dir, exist_ok=True)
        os.makedirs(post_dir, exist_ok=True)

        # Busca ficheros de máscara en la carpeta principal (antes de mover)
        mask_files = sorted([f for f in os.listdir(images_dir) if f.startswith("MASK_layer") and f.endswith(".png")])
        if not mask_files:
            # Si no hay MASK_layer*, toma cualquier PNG (comodín)
            mask_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])

        if not mask_files:
            QMessageBox.information(self, "Split components", f"No se han encontrado PNG para procesar en {images_dir}.")
            return

        total_components = 0
        moved_to_pre = 0
        moved_to_post = 0

        for fname in mask_files:
            src_path = os.path.join(images_dir, fname)
            # Lee la imagen
            img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # Si no se puede leer, intenta mover al post para no perderlo
                try:
                    dst = os.path.join(post_dir, fname)
                    if os.path.abspath(src_path) != os.path.abspath(dst):
                        shutil.move(src_path, dst)
                        moved_to_post += 1
                except Exception:
                    pass
                continue

            # Asegura binario
            _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            num_labels, labels = cv2.connectedComponents(bw, connectivity=8)

            if num_labels > 2:  # más de 1 componente real (num_labels incluye el fondo)
                # mover original a pre/
                try:
                    dst_pre = os.path.join(pre_dir, fname)
                    if os.path.abspath(src_path) != os.path.abspath(dst_pre):
                        shutil.move(src_path, dst_pre)
                    else:
                        # ya estaba en la carpeta base como pre (no suele ocurrir)
                        pass
                    moved_to_pre += 1
                except Exception:
                    # si falla mover, intentamos continuar sin mover
                    pass

                # guardar cada componente en post/
                for comp_id in range(1, num_labels):
                    comp_mask = (labels == comp_id).astype('uint8') * 255
                    out_name = fname.replace(".png", f"_comp{comp_id}.png")
                    out_path = os.path.join(post_dir, out_name)
                    cv2.imwrite(out_path, comp_mask)
                    total_components += 1
            else:
                # 0 o 1 componente -> guardar original directamente en post/
                try:
                    dst_post = os.path.join(post_dir, fname)
                    if os.path.abspath(src_path) != os.path.abspath(dst_post):
                        shutil.move(src_path, dst_post)
                    moved_to_post += 1
                except Exception:
                    # si falla mover, intenta copiar
                    try:
                        dst_post = os.path.join(post_dir, fname)
                        cv2.imwrite(dst_post, img)
                        moved_to_post += 1
                    except Exception:
                        pass

        QMessageBox.information(
            self,
            "Split components",
            f"Proceso completado.\nComponentes guardados en: {post_dir}\nOriginales movidos a pre (los que se separaron): {moved_to_pre}\nOriginales movidos directamente a post: {moved_to_post}\nTotal componentes individuales generados: {total_components}"
        )

class MaterialSelectDialog(QDialog):
    def __init__(self, materiales_dir, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Seleccionar material")
        self.selected_file = None

        layout = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(1)
        self.table.setHorizontalHeaderLabels(["Nombre de material"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        # Lee todos los archivos .material en la carpeta
        material_files = [f for f in os.listdir(materiales_dir) if f.endswith(".material")]
        self.table.setRowCount(len(material_files))
        for i, fname in enumerate(material_files):
            name = fname[:-9]  # Elimina ".material"
            self.table.setItem(i, 0, QTableWidgetItem(name))

        layout.addWidget(self.table)

        select_btn = QPushButton("Seleccionar")
        select_btn.clicked.connect(self.select_material)
        layout.addWidget(select_btn)

        self.setLayout(layout)

    def select_material(self):
        selected = self.table.currentRow()
        if selected >= 0:
            name = self.table.item(selected, 0).text()
            self.selected_file = name + ".material"
            self.accept()
        else:
            QMessageBox.warning(self, "Error", "Selecciona un material de la lista.")

if __name__ == "__main__":
    materials = [
        Material("Material_1", 10, 5, "Rojo"),
        Material("Material_2", 12, 6, "Verde"),
        Material("Material_3", 8, 4, "Azul"),
        Material("Material_4", 15, 7, "Amarillo"),
        Material("Material_5", 9, 3, "Blanco"),
    ]
    slicer_settings = SlicerSettings(materials=materials)
    app = QApplication([])
    window = MainWindow(slicer_settings)
    window.resize(1200, 800)
    window.show()

    app.exec_()
