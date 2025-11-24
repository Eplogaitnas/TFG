import json
import math
import os
import zipfile
import OpenGL
import glm
import numpy as np
from PyQt5.QtGui import QImage
import zlib
import cv2
import zipfile
import shutil
import numpy as np

from OpenGL.GL.framebufferobjects import glBindRenderbuffer
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QOpenGLFramebufferObject, QSurfaceFormat, QOffscreenSurface, QOpenGLContext, QOpenGLDebugLogger, \
    QOpenGLWindow

from gl_utils import load_shader, GLModel, get_models_bounding_box, get_models_bounding_box_transformed

OpenGL.ERROR_ON_COPY = True
OpenGL.STORE_POINTERS = False
from OpenGL.GL.shaders import *
from OpenGL.GL import *
from stl import Mesh


class SlicerSettings:
    def __init__(self, model_transforms=None, materials=None, layer_thickness=0.1, first_layer_thickness=0.1,
                 image_size=500, physical_size=15.0):
        # model_transforms: lista de transformaciones (posición, orientación, escala) para cada modelo/material.
        # Si no se proporciona, se inicializa con una transformación por defecto.
        if model_transforms is None:
            model_transforms = [ModelTransform()]
        if materials is None:
            materials = [Material("Material_1")]
        self.model_transforms = model_transforms
        self.materials = materials

        # _layer_thickness: grosor de cada capa (mm)
        self._layer_thickness = layer_thickness

        # _first_layer_thickness: grosor de la primera capa (mm)
        self._first_layer_thickness = first_layer_thickness

        # image_size: tamaño (en píxeles) de las imágenes generadas para cada capa
        self.image_size = image_size

        # physical_size: tamaño físico (en mm) del área de impresión/proyección
        self.physical_size = physical_size
        # si True, dividir cada máscara en componentes conectados (requiere OpenCV)
        self.split_components = False
    #Se define las propiedades para acceder y modificar los atributos de la clase.
    #(Se utilizará para que el usuario pueda modificar los valores de los atributos de la clase)
    @property
    def layer_thickness(self):
        return self._layer_thickness

    @layer_thickness.setter
    def layer_thickness(self, value):
        self._layer_thickness = value

    @property
    def first_layer_thickness(self):
        return self._first_layer_thickness

    @first_layer_thickness.setter
    def first_layer_thickness(self, value):
        self._first_layer_thickness=value

    def set_materials(self, materials):
        """
        Permite actualizar la lista de materiales.
        :param materials: Lista de objetos Material.
        """
        self.materials = materials

class ModelTransform:
    def __init__(self, position=None, orientation=None, scale=None):
        # Si no se proporciona orientación, se inicializa a [0.0, 0.0, 0.0] (sin rotación)
        if orientation is None:
            orientation = [0.0, 0.0, 0.0]
        # Si no se proporciona posición, se inicializa a [0.0, 0.0, 0.0] (origen)
        if position is None:
            position = [0.0, 0.0, 0.0]
        # Si no se proporciona escala, se inicializa a [1.0, 1.0, 1.0] (sin escalado)
        if scale is None:
            scale = [1.0, 1.0, 1.0]
        # Guarda la posición del modelo (x, y, z)
        self.position = position
        # Guarda la orientación del modelo (rotación en grados o radianes para x, y, z)
        self.orientation = orientation
        # Guarda la escala del modelo (factor de escala para x, y, z)
        self.scale = scale

class Material:
    def __init__(self, nombre, tiempo_exposicion=0, tiempo_cura=0, color=None):
        self.nombre = nombre
        self.tiempo_exposicion = tiempo_exposicion
        self.tiempo_cura = tiempo_cura
        self.color = color  # Opcional, para visualización

class Slicer(QObject):
    # Es una subclase de QObject, para poder asociarla a un hilo de Qt y enviar señales a la interfaz gráfica
    finished = pyqtSignal()  # Señal que indica que se completó la generación de imágenes
    progress = pyqtSignal(int)  # Señal que indica el progreso de la generación de imágenes (porcentaje)
    request_render = pyqtSignal(int)  # Señal para solicitar el renderizado de una capa específica

    def __init__(self, model_filenames, images_metadata, settings):
        super().__init__()
        # Lista de rutas de archivos STL a procesar
        self.model_filenames = model_filenames
        # Directorio donde se guardarán las imágenes generadas
        self.images_dir = os.path.dirname(images_metadata)
        # Nombre base para los archivos de imagen y metadatos
        self.images_name = os.path.splitext(os.path.basename(images_metadata))[0]
        # Configuración del slicer (instancia de SlicerSettings)
        self.settings: SlicerSettings = settings
        # Logger para depuración de OpenGL
        self.logger = QOpenGLDebugLogger(self)
        # Bandera para detener el proceso de slicing si es necesario
        self.stop_requested = False

    def save_metadata(self, filename, num_layers, num_materials=1):
        metadata = {
            "layer_thickness": self.settings.layer_thickness,
            "first_layer_thickness": self.settings.first_layer_thickness,
            "physical_size": self.settings.physical_size,
            "num_layers": num_layers,
            "num_materials": num_materials
        }
        with open(filename, 'w') as fp:
            json.dump(metadata, fp)


    def run(self):
        # Inicializa el renderizado OpenGL fuera de pantalla
        surface = QOffscreenSurface()
        format = QSurfaceFormat()
        format.setRenderableType(QSurfaceFormat.OpenGL)
        format.setVersion(3, 3)
        surface.setFormat(format)
        surface.create()
        # Crea un contexto OpenGL para renderizar en segundo plano
        ctx = QOpenGLContext()
        ctx.setFormat(format)
        ctx.create()
        ctx.makeCurrent(surface)
        # Inicializa el logger de OpenGL
        self.logger.initialize()
        self.logger.messageLogged.connect(lambda m: print("OpenGL: " + m.message()))
        self.logger.startLogging()
        # Define el tamaño de las imágenes y el área física de proyección
        w = h = self.settings.image_size
        physical_size = self.settings.physical_size  # Diámetro del área de proyección en mm

        # Crea un Framebuffer Object (FBO) para renderizar sin mostrar en pantalla
        fb = QOpenGLFramebufferObject(w, h)
        fb.setAttachment(QOpenGLFramebufferObject.Depth)
        fb.bind()

        # Crea un renderbuffer con un stencil buffer para operaciones de recorte
        rb = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, rb)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_STENCIL, w, h)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rb)

        glViewport(0, 0, w, h)

        # Carga los shaders necesarios para el renderizado
        shader_program = load_shader("shaderTFG.vert", "shaderSlicer.frag")

        # Configura parámetros de OpenGL
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClearDepthf(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)

        glUseProgram(shader_program)
        model_loc = glGetUniformLocation(shader_program, "model")
        view_loc = glGetUniformLocation(shader_program, "view")
        projection_loc = glGetUniformLocation(shader_program, "projection")

        # Carga los modelos STL en una lista (uno por material)
        models = [GLModel(filename) for filename in self.model_filenames]

        # Calcula el bounding box global y las transformaciones para todos los modelos
        global_p1, global_p2 = get_models_bounding_box(models)
        orientations = [tr.orientation for tr in self.settings.model_transforms]
        scales = [tr.scale for tr in self.settings.model_transforms]
        global_p1_t, global_p2_t = get_models_bounding_box_transformed(models, orientations, scales)

        # Calcula la altura total del modelo y el centro
        model_height = global_p2_t[2] - global_p1_t[2]
        center = (global_p2[0] + global_p1[0]) / 2, (global_p2[1] + global_p1[1]) / 2, (global_p2[2] + global_p1[2]) / 2

        # Calcula el número de capas a generar
        num_layers = math.ceil(
            max(model_height - self.settings.first_layer_thickness, 0) / self.settings.layer_thickness + 1)

        # Guarda los metadatos del slicing
        self.save_metadata(os.path.join(self.images_dir, self.images_name + ".json"), num_layers, len(models))

        # --- Nuevo flujo: por cada capa rasterizar todas las máscaras, detectar orden y guardar slices separados ---
        # prev_img por material (para XOR por material)
        prev_imgs = {mi: None for mi in range(len(models))}
        # añadimos secuencia por capa en metadatos
        layer_sequences = []
        # componentes por material en cada capa
        layer_components = []

        for i in range(num_layers):
            if self.stop_requested:
                break

            # Calcula la posición Z de la capa actual
            if i == 0:
                layer_offset = 0
            else:
                layer_offset = self.settings.first_layer_thickness + (i - 1) * self.settings.layer_thickness
            layer_z = -model_height / 2 + layer_offset

            # rasteriza máscara para cada modelo en esta capa
            masks = []
            for material_index, model in enumerate(models):
                # calcula transform y set uniforms
                model_transform = model.get_model_matrix_slicer(
                    self.settings.model_transforms[material_index].position,
                    self.settings.model_transforms[material_index].orientation,
                    self.settings.model_transforms[material_index].scale,
                    center
                )
                glUseProgram(shader_program)
                glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model_transform))

                view = glm.lookAt(glm.vec3(0, 0, layer_z + 0.0001),
                                  glm.vec3(0, 0, float(-model_height)),
                                  glm.vec3(0, 1, 0))
                projection = glm.ortho(-physical_size / 2 * w / h, physical_size / 2 * w / h,
                                       -physical_size / 2, physical_size / 2,
                                       0, model_height + 1)
                glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
                glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm.value_ptr(projection))

                glDisable(GL_DEPTH_TEST)
                glEnable(GL_STENCIL_TEST)
                glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)

                # stencil render to obtain interior mask
                glStencilFunc(GL_ALWAYS, 0, 0xFF)
                glStencilOpSeparate(GL_BACK, GL_KEEP, GL_KEEP, GL_INCR)
                glStencilOpSeparate(GL_FRONT, GL_KEEP, GL_KEEP, GL_KEEP)
                model.paint()

                view_front = glm.lookAt(glm.vec3(0, 0, layer_z - 0.0001),
                                        glm.vec3(0, 0, float(-model_height)),
                                        glm.vec3(0, 1, 0))
                glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view_front))
                glStencilOpSeparate(GL_BACK, GL_KEEP, GL_KEEP, GL_KEEP)
                glStencilOpSeparate(GL_FRONT, GL_KEEP, GL_KEEP, GL_DECR)
                model.paint()

                # paint only where stencil != 0
                glClear(GL_COLOR_BUFFER_BIT)
                glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
                glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP)
                glStencilFunc(GL_NOTEQUAL, 0, 0xFF)
                model.paint()

                # read FBO -> numpy grayscale mask
                image = fb.toImage()
                image_gray = image.convertToFormat(QImage.Format_Grayscale8)
                ptr = image_gray.bits()
                ptr.setsize(h * w)
                img_array = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w))
                mask = img_array > 0
                masks.append(mask)

                # clear for next model
                glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)

            # detect print order for this layer
            try:
                if any(m.any() for m in masks):
                    order = compute_print_order_from_masks(masks)
                else:
                    order = list(range(len(models)))
            except Exception:
                order = list(range(len(models)))

            # save detected sequence for metadata
            layer_sequences.append(order)
            # componentes por material en esta capa
            comps_for_layer = {}

            # save per-material masks in the detected order and compute XOR per material
            seq_position = 0
            for mat_idx in order:
                seq_position += 1
                mask = masks[mat_idx]
                mask_u8 = (mask.astype(np.uint8) * 255)

                if getattr(self.settings, "split_components", False):
                    # connected components (OpenCV)
                    num_labels, labels = cv2.connectedComponents(mask_u8, connectivity=8)
                    num_components = max(0, num_labels - 1)
                    comps_for_layer[mat_idx] = num_components
                    # save each component separately
                    if num_components > 0:
                        for comp_id in range(1, num_labels):
                            comp_mask = (labels == comp_id).astype(np.uint8) * 255
                            qimg_comp = QImage(comp_mask.data, w, h, w, QImage.Format_Grayscale8)
                            comp_name = f"MASK_layer{ i }_seq{ seq_position }_mat{ mat_idx }_comp{ comp_id }.png"
                            qimg_comp.save(os.path.join(self.images_dir, comp_name))
                    else:
                        qimg_mask = QImage(mask_u8.data, w, h, w, QImage.Format_Grayscale8)
                        qimg_mask.save(os.path.join(self.images_dir, f"MASK_layer{ i }_seq{ seq_position }_mat{ mat_idx }.png"))
                else:
                    # guarda la máscara completa
                    qimg_mask = QImage(mask_u8.data, w, h, w, QImage.Format_Grayscale8)
                    qimg_mask.save(os.path.join(self.images_dir, f"MASK_layer{ i }_seq{ seq_position }_mat{ mat_idx }.png"))
                    comps_for_layer[mat_idx] = 0

                # XOR per material: usar la máscara completa (no por componente)
                prev = prev_imgs.get(mat_idx)
                if prev is None:
                    xor_base = np.zeros_like(mask_u8, dtype=np.uint8)
                else:
                    xor_base = prev
                xor_result = np.bitwise_xor(mask_u8, xor_base)
                xor_qimage = QImage(xor_result.data, w, h, w, QImage.Format_Grayscale8)
                xor_name = f"XOR_layer{ i }_mat{ mat_idx }.png"
                xor_qimage.save(os.path.join(self.images_dir, xor_name))

                # RLE + VarInt + Deflate per material (same pipeline)
                rle_data = rle_encode(xor_result)
                raw_bytes = bytearray()
                for value, length in rle_data:
                    raw_bytes += varint_encode(value)
                    raw_bytes += varint_encode(length)
                deflated_bytes = zlib.compress(bytes(raw_bytes))
                deflate_path = os.path.join(self.images_dir, f"DEFLATE_layer{ i }_mat{ mat_idx }.deflate")
                with open(deflate_path, "wb") as f:
                    f.write(deflated_bytes)

                # update prev for next layer for this material
                prev_imgs[mat_idx] = mask_u8.copy()

            layer_components.append(comps_for_layer)

            # progreso
            self.progress.emit(round((i + 1) / num_layers * 100))
            self.request_render.emit(i)


        # If split_components is enabled, run split first, compress post/ with deflate into a single zip,
        # then remove intermediate PNGs so only .json and the .zip remain.
        if getattr(self.settings, "split_components", False):
            try:
                split_result = _split_masks_pre_post(self.images_dir)
                zip_path = os.path.join(self.images_dir, self.images_name + ".zip")
                zip_result = compress_post_masks_to_zip(self.images_dir, zip_path)

                # Cleanup PNGs and post/pre dirs to leave only .json and the .zip
                try:
                    pre_dir = os.path.join(self.images_dir, "pre")
                    post_dir = os.path.join(self.images_dir, "post")
                    # remove any remaining pngs in root images_dir
                    for f in os.listdir(self.images_dir):
                        if f.lower().endswith(".png"):
                            try:
                                os.remove(os.path.join(self.images_dir, f))
                            except Exception:
                                pass
                    if os.path.isdir(post_dir):
                        shutil.rmtree(post_dir, ignore_errors=True)
                    if os.path.isdir(pre_dir):
                        shutil.rmtree(pre_dir, ignore_errors=True)
                except Exception:
                    pass

                # store split/zip info in metadata JSON
                try:
                    meta_path = os.path.join(self.images_dir, self.images_name + ".json")
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        meta = json.load(mf)
                    meta["split_result"] = split_result
                    meta["zip_result"] = zip_result
                    with open(meta_path, "w", encoding="utf-8") as mf:
                        json.dump(meta, mf, indent=4)
                except Exception:
                    pass
            except Exception as e:
                try:
                    self.logger.warning(f"split/compress failed: {e}")
                except Exception:
                    pass

        # No additional image archives: keep only metadata.zip (existing deflate pipeline) if you still use it,
        # otherwise skip additional compression. Here we skip writing other archives to ensure outputs are
        # only the .json and the .zip produced above.

        # compress deflate files into zip (optional) - keep only metadata.zip as output
        compress_deflate_files_to_zip(self.images_dir, os.path.join(self.images_dir, "metadata.zip"))

        # Emite la señal de finalización
        self.finished.emit()

        # Libera los recursos de OpenGL
        for model in models:
            model.destroy()
        glDeleteProgram(shader_program)
        fb.release()
        surface.destroy()
        self.logger.stopLogging()

    def stop(self):
        # Permite detener el proceso
        self.stop_requested = True

class SlicerNormal(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, model_filenames, images_metadata, settings):
        super().__init__()
        self.model_filenames = model_filenames
        self.images_dir = os.path.dirname(images_metadata)
        self.images_name = os.path.splitext(os.path.basename(images_metadata))[0]
        self.settings: SlicerSettings = settings
        self.logger = QOpenGLDebugLogger(self)
        self.stop_requested = False

    def save_metadata(self, filename, num_layers, num_materials=1):
        metadata = {
            "layer_thickness": self.settings.layer_thickness,
            "first_layer_thickness": self.settings.first_layer_thickness,
            "physical_size": self.settings.physical_size,
            "num_layers": num_layers,
            "num_materials": num_materials
        }
        with open(filename, 'w') as fp:
            json.dump(metadata, fp)

    def run(self):
        surface = QOffscreenSurface()
        format = QSurfaceFormat()
        format.setRenderableType(QSurfaceFormat.OpenGL)
        format.setVersion(3, 3)
        surface.setFormat(format)
        surface.create()
        ctx = QOpenGLContext()
        ctx.setFormat(format)
        ctx.create()
        ctx.makeCurrent(surface)
        self.logger.initialize()
        self.logger.messageLogged.connect(lambda m: print("OpenGL: " + m.message()))
        self.logger.startLogging()
        w = h = self.settings.image_size
        physical_size = self.settings.physical_size

        fb = QOpenGLFramebufferObject(w, h)
        fb.setAttachment(QOpenGLFramebufferObject.Depth)
        fb.bind()

        rb = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, rb)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_STENCIL, w, h)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rb)

        glViewport(0, 0, w, h)

        shader_program = load_shader("shaderTFG.vert", "shaderSlicer.frag")

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClearDepthf(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)

        glUseProgram(shader_program)
        model_loc = glGetUniformLocation(shader_program, "model")
        view_loc = glGetUniformLocation(shader_program, "view")
        projection_loc = glGetUniformLocation(shader_program, "projection")

        models = [GLModel(filename) for filename in self.model_filenames]

        global_p1, global_p2 = get_models_bounding_box(models)
        orientations = [tr.orientation for tr in self.settings.model_transforms]
        scales = [tr.scale for tr in self.settings.model_transforms]
        global_p1_t, global_p2_t = get_models_bounding_box_transformed(models, orientations, scales)

        model_height = global_p2_t[2] - global_p1_t[2]
        center = (global_p2[0] + global_p1[0]) / 2, (global_p2[1] + global_p1[1]) / 2, (global_p2[2] + global_p1[2]) / 2

        num_layers = math.ceil(
            max(model_height - self.settings.first_layer_thickness, 0) / self.settings.layer_thickness + 1)

        self.save_metadata(os.path.join(self.images_dir, self.images_name + ".json"), num_layers, len(models))

        for material_index, model in enumerate(models):
            model_transform = model.get_model_matrix_slicer(
                self.settings.model_transforms[material_index].position,
                self.settings.model_transforms[material_index].orientation,
                self.settings.model_transforms[material_index].scale,
                center
            )
            view = glm.lookAt([0, 0, 1], [0, 0, 0], [0, 1, 0])
            projection = glm.ortho(-physical_size / 2 * w / h, physical_size / 2 * w / h,
                                   -physical_size / 2, physical_size / 2,
                                   0, model_height + 1)
            glUniformMatrix4fv(model_loc, 1, GL_FALSE, glm.value_ptr(model_transform))
            glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
            glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm.value_ptr(projection))

            for i in range(num_layers):
                if self.stop_requested:
                    break
                glUseProgram(shader_program)
                if i == 0:
                    layer_offset = 0
                else:
                    layer_offset = self.settings.first_layer_thickness + (i - 1) * self.settings.layer_thickness
                layer_z = -model_height / 2 + layer_offset

                glDisable(GL_DEPTH_TEST)
                glEnable(GL_STENCIL_TEST)

                glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)

                view = glm.lookAt(glm.vec3(0, 0, layer_z + 0.0001),
                                  glm.vec3(0, 0, float(-model_height)),
                                  glm.vec3(0, 1, 0))
                glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
                glStencilFunc(GL_ALWAYS, 0, 0xFF)
                glStencilOpSeparate(GL_BACK, GL_KEEP, GL_KEEP, GL_INCR)
                glStencilOpSeparate(GL_FRONT, GL_KEEP, GL_KEEP, GL_KEEP)
                model.paint()

                view = glm.lookAt(glm.vec3(0, 0, layer_z - 0.0001),
                                  glm.vec3(0, 0, float(-model_height)),
                                  glm.vec3(0, 1, 0))
                glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
                glStencilOpSeparate(GL_BACK, GL_KEEP, GL_KEEP, GL_KEEP)
                glStencilOpSeparate(GL_FRONT, GL_KEEP, GL_KEEP, GL_DECR)
                model.paint()

                glClear(GL_COLOR_BUFFER_BIT)

                view = glm.lookAt(glm.vec3(0, 0, layer_z + 0.0001),
                                  glm.vec3(0, 0, float(-model_height)),
                                  glm.vec3(0, 1, 0))
                glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm.value_ptr(view))
                glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP)
                glStencilFunc(GL_NOTEQUAL, 0, 0xFF)
                model.paint()

                image = fb.toImage()
                material_suffix = f"_mat{material_index}" if len(models) > 1 else ""
                image.save(os.path.join(self.images_dir, self.images_name + "_" + str(i) + material_suffix + ".png"))
                self.progress.emit(round((i + 1) / num_layers * 100))

        self.finished.emit()
        for model in models:
            model.destroy()
        glDeleteProgram(shader_program)
        fb.release()
        surface.destroy()
        self.logger.stopLogging()

    def stop(self):
        self.stop_requested = True


def rle_encode(arr):
    # Codifica una imagen 2D en una lista de (valor, longitud)
    flat = arr.flatten()
    result = []
    prev = flat[0]
    count = 1
    for pixel in flat[1:]:
        if pixel == prev:
            count += 1
        else:
            result.append((int(prev), count))
            prev = pixel
            count = 1
    result.append((int(prev), count))
    return result

def varint_encode(number):
    """Codifica un entero en formato VarInt (longitud variable, estilo protobuf)."""
    bytes_out = []
    while True:
        to_write = number & 0x7F
        number >>= 7
        if number:
            bytes_out.append(to_write | 0x80)
        else:
            bytes_out.append(to_write)
            break
    return bytes(bytes_out)

def compress_deflate_files_to_zip(input_folder, output_zip):
    """
    Comprime todos los archivos .deflate de una carpeta en un único archivo .zip
    
    Parámetros:
    - input_folder: ruta a la carpeta donde están los .deflate
    - output_zip: ruta/nombre del archivo .zip de salida
    """
    with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(input_folder):
            for file in files:
                if file.endswith(".deflate"):
                    filepath = os.path.join(root, file)
                    arcname = os.path.relpath(filepath, input_folder)  # mantiene estructura relativa
                    zipf.write(filepath, arcname)
                    os.remove(filepath)  # elimina el archivo original

def compress_post_masks_to_zip(images_dir, out_zip):
    """
    Lee PNGs en images_dir/post, aplica RLE+VarInt y comprime con zlib (deflate).
    Guarda cada entrada comprimida dentro de out_zip con sufijo .deflate (no deja .deflate sueltos en disco).
    Devuelve dict con estadísticas.
    """
    post_dir = os.path.join(images_dir, "post")
    if not os.path.isdir(post_dir):
        return {"zipped": 0}

    png_files = sorted([f for f in os.listdir(post_dir) if f.lower().endswith(".png")])
    if not png_files:
        return {"zipped": 0}

    zipped = 0
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in png_files:
            full = os.path.join(post_dir, fname)
            img = cv2.imread(full, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # for safety ensure binary mask
            _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            # flatten (row-major) and RLE encode
            flat = np.ascontiguousarray(bw.reshape(-1))
            # rle_encode espera una secuencia de bytes/píxeles; si tu implementación espera 2D adapta aquí
            rle_data = rle_encode(flat)
            raw_bytes = bytearray()
            for value, length in rle_data:
                raw_bytes += varint_encode(int(value))
                raw_bytes += varint_encode(int(length))
            deflated = zlib.compress(bytes(raw_bytes))
            arcname = fname.replace(".png", ".deflate")
            zf.writestr(arcname, deflated)
            zipped += 1

    return {"zipped": zipped, "files": png_files}

import numpy as np
from collections import deque

def compute_print_order_from_masks(masks):
    """
    masks: list of boolean 2D numpy arrays, one per material for the current layer
    devuelve: lista de índices de materiales en el orden de impresión (inner -> outer preferido)
    """
    n = len(masks)
    # calcula centroides y areas
    areas = []
    centroids = []
    h, w = masks[0].shape
    for m in masks:
        coords = np.argwhere(m)
        area = coords.shape[0]
        areas.append(area)
        if area == 0:
            centroids.append(None)
        else:
            cy, cx = coords.mean(axis=0)
            centroids.append((int(round(cy)), int(round(cx))))

    # construye aristas: if mask_i contains centroid of mask_j => j before i (edge j->i)
    edges = [[] for _ in range(n)]
    indeg = [0] * n
    for i in range(n):
        if areas[i] == 0:
            continue
        for j in range(n):
            if i == j or areas[j] == 0:
                continue
            c = centroids[j]
            if c is None:
                continue
            cy, cx = c
            # comprueba límites
            if 0 <= cy < h and 0 <= cx < w:
                if masks[i][cy, cx]:
                    # j must come before i
                    edges[j].append(i)
                    indeg[i] += 1

    # intento de topological sort (Kahn)
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in edges[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(order) == n:
        # topological OK -> mantenemos sólo materiales con area>0 y devolvemos en ese orden
        return [i for i in order if areas[i] > 0]
    else:
        # hay ciclos: fallback determinista por área ascendente (pequeñas primero)
        idxs = [i for i in range(n) if areas[i] > 0]
        idxs.sort(key=lambda k: areas[k])
        return idxs

def compose_layer_from_masks(masks, materials=None, color_map_rgb=None):
    """
    Dada una lista de máscaras booleanas (una por material) devuelve una imagen RGB (numpy uint8)
    compuesta en un orden seguro (compute_print_order_from_masks). materials es la lista de objetos
    Material (opcional) para obtener el nombre/color.
    """
    if not masks:
        return None
    h, w = masks[0].shape
    print_order = compute_print_order_from_masks(masks)

    assigned = np.zeros((h, w), dtype=bool)
    result_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    if color_map_rgb is None:
        color_map_rgb = {
            "Rojo": (255, 0, 0),
            "Verde": (0, 255, 0),
            "Azul": (0, 0, 255),
            "Amarillo": (255, 255, 0),
            "Blanco": (255, 255, 255)
        }

    for mat_idx in print_order:
        mask = masks[mat_idx]
        new_pixels = mask & (~assigned)
        if not new_pixels.any():
            continue
        if materials is not None and 0 <= mat_idx < len(materials):
            mat_name = getattr(materials[mat_idx], "color", "Rojo")
        else:
            mat_name = "Rojo"
        rgb = color_map_rgb.get(mat_name, (255, 0, 0))
        # asigna color a los píxeles nuevos
        result_rgb[new_pixels] = rgb
        # marca esos píxeles como asignados
        assigned |= new_pixels

    return result_rgb

def _split_masks_pre_post(images_dir):
    """
    Move original MASK_*.png to 'pre' if they split into >1 component, otherwise move them to 'post'.
    Save component masks into 'post' as *_compN.png.
    Returns a dict with statistics.
    """
    if not os.path.isdir(images_dir):
        return {"error": "no images dir", "processed": 0}

    pre_dir = os.path.join(images_dir, "pre")
    post_dir = os.path.join(images_dir, "post")
    os.makedirs(pre_dir, exist_ok=True)
    os.makedirs(post_dir, exist_ok=True)

    # candidate mask files
    mask_files = sorted([f for f in os.listdir(images_dir) if f.startswith("MASK_layer") and f.endswith(".png")])
    if not mask_files:
        mask_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])

    total_components = 0
    moved_to_pre = 0
    moved_to_post = 0

    for fname in mask_files:
        src_path = os.path.join(images_dir, fname)
        img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            try:
                dst = os.path.join(post_dir, fname)
                if os.path.abspath(src_path) != os.path.abspath(dst):
                    shutil.move(src_path, dst)
                    moved_to_post += 1
            except Exception:
                pass
            continue

        _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        num_labels, labels = cv2.connectedComponents(bw, connectivity=8)

        if num_labels > 2:
            try:
                dst_pre = os.path.join(pre_dir, fname)
                if os.path.abspath(src_path) != os.path.abspath(dst_pre):
                    shutil.move(src_path, dst_pre)
                moved_to_pre += 1
            except Exception:
                pass
            for comp_id in range(1, num_labels):
                comp_mask = (labels == comp_id).astype('uint8') * 255
                out_name = fname.replace(".png", f"_comp{comp_id}.png")
                out_path = os.path.join(post_dir, out_name)
                cv2.imwrite(out_path, comp_mask)
                total_components += 1
        else:
            try:
                dst_post = os.path.join(post_dir, fname)
                if os.path.abspath(src_path) != os.path.abspath(dst_post):
                    shutil.move(src_path, dst_post)
                moved_to_post += 1
            except Exception:
                try:
                    dst_post = os.path.join(post_dir, fname)
                    cv2.imwrite(dst_post, img)
                    moved_to_post += 1
                except Exception:
                    pass

    return {
        "total_components": total_components,
        "moved_to_pre": moved_to_pre,
        "moved_to_post": moved_to_post,
        "processed": len(mask_files)
    }

def zip_post_images(images_dir, out_zip):
    """
    Create a zip with PNG files inside images_dir/post (if exists) else images_dir.
    """
    post_dir = os.path.join(images_dir, "post")
    source_dir = post_dir if os.path.isdir(post_dir) else images_dir
    png_files = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(".png"):
                full = os.path.join(root, f)
                arc = os.path.relpath(full, source_dir)
                png_files.append((full, arc))
    if not png_files:
        return {"zipped": 0}
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for full, arc in png_files:
            zf.write(full, arc)
    return {"zipped": len(png_files)}

