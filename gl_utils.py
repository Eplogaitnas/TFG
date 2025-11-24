"""Funciones auxiliares para el renderizado con OpenGL"""

import ctypes

import glm
import numpy
import OpenGL
import stl

OpenGL.ERROR_ON_COPY = True
OpenGL.STORE_POINTERS = False
from OpenGL.GL.shaders import *
from OpenGL.GL import *
from stl import Mesh

"""
Función auxiliar para cargar los shaders.
Parámetros:
    vertex_filename: Nombre del fichero que contiene el vertex shader a cargar.
    fragment_filename: Nombre del fichero que contiene el fragment shader a cargar.
Devuelve el handle OpenGL del shader compilado y enlazado. 
"""


def load_shader(vertex_filename, fragment_filename):
    vertex_shader = compileShader(open(vertex_filename, "r").read(), GL_VERTEX_SHADER)
    fragment_shader = compileShader(open(fragment_filename, "r").read(), GL_FRAGMENT_SHADER)
    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return shader_program


"""
Función auxiliar para cargar un compute shader.
Parámetros:
    filename: Nombre del fichero que contiene el compute shader a cargar.
Devuelve el handle OpenGL del shader compilado y enlazado. 
"""


def load_compute_shader(filename):
    shader = compileShader(open(filename, "r").read(), GL_COMPUTE_SHADER)
    shader_program = glCreateProgram()
    glAttachShader(shader_program, shader)
    glLinkProgram(shader_program)
    glDeleteShader(shader)
    return shader_program


"""
Función auxiliar para crear un VAO a partir de una malla de numpy-stl.
Parámetros:
    mesh (stl.Mesh): Malla que se cargará en el VAO.
Devuelve:
    vao: Handle OpenGL del VAO creado.
    num_vertices: Número de vértices que contiene el VAO.  
"""


def vao_from_mesh(mesh: stl.Mesh):
    # mesh.points es un array de triángulos, cada triángulo es un array de 3 vectores de 3 elementos cada uno.
    num_vertices = len(mesh.points) * 3
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, len(mesh.points) * 4 * 9, mesh.points.flatten(), GL_STATIC_DRAW)

    vertex_offset = 0
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(vertex_offset))
    glEnableVertexAttribArray(0)

    glBindVertexArray(0)
    glDeleteBuffers(1, [vbo])
    return vao, num_vertices


"""
Función auxiliar para crear un VAO que contiene un cuadrado formado por dos triángulos
Devuelve:
    vao: Handle OpenGL del VAO creado.
"""


def create_quad_vao():
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    quad = numpy.asarray([
        0., 1., 0.,
        1., 1., 0.,
        1., 0., 0.,
        0., 1., 0.,
        1., 0., 0.,
        0., 0., 0.
    ], dtype=numpy.single)
    glBufferData(GL_ARRAY_BUFFER, len(quad) * 36, quad, GL_STATIC_DRAW)

    vertex_offset = 0
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(vertex_offset))
    glEnableVertexAttribArray(0)

    glBindVertexArray(0)
    glDeleteBuffers(1, [vbo])

    return vao


"""Clase auxiliar para cargar y renderizar un modelo 3D"""


class GLModel:
    def __init__(self, path):
        self.num_vertices = None
        self.scale = None
        self.center = None
        self.path = path
        self.vao = None
        self.mesh: Mesh = None

    def create_vao(self):
        self.mesh = Mesh.from_file(self.path)
        volume, cog, inertia = self.mesh.get_mass_properties()
        self.center = (self.mesh.max_ + self.mesh.min_) / 2
        # Centrar el modelo en su centro de gravedad y escalarlo de forma que su dimensión más larga sea igual a 1
        self.scale = 1 / max(self.mesh.max_[0] - self.mesh.min_[0],
                             self.mesh.max_[1] - self.mesh.min_[1],
                             self.mesh.max_[2] - self.mesh.min_[2])
        # Cargar el modelo en un VAO
        self.vao, self.num_vertices = vao_from_mesh(self.mesh)

    def destroy(self):
        if self.vao is not None:
            glDeleteVertexArrays(1, [self.vao])

    def get_model_height(self, rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)):
        return self.get_model_size(rotation, scale)[2]

    def get_model_bounding_box_transformed(self, rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)):
        if self.vao is None:
            self.create_vao()
        transformed_mesh = Mesh(self.mesh.data.copy())
        transformed_mesh.transform(numpy.array(self.get_model_matrix_slicer(
            [0, 0, 0], rotation, [1, 1, 1], [0, 0, 0])))
        transformed_mesh.x *= scale[0]
        transformed_mesh.y *= scale[1]
        transformed_mesh.z *= scale[2]
        p1 = (transformed_mesh.x.min(), transformed_mesh.y.min(), transformed_mesh.z.min())
        p2 = (transformed_mesh.x.max(), transformed_mesh.y.max(), transformed_mesh.z.max())
        return p1, p2

    def get_model_bounding_box(self):
        if self.vao is None:
            self.create_vao()
        return self.mesh.min_, self.mesh.max_

    def get_model_size(self, rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)):
        p1, p2 = self.get_model_bounding_box_transformed(rotation, scale)
        size = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
        return size

    def get_model_matrix(self, translation, rotation, scale):
        if self.vao is None:
            # Cargar el modelo si no se hizo antes, para tener la escala y el centro de gravedad
            self.create_vao()
        # Rotar el modelo, escalarlo según el factor de escala proporcionado y las dimensiones
        # calculadas al cargarlo, y centrarlo
        model = glm.identity(glm.mat4)
        model = glm.scale(model, glm.vec3(self.scale, self.scale, self.scale) * glm.vec3(scale))
        model = glm.translate(model, glm.vec3(translation[0], translation[1], translation[2]))
        model = glm.rotate(model, glm.radians(rotation[0]), glm.vec3(1, 0, 0))
        model = glm.rotate(model, glm.radians(rotation[1]), glm.vec3(0, 1, 0))
        model = glm.rotate(model, glm.radians(rotation[2]), glm.vec3(0, 0, 1))
        model = glm.translate(model, glm.vec3(-self.center[0], -self.center[1], -self.center[2]))
        return model

    def get_model_matrix_slicer(self, translation, rotation, scale, center=None):
        # Calcula la matriz de transformación del modelo usada en el slicer. A diferencia de get_model_matrix,
        # mantiene las dimensiones originales del modelo en lugar de escalarlo para que tenga tamaño 1 (si el parámetro
        # scale es [1, 1, 1])
        if self.vao is None:
            # Cargar el modelo si no se hizo antes, para tener la escala y el centro de gravedad
            self.create_vao()
        # Rotar el modelo, escalarlo según el factor de escala proporcionado, y centrarlo
        model = glm.identity(glm.mat4)
        model = glm.scale(model, glm.vec3(scale))
        model = glm.translate(model, glm.vec3(translation[0], translation[1], translation[2]))
        model = glm.rotate(model, glm.radians(rotation[0]), glm.vec3(1, 0, 0))
        model = glm.rotate(model, glm.radians(rotation[1]), glm.vec3(0, 1, 0))
        model = glm.rotate(model, glm.radians(rotation[2]), glm.vec3(0, 0, 1))
        if center is None:
            center = self.center
        model = glm.translate(model, glm.vec3(-center[0], -center[1], -center[2]))
        return model

    def paint(self):
        if self.vao is None:
            self.create_vao()
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.num_vertices)


"""
Función auxiliar para calcular la envolvente de un conjunto de modelos
Parámetros:
    models (list(GLModel)): Modelos cuya envolvente se calculará
    rotation (vec3): Rotación aplicada a los modelos.
    scale (vec3): Escalado aplicado a los modelos.
Devuelve:
    global_p1: Esquina del paralelepípedo que contiene a los modelos con las coordenadas más bajas
    global_p2: Esquina del paralelepípedo que contiene a los modelos con las coordenadas más bajas
"""


def get_models_bounding_box_transformed(models, rotations, scales):
    global_p1, global_p2 = models[0].get_model_bounding_box_transformed(rotations[0], scales[0])
    for model, rotation, scale in zip(models[1:], rotations[1:], scales[1:]):
        p1, p2 = model.get_model_bounding_box_transformed(rotation, scale)
        global_p1 = (min(global_p1[0], p1[0]), min(global_p1[1], p1[1]), min(global_p1[2], p1[2]))
        global_p2 = (max(global_p2[0], p2[0]), max(global_p2[1], p2[1]), max(global_p2[2], p2[2]))

    return global_p1, global_p2


def get_models_bounding_box(models):
    global_p1, global_p2 = models[0].get_model_bounding_box()
    for model in models[1:]:
        p1, p2 = model.get_model_bounding_box()
        global_p1 = (min(global_p1[0], p1[0]), min(global_p1[1], p1[1]), min(global_p1[2], p1[2]))
        global_p2 = (max(global_p2[0], p2[0]), max(global_p2[1], p2[1]), max(global_p2[2], p2[2]))

    return global_p1, global_p2
