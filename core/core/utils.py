import face_recognition as fr
import numpy as np
from profiles.models import Profile


def is_ajax(request):
  return request.headers.get('x-requested-with') == 'XMLHttpRequest'


def get_encoded_faces():
    """
    Esta funcion carga todos las imagenes de perfil de los usuarios
    y codifica tus rostros 
    """
    # Obtener todos los perfiles de usuarios desde la base de datos 
    qs = Profile.objects.all()

    # Crear un diccionario para guardar la cara codificada de cada usuario
    encoded = {}

    for p in qs:
        # Iniciando la variable de codificacion como None
        encoding = None

        # Obteniendo la imagen de perfil del usuario
        face = fr.load_image_file(p.photo.path)

        # Codifica el rostro si lo detecta
        face_encodings = fr.face_encodings(face)
        if len(face_encodings) > 0:
            encoding = face_encodings[0]
        else:
            print("Rostro no encontrado en la imagen")

        # Agregar el rostro codificado al diccionario si encoding no es None
        if encoding is not None:
            encoded[p.user.username] = encoding

    # hace return del diccionario con los rostros codificados
    return encoded


def classify_face(img):
    """
    Esta funcion toma una imagen como input y regresa el nombre del rostro si existe uno en la imagen
    """
    # Cargando todas las caras conocidas y su codificaciones 
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    # Cargando la imagen de input
    img = fr.load_image_file(img)
 
    try:
        # Encontrando todos los rostros en la imagen input 
        face_locations = fr.face_locations(img)

        # Codificar los rostros en la imagen input
        unknown_face_encodings = fr.face_encodings(img, face_locations)

        # Identificar los rostros en la imagen input
        face_names = []
        for face_encoding in unknown_face_encodings:
            # Comparar la codificación de la cara actual con las codificaciones de todas las caras conocidas
            matches = fr.compare_faces(faces_encoded, face_encoding)

            # Encuentra la cara conocida con la codificación más cercana a la cara actual
            face_distances = fr.face_distance(faces_encoded, face_encoding)
            best_match_index = np.argmin(face_distances)

            # Si la cara conocida más cercana coincide con la cara actual, etiquete la cara con el nombre conocido.
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            else:
                name = "Desconocido"

            face_names.append(name)

        # Devolver el nombre de la primera cara de la imagen de entrada
        return face_names[0]
    except:
        # Si no se encuentran caras en la imagen de entrada o se produce un error, devuelva False
        return False