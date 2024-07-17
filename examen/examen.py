from figura import Figura
from ClaseRF import ReconocimientoFacial
import cv2

reconocimiento_facial = ReconocimientoFacial("ruta_empleados")


imagen_actual = cv2.imread("ruta_de_la_imagen_actual")

figura_usuario = input("Ingrese la figura que quiere ver (circulo/esfera): ").lower()


figura = Figura("ruta_empleados")
figura.set_figura(figura_usuario)


rostro_reconocido = figura.reconocer_rostro(imagen_actual)

if rostro_reconocido:
    if figura_usuario == "circulo":
        figura.calcular()
    elif figura_usuario == "esfera":
        figura.calcular()
else:
    if figura_usuario == "circulo":
        figura.set_figura("esfera")
        figura.calcular()
    elif figura_usuario == "esfera":
        figura.set_figura("circulo")
        figura.calcular()
