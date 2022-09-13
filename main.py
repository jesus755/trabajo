#---------importamos las librerias--------------
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
##########################-------declaramos el detector--------#######################################################
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
wCam, hCam = 648, 488
#----------------Para entrada de cámara web-----------------------:
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      #-----------------------------Si carga un video, use 'pausa' en lugar de 'continuar'---------------------
    #-------------------------------------------Para mejorar el rendimiento, opcionalmente marque la imagen como no se puede escribir en-------------
    # ------------------------------------ pasar por referencia------------------------------------
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # -----------------------Dibujar una anotación de punto de referencia en la imagen--------------------------
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    #---------------------Voltea la imagen horizontalmente para una visualización de selfie-------------.
    cv2.imshow('hola carita', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()