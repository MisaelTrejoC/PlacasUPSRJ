import cv2
import numpy as np
import pytesseract
import re
import json

# Configuración del módulo de reconocimiento de caracteres
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Expresiones regulares para los formatos de placa
format1_pattern = re.compile(r'^[A-Z]{3}[1-9]{3}[A-Z]$')  # Formato 1: ABC-123-X
format2_pattern = re.compile(r'^[A-Z]{2}[0-9]{4}[A-Z]$')  # Formato 2: AB-1234-X

# Cargar datos de prefijos de placas desde un archivo JSON
with open('placas_estados_mexico.json', 'r', encoding='utf-8') as file:
    placas_data = json.load(file)

# Función para buscar el estado basado en el prefijo de la placa
def buscar_estado_por_prefijo(prefijo):
    for entrada in placas_data:
        if entrada["prefijo"] == prefijo:
            return entrada["estado"]
    return "Estado no encontrado"

# Función para detectar placas
def detect_plate(frame):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbral adaptativo para resaltar los bordes
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Encontrar contornos en la imagen umbralizada
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Obtener el perímetro del contorno
        perimeter = cv2.arcLength(contour, True)
        
        # Aproximar un polígono al contorno
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        
        # Si el contorno tiene 4 vértices (forma rectangular)
        if len(approx) == 4:
            # Dibujar el contorno delimitador
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
            
            # Recortar y extraer la región de la placa
            x, y, w, h = cv2.boundingRect(approx)
            plate_roi = frame[y:y+h, x:x+w]
            
            # Aplicar OCR a la región de la placa
            plate_text = pytesseract.image_to_string(plate_roi, lang='eng', config='--psm 6')
            
            # Filtrar solo letras y números
            plate_text_filtered = re.sub(r'[^A-Z0-9]', '', plate_text)
            
            # Verificar si el texto cumple con el formato 1 o el formato 2
            if format1_pattern.match(plate_text_filtered):
                # Formatear el texto de la placa con guiones para Formato 1
                formatted_plate_text = f"{plate_text_filtered[:3]}-{plate_text_filtered[3:6]}-{plate_text_filtered[6]}"
                tipo_vehiculo = "Automóvil"
                
            elif format2_pattern.match(plate_text_filtered):
                # Formatear el texto de la placa con guiones para Formato 2
                formatted_plate_text = f"{plate_text_filtered[:2]}-{plate_text_filtered[2:6]}-{plate_text_filtered[6]}"
                tipo_vehiculo = "Camioneta"
                
            else:
                # Si no coincide con ningún formato, continuar con la siguiente iteración
                continue

            # Extraer los primeros 2 caracteres y buscar el estado
            prefijo_placa = plate_text_filtered[:2]
            estado = buscar_estado_por_prefijo(prefijo_placa)
            print(f"Placa detectada: {formatted_plate_text}, Estado: {estado}, Tipo: {tipo_vehiculo}")
            
            # Mostrar el texto de la placa y el estado en el marco
            cv2.putText(frame, f"{formatted_plate_text} ({estado, tipo_vehiculo})", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(1)

while True:
    # Leer un nuevo frame del flujo de video
    ret, frame = cap.read()
    
    # Si la captura de video es exitosa
    if ret:
        # Detectar las placas en el frame
        frame = detect_plate(frame)
        
        # Mostrar el frame resultante
        cv2.imshow('Placa Detector', frame)
        
        # Esperar por la tecla 'q' para salir del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
