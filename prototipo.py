# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time
import math
import bluetooth 
import speech_recognition as sr
import gtts
from gtts import gTTS
import os
from playsound import playsound

from pydub import AudioSegment
from pydub.playback import play

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

encontrar = AudioSegment.from_mp3("vou_encontrar.mp3")
cocorico = AudioSegment.from_mp3('test3.mp3')

def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  opcao = "continuar"
  
  
  while True:
    
    distancia = ""
    D = "."
    objeto = ""
    
    # Cria um reconhecedor de fala
    r = sr.Recognizer()
    
    # Reconhecimento de voz - abrir microfone
    with sr.Microphone() as source:
      r.adjust_for_ambient_noise(source) # Ajusta o ruido do microfone
      print("Diga algo!")
      audio = r.listen(source)
      
      # Usa o reconhecimento de fala em português do Brasil para reconhecer a fala
      try:
        print("Você disse: " + r.recognize_google(audio, language='pt-BR'))
      except sr.UnknownValueError:
        print("Não foi possível entender o áudio")
      except sr.RequestError as e:
        print("Erro ao solicitar o reconhecimento de fala; {0}".format(e))   
      
      fala = r.recognize_google(audio, language='pt-BR')   # 'fala=' utiliza o reconhecimento de voz
      fala = fala.lower()                                  # fala.lower() transforma todas as letras em minúsculas, solucionando erros com a palavra "Encontre"
      comando = 'encontre' in fala
      if comando != False:
        if "garrafa" in fala:
          objeto = "bottle"
          play(encontrar)
          
          
        elif "teclado" in fala:
          objeto = "keyboard"
          play(encontrar)
            
        # Variables to calculate FPS
        counter, fps = 0, 0
        start_time = time.time()

        # Start capturing video input from the camera
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Visualization parameters
        row_size = 20  # pixels
        left_margin = 24  # pixels
        text_color = (0, 0, 255)  # red
        font_size = 1
        font_thickness = 1
        fps_avg_frame_count = 10

        # Initialize the object detection model
        base_options = core.BaseOptions(
            file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
        detection_options = processor.DetectionOptions(
            max_results=3, score_threshold=0.3)
        options = vision.ObjectDetectorOptions(
            base_options=base_options, detection_options=detection_options)
        detector = vision.ObjectDetector.create_from_options(options)
        
        # Endereco MAC do ESP32
        addr = "78:E3:6D:09:2D:12"

        # Canal de servico RFCOMM
        port = 1

        # Mensagem a ser enviada

        # Cria um socket Bluetooth
        sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)

        # Conecta-se ao dispositivo ESP32
        sock.connect((addr, port))

        # Continuously capture images from the camera and run inference
        while cap.isOpened():
          success, image = cap.read()
          if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

          counter += 1
          image = cv2.flip(image, 1)

          # Convert the image from BGR to RGB as required by the TFLite model.
          rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

          # Create a TensorImage object from the RGB image.
          input_tensor = vision.TensorImage.create_from_array(rgb_image)

          # Run object detection estimation using the model.
          detection_result = detector.detect(input_tensor)
          
          # Escreve os resultados da detecção em um arquivo de texto
          with open('rasp_objetos.txt','w') as file:
            file.write(str(detection_result))
          with open('rasp_objetos.txt') as leitura:

           # Divide os resultados em itens de uma lista
            
            result = leitura.read()
            list = result[28:-1].split(",")
            
            x = []
            y = []
            w = []
            h = []
            category = []
            categoryTratada = []
          # Define as variáveis para medição de distância
          #Objeto 1
          if len(list) >= 7:
            item_0 = list[0].split("=")
            x1 = int(item_0[2])
            item_1 = list[1].split("=")
            y1 = int(item_1[1])
            item_2 = list[2].split("=")
            w1 = int(item_2[1])
            item_3 = list[3].split("=")
            h1 = int(item_3[1][:-1])
            item_4 = list[7].split("=")
            category_0 = item_4[1][1:-4]
            
            x.append(x1)
            y.append(y1)
            w.append(w1)
            h.append(h1)
            category.append(category_0)
            
          #Objeto 2
          if len(list) >= 15:
            item_10 = list[8].split("=")
            x2 = int(item_10[2])
            item_11 = list[9].split("=")
            y2 = int(item_11[1])
            item_12 = list[10].split("=")
            w2 = int(item_12[1])
            item_13 = list[11].split("=")
            h2 = int(item_13[1][:-1])
            item_14 = list[15].split("=")
            category_1 = item_14[1][1:-4]
            
            x.append(x2)
            y.append(y2)
            w.append(w2)
            h.append(h2)
            category.append(category_1)
                  
          #Objeto 3
          if len(list) >= 23:
            item_20 = list[16].split("=")
            x3 = item_20[2]
            item_21 = list[17].split("=")
            y3 = item_21[1]
            item_22 = list[18].split("=")
            w3 = item_22[1]
            item_23 = list[19].split("=")
            h3 = item_23[1][:-1]
            item_24 = list[23].split("=")
            category_2 = item_24[1][1:-4]
            
            x.append(x3)
            y.append(y3)
            w.append(w3)
            h.append(h3)
            category.append(category_2)

          
          #Objeto 4
          if len(list) >= 31:
            item_30 = list[24].split("=")
            x4 = item_30[2]
            item_31 = list[25].split("=")
            y4 = item_31[1]
            item_32 = list[26].split("=")
            w4 = item_32[1]
            item_33 = list[27].split("=")
            h4 = item_33[1][:-1]
            item_34 = list[31].split("=")
            category_3 = item_34[1][1:-4]
            
            x.append(x4)
            y.append(y4)
            w.append(w4)
            h.append(h4)
            category.append(category_3)
            
          for cat in category:
              if cat.find("'"):
                cat = cat.replace("'", "")
                categoryTratada.append(cat)
                
          
                
          #Medir distância em pixels entre um celular e um objeto a ser definido
            
          if len(list) >= 15:
            if 'cell phone' in categoryTratada:
              obj_A = categoryTratada.index('cell phone')
            
              if objeto in categoryTratada:
                obj_B = categoryTratada.index(objeto)
                
                xa = int(x[obj_A])
                ya = int(y[obj_A])
                wa = int(w[obj_A])
                ha = int(h[obj_A])
                
                xb = int(x[obj_B])
                yb = int(y[obj_B])
                hb = int(h[obj_B])
                wb = int(w[obj_B])
                
                # calcular as coordenadas dos pontos médios dos dois retângulos
                center1 = (xa + wa // 2, ya + ha // 2)
                center2 = (xb + wb // 2, yb + hb // 2)
                # calcular a distância entre os pontos médios
                distance = int(math.sqrt((center2[0] - center1[0])**2 + (center2[1] - center1[1])**2))
                # desenhar uma linha entre os pontos médios
                cv2.line(image, center1, center2, (0, 0, 255), 2)
                # adicionar um texto com a distância entre os pontos médios na imagem
                cv2.putText(image, f"Distancia: {distance} pixels", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                
                '''if distance >=300:
                  sock.send('A')
                elif distance >=280:
                  sock.send('B')
                elif distance >=260:
                  sock.send('C')
                elif distance >=220:
                  sock.send('D')
                elif distance >=140:                
                  cv2.imshow('object_detector', image)
                  cap.release()
                  cv2.destroyAllWindows()
                '''     
                  
                distancia = 'A'
                if distancia != D:
                  sock.send(distancia)
                  D = 'A'
                  print(distancia)
              
            
                if distance <=299 and distance >=260:
                  distancia = 'B'
                  if distancia != D:
                    sock.send(distancia)
                    D = 'B'
                    print(distancia)

                if distance <=259 and distance >=230:
                  distancia = 'C'
                  if distancia != D:
                    sock.send(distancia)
                    D = 'C'
                    print(distancia)
            
                if distance <=229 and distance >=200:
                  distancia = 'D'
                  if distancia != D:
                    sock.send(distancia)
                    D = 'D'
                    print(distancia)
            
                if distance <=199 and distance >=160:
                  distancia = 'E'
                  if distancia != D:
                    sock.send(distancia)
                    D = 'E'
                    print(distancia)
            
                if distance <= 159:
                  distancia = 'F'
                  if distancia != D:
                    sock.send(distancia)
                    D = 'F'
                    print(distancia)
                    #play(cocorico)
                    cap.release()
                    cv2.destroyAllWindows()
                    sock.close()
                    
                    with sr.Microphone() as source:
                      r.adjust_for_ambient_noise(source) # Ajusta o ruido do microfone
                      print("Diga algo!")
                      audio = r.listen(source)
                      
                      # Usa o reconhecimento de fala em português do Brasil para reconhecer a fala
                      try:
                        print("Você disse: " + r.recognize_google(audio, language='pt-BR'))
                      except sr.UnknownValueError:
                        print("Não foi possível entender o áudio")
                      except sr.RequestError as e:
                        print("Erro ao solicitar o reconhecimento de fala; {0}".format(e))   
                      
                      fala = r.recognize_google(audio, language='pt-BR')   # 'fala=' utiliza o reconhecimento de voz
                      fala = fala.lower()                                  # fala.lower() transforma todas as letras em minúsculas, solucionando erros com a palavra "Encontre"
                      comando = 'encontrei' in fala
                      if comando != False:
                        if "encontrei" in fala:
                          print("MUITO BEM! Esperando proximo comando")
                          play(encontrar)
                    
                  
                # enviar o sinal de distância para o esp32 por bluetooth
                
                
                    
          #mensagem de teste
          if ('bottle') in result and ('cell phone') in result:
              print('achei um celular e uma garrafa')
            

          # Draw keypoints and edges on input image
          image = utils.visualize(image, detection_result)

          # Calculate the FPS
          if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()

          # Show the FPS
          fps_text = 'FPS = {:.1f}'.format(fps)
          text_location = (left_margin, row_size)
          cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                      font_size, text_color, font_thickness)
          
          # Stop the program if the ESC key is pressed.
          if cv2.waitKey(1) == 27:
            break
          cv2.imshow('object_detector', image)

        cap.release()
        cv2.destroyAllWindows()
        sock.close()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))


if __name__ == '__main__':
  main()


