import face_recognition
import cv2
import numpy as np

# carregue uma imagem de amostra e aprenda a reconhecê-la
obamaImage = face_recognition.load_image_file("C:/Users/Carolina/obamaExample.jpg")
obamaFaceEncoding = face_recognition.face_encodings(obamaImage)[0]

# carregue uma imagem de amostra e aprenda a reconhecê-la
elonMuskImage = face_recognition.load_image_file("C:/Users/Carolina/elonExample.jpg")
elonMuskFaceEncoding = face_recognition.face_encodings(elonMuskImage)[0]

#criar array do reconhecimento facil e seus nomes
knownFaceEncoding =[
    obamaFaceEncoding,
    elonMuskFaceEncoding,
]

knownFaceNames = [
    "Barack Obama",
    "Elon Musk"
]

faceLocations = []
faceEncodings = []
faceNames = []
processThisFrame = True

#pega a referencia da webcam
capture = cv2.VideoCapture(1)

while True:
    #pega um quadro de video
    ret, frame = capture.read()

    #redimensione o quadro do vídeo para 1/4 do tamanho para um processamento de reconhecimento facial mais rápido
    smallFrame =  cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgbFrameSmall = smallFrame[:,:,::-1]

    #processe apenas todos os outros quadros de vídeo para economizar tempo
    if processThisFrame:
        #encontre todos os rostos e codificações de rosto no quadro atual do vídeo
        faceLocations = face_recognition.face_locations(rgbFrameSmall)
        faceEncodings = face_recognition.face_encodings(rgbFrameSmall,faceLocations)

        faceNames = []
        for faceEncoding in faceEncodings:
            #veja se o rosto é compatível com o rosto conhecido
            matches = face_recognition.compare_faces(knownFaceEncoding,faceEncoding)
            name = "Unknown"

            faceDistances = face_recognition.face_distance(knownFaceEncoding,faceEncoding)
            bestMatchIndex = np.argmin(faceDistances)

            if matches[bestMatchIndex]:
                name = knownFaceEncoding[bestMatchIndex]

            faceNames.append(name)

    processThisFrame = not processThisFrame

    for(top,right,bottom,left),name in zip(faceLocations,faceNames):
        top*=4
        right*=4
        bottom*=4
        left *=4

        #desenha o retangulo no rosto
        cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)

        #desenha o nome em baixo do rosto
        cv2.rectangle(frame,(left,bottom -35),(right,bottom),(0,255,0),cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame,name,(left+4, bottom-4),font,1,(0,0,255),2)

        #mostra o resultado da imagem
        cv2.imshow('Image',frame)

        #aperte q para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #desconecta a webcam
    capture.release()
    cv2.destroyAllWindows()