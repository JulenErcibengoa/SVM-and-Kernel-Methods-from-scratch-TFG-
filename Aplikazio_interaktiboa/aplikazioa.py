import sys
import os

oraingo_bidea = os.path.dirname(os.path.realpath(__file__))
bide_orokorra = os.path.dirname(oraingo_bidea)
sys.path.insert(0,os.path.join(bide_orokorra, "Algoritmoak"))

import pygame # Digituak idazteko
import numpy as np
import time
import pandas as pd
from sklearn.svm import SVC # Jada SVM inplementatutako pythoneko pakete bat
from random import randint
import pickle
import plotly.graph_objects as go
import pygame_gui
from SGD_soft_SVM_Kernels import Nire_SGD_kernelekin

# Parametroak (alda daitezke):
grid_size = 20  # Karratu bakoitzaren tamaina pixeletan (minimoa 20)

# Koloreak
WHITE = (238,238,210)
BLACK = (186,202,68)
GRID_COLOR = (200, 200, 200) # Karratuen arteko kolorea eta pantaila atzeko kolorea
TEXT_COLOR = (0,0,0)
BUTTON_COLOR = (200, 200, 200)
GREEN = (0, 255, 0)
RED =(255,0,0)

# Hemendik aurrerakoa ez aldatu













# -------------------------------------------------------------------------------
# -----------------------------DATU BASEA INPORTATU------------------------------
# -------------------------------------------------------------------------------
mnist_test_bidea = os.path.join(bide_orokorra, "Datu_basea\mnist_test.csv")
mnist_train_bidea = os.path.join(bide_orokorra, "Datu_basea\mnist_train.csv")

entrenamendu_datuak = pd.read_csv(mnist_train_bidea)
testeatzeko_datuak = pd.read_csv(mnist_test_bidea)
# print(entrenamendu_datuak.head())
X_entrenamendu = entrenamendu_datuak.iloc[:,1:]
Y_entrenamendu = entrenamendu_datuak["label"]

X_test = testeatzeko_datuak.iloc[:,1:]
Y_test = testeatzeko_datuak["label"]

# zutabeen_izenak = X_entrenamendu.columns.tolist()
# print(zutabeen_izenak)

# -------------------------------------------------------------------------------
# ------------------------------MODELOAK INPORTATU-------------------------------
# -------------------------------------------------------------------------------

# Scikit modelo hoberena GAUSS:
Scikit_modelo_hoberena_gauss = pickle.load(open(os.path.join(bide_orokorra,"Entrenatutako_modeloak","Scikit_modelo_hoberena_gauss.pkl"),"rb"))
Scikit_modelo_hoberena_gauss_info = pickle.load(open(os.path.join(bide_orokorra,"Entrenatutako_modeloak","Scikit_modelo_hoberena_gauss_info.pkl"),"rb"))
# print(Scikit_modelo_hoberena_gauss_info)

# Scikit modelo hoberena POLY:
Scikit_modelo_hoberena_poly = pickle.load(open(os.path.join(bide_orokorra,"Entrenatutako_modeloak","Scikit_modelo_hoberena_poly.pkl"),"rb"))
Scikit_modelo_hoberena_poly_info = pickle.load(open(os.path.join(bide_orokorra,"Entrenatutako_modeloak","Scikit_modelo_hoberena_poly_info.pkl"),"rb"))
# print(Scikit_modelo_hoberena_poly_info)

# Nire modelo hoberena GAUSS:
Nire_modelo_hoberena_gauss = pickle.load(open(os.path.join(bide_orokorra,"Entrenatutako_modeloak","Nire_modelo_hoberena_gauss.pkl"),"rb"))
Nire_modelo_hoberena_gauss_info = pickle.load(open(os.path.join(bide_orokorra,"Entrenatutako_modeloak","Nire_modelo_hoberena_gauss_info.pkl"),"rb"))
# print(Nire_modelo_hoberena_gauss_info)

# Nire modelo hoberena POLY:
Nire_modelo_hoberena_poly = pickle.load(open(os.path.join(bide_orokorra,"Entrenatutako_modeloak","Nire_modelo_hoberena_poly.pkl"),"rb"))
Nire_modelo_hoberena_poly_info = pickle.load(open(os.path.join(bide_orokorra,"Entrenatutako_modeloak","Nire_modelo_hoberena_poly_info.pkl"),"rb"))
# print(Nire_modelo_hoberena_poly_info)












# -------------------------------------------------------------------------------
# ----------------------------APLIKAZIO INTERAKTIBOA-----------------------------
# -------------------------------------------------------------------------------

# Pantailaren tamainiako aldagai orokorrak:
n = 28

grid_width, grid_height = n * grid_size, n * grid_size
width = grid_width + 300 # Instrukzioak / botoiak gehitzeko
height = grid_height


# Hainbat funtzio auxiliar:
def kuadrikula_marraztu(kuadrikula,screen):
    """
    grid = 28 x 28 tamainiako matrize bat: matrizearen elementuak 0 - 255 tarteko zenbakiak
    izango dira, 28 x 28 pixeleko pantaila baten gris-eskalak izanik
    """
    screen.fill(WHITE)
    for i in range(n):
        for j in range(n):
            balioa = kuadrikula[i][j]
            pygame.draw.rect(screen, (255-balioa,255-balioa,255-balioa), (j * grid_size, i * grid_size, grid_size, grid_size))
            pygame.draw.rect(screen, GRID_COLOR, (j * grid_size, i * grid_size, grid_size, grid_size), 1) # 1 Balioa da
            # karratuaren zenbat betetzen den definitzeko, 0 da "default" balioa eta karratu osoa betetzen du.

def digituak_marraztu(zenbakia,errenkada,zutabea):
    # Klikatutako karratua
    if 0 <= errenkada < n and 0 <= zutabea < n:
        zenbakia[errenkada][zutabea] = 255
    inguruneak = [(1,0),(-1,0),(0,1),(0,-1)]
    for dx,dy in inguruneak:
        zut = zutabea + dx
        errenk = errenkada + dy
        if 0 <= errenk < n and 0 <= zut < n and zenbakia[errenk][zut] <= 200:
            zenbakia[errenk][zut] = randint(150,220)

def digituak_garbitu(zenbakia,errenkada,zutabea):
    # Klikatutako karratua
    if 0 <= errenkada < n and 0 <= zutabea < n:
        zenbakia[errenkada][zutabea] = 0
    inguruneak = [(1,0),(-1,0),(0,1),(0,-1)]
    for dx,dy in inguruneak:
        zut = zutabea + dx
        errenk = errenkada + dy
        if 0 <= errenk < n and 0 <= zut < n:
            zenbakia[errenk][zut] = 0

def modeloa_sortu(puntuak,izenak,kernel_mota="kernel gaussiarra",koefizientea = 4,modelo_mota = "Scikit modeloa", parametroa = 2):
    X = np.array(puntuak)
    Y = np.array(izenak)
    mean = np.mean(X,0)
    sd = np.std(X,0)
    X = (X-mean)/sd

    X[:,1] = -X[:,1]
    print(f"INFO \n{kernel_mota},{modelo_mota}\n")
    if modelo_mota == "Scikit modeloa":
        if kernel_mota == "kernel gaussiarra":
            model = SVC(C = koefizientea, kernel = "rbf", gamma = parametroa)
        elif kernel_mota == "kernel polinomiala":
            model = SVC(C = koefizientea, kernel = "poly",coef0=1, degree= int(parametroa))
        elif kernel_mota == "kernel lineala":
            model = SVC(C = koefizientea, kernel = "linear")
    else:
        model = Nire_SGD_kernelekin(koeficient= koefizientea, kernel = kernel_mota, sigma= parametroa, deg= int(parametroa))


    #print(f"Modeloaren parametroak = \n{model.get_params()}")
    model.fit(X,Y)

    # Izenak aldatzeko:
    koefiziente_izena = "C" if modelo_mota == "Scikit modeloa" else u"\u03BB"
    parametro_izena = u"\u03C3" if modelo_mota != "Scikit modeloa" else u"\u03B3"

    if kernel_mota == "kernel polinomiala":
        parametro_izena = "maila"
    
    if kernel_mota == "kernel lineala":
        testua = f'Modeloaren erabaki-gainazala <br><sup>Modelo mota = {modelo_mota}, Kernel = {kernel_mota}, {koefiziente_izena} = {koefizientea}, Asmatze proportzioa = {round(model.score(X,Y),3)}</sup>'
    else:
        testua = f'Modeloaren erabaki-gainazala <br><sup>Modelo mota = {modelo_mota}, Kernel = {kernel_mota}, {koefiziente_izena} = {koefizientea}, {parametro_izena} = {parametroa}, Asmatze proportzioa = {round(model.score(X,Y),3)}</sup>'



    # Datuen minimo eta maximoak
    h = 0.01 if modelo_mota == "Scikit modeloa" else 0.1
    x_min, x_max = np.min(X[:, 0]) - 0.2, np.max(X[:, 0]) + 0.2
    y_min, y_max = np.min(X[:, 1]) - 0.2, np.max(X[:, 1]) + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
    if modelo_mota == "Scikit modeloa":
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    else: 
        Z = model.predict_anitzkoitza(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig = go.Figure(data=go.Contour(
    z=Z,
    x=np.arange(x_min, x_max, h),
    y=np.arange(y_min, y_max, h),
    colorscale='RdBu',  # Especificar la escala de colores
    opacity=0.6,  # Opacidad de los contornos
    showlegend=False
))
    fig.add_trace(go.Scatter(
    x=X[:, 0],
    y=X[:, 1],
    mode='markers',
    marker=dict(
        color=Y,
        colorscale='RdBu',
        size=8,
        line=dict(width=1, color='Black'),
    ),
    showlegend=False
))
    fig.update_layout(
    xaxis_title=r'$x_1$',
    yaxis_title=r'$x_2$',
    title=testua,
    
)
    fig.show()

# Botoien funtzionalitate orokorra         
class Botoia:
    def __init__(self, x, y, width, height, color, text, font_size = 24):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.text = text
        self.font = pygame.font.Font(None, font_size)

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)
        text_surface = self.font.render(self.text, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

# Menu desberdiñak: 
def adibideak_ikusi():
    """
    Programa honek MNIST datu-baseko balioak ikusteko balio du
    """
    # Pygame hasi
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("MNIST datu-baseko hainbat adibide ikusi")

    # Beharrezko hainbat aldagai
    running = True
    zenbagarren_adibidea = 0

    # Textuak idazteko beharrezkoa:
    izenburua = pygame.font.Font(None, 30)
    digitua = pygame.font.Font(None, 300)
    instrukzioak = pygame.font.Font(None,20)

    # Botoia:
    botoia_hurrengo_adibidea = Botoia(width - 230, 325, 170, 40, BUTTON_COLOR, "Hurrengo adibidea")
    botoia_itzuli_menura = Botoia(width - 230, height - 50, 170, 40, BUTTON_COLOR, "Menura itzuli")

    # Loop orokorra
    while running:
        for event in pygame.event.get():
            keys = pygame.key.get_pressed()
            if event.type == pygame.QUIT: # Exekutatzeaz bukatzeko
                running = False
                zer_egin = "amaitu"
            elif keys[pygame.K_RETURN]: # Enter botoia sakatzerakoan
                zenbagarren_adibidea = randint(0,1000)
            elif event.type == pygame.MOUSEBUTTONDOWN: # Xaguaren botoiren bat sakatzen bada
                if event.button == 1: # Ezkerreko botoia sakatzen bada
                    if botoia_itzuli_menura.rect.collidepoint(event.pos):
                        running = False
                        zer_egin = "menua_ireki"
                    elif botoia_hurrengo_adibidea.rect.collidepoint(event.pos):
                        zenbagarren_adibidea = randint(0,1000)

        # while buklearen amaieran, pantaila marraztu, textua idatzi eta aktualizatzeko
        kuadrikula_marraztu(np.array(X_entrenamendu.iloc[zenbagarren_adibidea,:]).reshape(n,n),screen)
        screen.blit(izenburua.render("MNIST datu-baseko adibideak", True, TEXT_COLOR),(width - 295, 30))
        screen.blit(digitua.render(str(Y_entrenamendu[zenbagarren_adibidea]), True, BLACK),(width - 205, 130))
        screen.blit(instrukzioak.render("Sakatu 'enter' hurrengo digitua ikusteko", True, TEXT_COLOR),(width - 295, 55))
        botoia_itzuli_menura.draw(screen)
        botoia_hurrengo_adibidea.draw(screen)
        pygame.display.flip()
        
    return zer_egin

def predezitu():
    """
    Programa honek xaguarekin digituak idazteko balio du, gero modeloak predezitzeko
    """
    # Pygame hasi
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Digituak predezitu")

    # Beharrezko hainbat aldagai
    botoien_dist = height//10
    running = True
    marrazten = False
    garbitzen = False
    zenbakia = np.zeros((n,n),dtype=int)
    predikzioa = "1"
    modeloak = [Scikit_modelo_hoberena_gauss, Scikit_modelo_hoberena_poly, Nire_modelo_hoberena_gauss, Nire_modelo_hoberena_poly]
    modeloen_informazioak = [Scikit_modelo_hoberena_gauss_info, Scikit_modelo_hoberena_poly_info, Nire_modelo_hoberena_gauss_info, Nire_modelo_hoberena_poly_info]
    modeloen_izenak = ["Scikit kernel gaussiarra", "Scikit kernel polinomiala", "Nirea, kernel gaussiarra", "Nirea, kernel polinomiala"]
    zein_modelo_indize = 0
    modeloa = modeloak[zein_modelo_indize]
    modeloaren_informazioa = modeloen_informazioak[zein_modelo_indize]



    # Textuak idazteko beharrezkoa:
    izenburua = pygame.font.Font(None, 37)
    digitua = pygame.font.Font(None, 300)
    instrukzioak = pygame.font.Font(None,20)
    modelo_letrak = pygame.font.Font(None,25)
    zein_modelo_font = pygame.font.Font(None,35)
    oharra = pygame.font.Font(None,20)

    # Botoia:
    botoia_garbitu = Botoia(width - 230, 325, 170, 40, BUTTON_COLOR, "Garbitu")
    botoia_itzuli_menura = Botoia(width - 230, height - 50, 170, 40, BUTTON_COLOR, "Menura itzuli")
    botoia_modeloa_aldatu = Botoia(width - 230, 325 + botoien_dist, 170, 40, BUTTON_COLOR, "Modeloz aldatu")
    botoia_predezitu = Botoia(width - 230, 325 + 2*botoien_dist, 170, 40, BUTTON_COLOR, "Predezitu zenbakia")
    

    # Loop orokorra
    while running:
        if np.array_equal(zenbakia,np.zeros((n,n))):
            predikzioa = "?"
        elif zein_modelo_indize == 0 or zein_modelo_indize == 1: # Scikit modeloen kasuan:
           predikzioa = modeloa.predict(zenbakia.reshape((1,n**2)) / 255)[0]
            # berria = zenbakia.reshape((1,n**2)) / 255
            # predikzioa = modeloa_nirea.predict(berria.tolist()[0])

        for event in pygame.event.get():
            keys = pygame.key.get_pressed()
            if event.type == pygame.QUIT: # Exekutatzeaz bukatzeko
                running = False
                zer_egin = "amaitu"
            elif keys[pygame.K_RETURN]: # Enter botoia sakatzerakoan pantaila garbitu
                zenbakia = np.zeros((n,n))
            elif event.type == pygame.MOUSEBUTTONDOWN: # Xaguaren botoiren bat sakatzen bada
                if event.button == 1: # Ezkerreko botoia sakatzen bada
                    if botoia_itzuli_menura.rect.collidepoint(event.pos):
                        running = False
                        zer_egin = "menua_ireki"
                    elif botoia_garbitu.rect.collidepoint(event.pos):
                        zenbakia = np.zeros((n,n))
                    elif botoia_modeloa_aldatu.rect.collidepoint(event.pos): # Modeloz aldatu
                        zein_modelo_indize = (zein_modelo_indize + 1) % 4
                        if zein_modelo_indize == 2 or zein_modelo_indize == 3:
                            predikzioa = "?"
                        modeloa = modeloak[zein_modelo_indize]
                        modeloaren_informazioa = modeloen_informazioak[zein_modelo_indize]

                    elif botoia_predezitu.rect.collidepoint(event.pos) and (zein_modelo_indize == 2 or zein_modelo_indize == 3):
                        berria = zenbakia.reshape((1,n**2)) / 255
                        predikzioa = modeloa.predict(berria.tolist()[0])
                    else:    
                        marrazten = True
                        x,y = event.pos
                        zutabea = x // grid_size
                        errenkada = y // grid_size
                        digituak_marraztu(zenbakia,errenkada,zutabea)
                elif event.button == 3: # Eskubiko botoia sakatzen bada
                    garbitzen = True
                    x,y = event.pos
                    zutabea = x // grid_size
                    errenkada = y // grid_size
                    digituak_garbitu(zenbakia,errenkada,zutabea)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Botón izquierdo del ratón
                    marrazten = False
                if event.button == 3:
                    garbitzen = False
            elif event.type == pygame.MOUSEMOTION: # Xagua mugitzen bada
                if marrazten:
                    x,y = event.pos
                    zutabea = x // grid_size
                    errenkada = y // grid_size
                    digituak_marraztu(zenbakia,errenkada,zutabea)
                elif garbitzen:
                    x,y = event.pos
                    zutabea = x // grid_size
                    errenkada = y // grid_size
                    digituak_garbitu(zenbakia,errenkada,zutabea)

        # while buklearen amaieran, pantaila marraztu, textua idatzi eta aktualizatzeko
        kuadrikula_marraztu(zenbakia,screen)
        screen.blit(izenburua.render("Modeloaren predikzioa", True, TEXT_COLOR),(width - 290, 20))
        screen.blit(digitua.render(str(predikzioa), True, BLACK),(width - 215, 80))
        screen.blit(instrukzioak.render("Sakatu 'enter' irudia garbitzeko", True, TEXT_COLOR),(width - 295, 55))
        screen.blit(zein_modelo_font.render(modeloen_izenak[zein_modelo_indize], True, TEXT_COLOR),(width - 295, 250))
        screen.blit(modelo_letrak.render(f"Modeloaren nota = {modeloaren_informazioa[1]}", True, TEXT_COLOR),(width - 295, 275))
        
        botoia_itzuli_menura.draw(screen)
        botoia_garbitu.draw(screen)
        botoia_modeloa_aldatu.draw(screen)
        if zein_modelo_indize == 2 or zein_modelo_indize == 3:
            botoia_predezitu.draw(screen)
            screen.blit(oharra.render("Nire modeloak denbora gehiago", True, RED),(width - 295, 290))
            screen.blit(oharra.render("behar du predezitzeko", True, RED),(width - 295, 305))
        else:
            screen.blit(oharra.render("Scikit-learn modeloak automatikoki", True, RED),(width - 295, 290))
            screen.blit(oharra.render("predezitzen du zenbakia", True, RED),(width - 295, 305))
        pygame.display.flip()
        
    return zer_egin

def SVM_bisuala():
    """
    Programa honek xaguarekin laginak sortzeko balio du ondoren bisualki SVM-ren outputa ikusteko
    """
    # Pygame hasi
    screen = pygame.display.set_mode((width, height))
    grafikoa = pygame.Surface((grid_width,grid_height))
    pygame.display.set_caption("SVM bisualizazioa")
    screen.fill(WHITE)
    grafikoa.fill((0,0,0))
    botoien_distantzia = height // 10

    # Modeloaren koefizientea aldatzeko barra sortzeko:
    manager = pygame_gui.UIManager((width, height))
    slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect((width - 230, height - 50 - 3 * botoien_distantzia - 30 - 20), (170, 20)),
        start_value=0,  # Valor inicial del deslizador
        value_range=(-4, 5),  # Rango de valores del deslizador
        manager=manager)
    slider2 = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect((width - 230, height - 50 - 3 * botoien_distantzia - 30), (170, 20)),
        start_value=10,  # Valor inicial del deslizador
        value_range=(0, 15),  # Rango de valores del deslizador
        manager=manager)

    # Beharrezko hainbat aldagai
    running = True
    marrazten = False
    puntu_berdeak = True
    berdeak = []
    gorriak = []
    puntu_guztiak = []
    izenak = []
    i = 0
    j = 0
    clock = pygame.time.Clock()
    delta_time = clock.tick(60) / 1000.0
    kernel_motak = ["kernel gaussiarra","kernel polinomiala","kernel lineala"]
    zein_kernel = "kernel gaussiarra"
    zein_modelo = "Scikit modeloa"
    koefizientea = 1
    parametroa = 8
    parametro_mota = u"\u03B3"

    # Textuak idazteko beharrezkoa:
    izenburua = pygame.font.Font(None, 37)
    instrukzioak = pygame.font.Font(None,20)
    puntuak = pygame.font.Font(None,30)

    # Botoia:
    botoia_garbitu = Botoia(width - 230, height - 50 - 1 * botoien_distantzia, 170, 40, BUTTON_COLOR, "Garbitu")
    botoia_itzuli_menura = Botoia(width - 230, height - 50, 170, 40, BUTTON_COLOR, "Menura itzuli")
    botoia_modeloz_aldatu = Botoia(width - 230, height - 50 - 2 * botoien_distantzia, 170, 40, BUTTON_COLOR, "Modeloz aldatu")
    botoia_kernel_mota = Botoia(width - 230, height - 50 - 3 * botoien_distantzia, 170, 40, BUTTON_COLOR, "Kernel mota")

    # Loop orokorra
    while running:
        for event in pygame.event.get():
            manager.process_events(event)
            keys = pygame.key.get_pressed()
            if event.type == pygame.QUIT: # Exekutatzeaz bukatzeko
                running = False
                zer_egin = "amaitu"
            elif keys[pygame.K_RETURN]: # Enter botoia sakatzerakoan
                if not(len(gorriak) == 0 or len(berdeak) == 0):
                    modeloa_sortu(puntu_guztiak,izenak,zein_kernel,koefizientea,modelo_mota=zein_modelo, parametroa = parametroa)

            elif event.type == pygame.MOUSEBUTTONDOWN: # Xaguaren botoiren bat sakatzen bada
                if event.button == 1: # Ezkerreko botoia sakatzen bada
                    if botoia_itzuli_menura.rect.collidepoint(event.pos):
                        running = False
                        zer_egin = "menua_ireki"
                    elif botoia_garbitu.rect.collidepoint(event.pos):
                        grafikoa.fill((0,0,0))
                        berdeak = []
                        gorriak = []
                        izenak = []
                        puntu_guztiak = []
                    elif botoia_kernel_mota.rect.collidepoint(event.pos):
                        j = (j + 1)%3
                        zein_kernel = kernel_motak[j]
                        if j == 0 and zein_modelo == "Scikit modeloa": # Kernel gaussiarraren kasuan:
                            parametro_mota = u"\u03B3"
                        elif j == 0 and zein_modelo == "Nire modeloa":
                            parametro_mota = u"\u03C3"
                        elif j == 1:
                            parametro_mota = "maila"
                    
                    elif botoia_modeloz_aldatu.rect.collidepoint(event.pos):
                        if zein_modelo == "Scikit modeloa":
                            zein_modelo = "Nire modeloa"
                        else:
                            zein_modelo = "Scikit modeloa"
                        if j == 0 and zein_modelo == "Scikit modeloa": # Kernel gaussiarraren kasuan:
                            parametro_mota = u"\u03B3"
                        elif j == 0 and zein_modelo == "Nire modeloa":
                            parametro_mota = u"\u03C3"
                        elif j == 1:
                            parametro_mota = "maila"

                    else: # Puntu berdeak marrazteko   
                        marrazten = True
                        puntu_berdeak = True
                        x,y = event.pos
                        if 0 <= x < grid_width and 0 <= y < grid_height:
                            berdeak.append(event.pos)
                            puntu_guztiak.append(event.pos)
                            izenak.append(1)
                elif event.button == 3: # Eskubiko botoia sakatzen bada puntu gorriak marraztu
                    marrazten = True
                    puntu_berdeak = False
                    x,y = event.pos
                    if 0 <= x < grid_width and 0 <= y < grid_height:
                        gorriak.append(event.pos)
                        puntu_guztiak.append(event.pos)
                        izenak.append(-1)

            elif event.type == pygame.MOUSEBUTTONUP:
                marrazten = False
            elif event.type == pygame.MOUSEMOTION: # Xagua mugitzen bada
                if marrazten and puntu_berdeak and i % 20 == 0: # i sartzen dugu "cadencia" bat egoteko puntuen gehikuntzan
                    x,y = event.pos
                    if 0 <= x < grid_width and 0 <= y < grid_height:
                        berdeak.append(event.pos)
                        puntu_guztiak.append(event.pos)
                        izenak.append(1)
                if marrazten and not puntu_berdeak and i % 20 == 0: 
                    x,y = event.pos
                    if 0 <= x < grid_width and 0 <= y < grid_height:
                        gorriak.append(event.pos)
                        puntu_guztiak.append(event.pos)
                        izenak.append(-1)  
                i += 1

        # while buklearen amaieran, pantaila marraztu, textua idatzi eta aktualizatzeko
        screen.fill(WHITE)
        screen.blit(grafikoa,(0,0))

        # Modeloaren koefizientea lortu
        manager.update(delta_time)
        manager.draw_ui(screen)
        koefizientea = round(2**(slider.get_current_value()),3)

        # Parametroa lortu
        if j == 1: # Polinomiala
            parametroa = round(slider2.get_current_value()) + 2
        else:
            parametroa = round(4 **(slider2.get_current_value() - 10),8)


        # Puntuak marraztu
        for punto in berdeak:
            pygame.draw.circle(screen, GREEN, punto, 5)
        for punto in gorriak:
            pygame.draw.circle(screen, RED, punto, 5)
        
        screen.blit(izenburua.render("SVM modelo sortzailea", True, BLACK),(width - 290, 20))
        screen.blit(instrukzioak.render("- Marraztu lagina eta modeloa sortu", True, TEXT_COLOR),(width - 295, 55))
        screen.blit(instrukzioak.render("- Sakatu 'enter' modeloa sortzeko", True, TEXT_COLOR),(width - 295, 75))
        screen.blit(instrukzioak.render("- Kontuz! Puntu berde eta gorri kopurua", True, TEXT_COLOR),(width - 295, 95))
        screen.blit(instrukzioak.render("  antzekoa izan dadila, bestela baliteke", True, TEXT_COLOR),(width - 295, 115))
        screen.blit(instrukzioak.render("  modelorik ez lortzea!", True, TEXT_COLOR),(width - 295, 135))
        screen.blit(puntuak.render(f"Puntu berde kopurua = {len(berdeak)}", True, (0,100,0)),(width - 295, 175 - 25))
        screen.blit(puntuak.render(f"Puntu gorri kopurua = {len(gorriak)}", True, (100,0,0)),(width - 295, 195- 25))
        screen.blit(puntuak.render(f"Modelo mota:", True, TEXT_COLOR),(width - 295, 215- 25))
        screen.blit(puntuak.render(zein_modelo, True, TEXT_COLOR),(width - 270, 235- 25))
        screen.blit(puntuak.render(zein_kernel, True, TEXT_COLOR),(width - 270, 255- 25))
        elementua = "C" if zein_modelo == "Scikit modeloa" else u"\u03BB"
        screen.blit(puntuak.render(f"{elementua} = {koefizientea}", True, TEXT_COLOR),(width - 270, 275- 25))
        if j != 2:
            screen.blit(puntuak.render(f"{parametro_mota} = {parametroa}", True, TEXT_COLOR),(width - 270, 295- 25))
        botoia_itzuli_menura.draw(screen)
        botoia_garbitu.draw(screen)
        botoia_modeloz_aldatu.draw(screen)
        botoia_kernel_mota.draw(screen)

        pygame.display.flip()
        

    return zer_egin

def menua():
    """
    Programa hau menua exekutatzeko da
    """
    # Pygame hasi
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("MNIST interaktiboa")

    # Irudiak:
    botoien_dist = height//10
    irudia = pygame.image.load(os.path.join(bide_orokorra,"Aplikazio_interaktiboa","Menu_figure.png")) 
    irudia = pygame.transform.scale(irudia, (botoien_dist * 5.5, botoien_dist*4))
    irudia_rect = irudia.get_rect()
    irudia_rect.center = (width // 2, height // 2 - botoien_dist* 1.3)

    # Beharrezko aldagaia
    running = True

    # Textuak idazteko beharrezkoa:
    izenburua = pygame.font.Font(None, 107)

    # Botoia:
    botoia_SVC_bisualizadorea = Botoia(width//2-350//2, height // 2 + botoien_dist * 1, 350, 40, BUTTON_COLOR, "Sortu laginak eta SVM modeloak")
    botoia_adibideak_ikusi = Botoia(width//2-350//2, height // 2 + botoien_dist * 2 , 350, 40, BUTTON_COLOR, "Ikusi MNIST-en adibideak")
    botoia_digituak_predezitu = Botoia(width//2-350//2, height // 2 + botoien_dist * 3 , 350, 40, BUTTON_COLOR, "Marraztu digituak modeloak aurresateko")

    botoia_irten = Botoia(width//2-350//2, height // 2 + botoien_dist * 4, 350, 40, BUTTON_COLOR, "Irten")

    # Loop orokorra
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # Exekutatzeaz bukatzeko
                running = False
                zer_egin = "amaitu"
            elif event.type == pygame.MOUSEBUTTONDOWN: # Xaguaren botoiren bat sakatzen bada
                if event.button == 1: # Ezkerreko botoia sakatzen bada
                    if botoia_adibideak_ikusi.rect.collidepoint(event.pos):
                        running = False
                        zer_egin = "adibideak_ireki"
                    elif botoia_digituak_predezitu.rect.collidepoint(event.pos):
                        running = False
                        zer_egin = "predezitu_ireki"
                    elif botoia_irten.rect.collidepoint(event.pos):
                        running = False
                        zer_egin = "amaitu"
                    elif botoia_SVC_bisualizadorea.rect.collidepoint(event.pos):
                        running = False
                        zer_egin = "SVM_bisuala"

        # while buklearen amaieran, pantaila marraztu, textua idatzi eta aktualizatzeko
        screen.fill(WHITE)
        screen.blit(izenburua.render("MNIST interaktiboa", True, BLACK),(width//2 - 345, 20))
        botoia_adibideak_ikusi.draw(screen)
        botoia_digituak_predezitu.draw(screen)
        botoia_irten.draw(screen)
        botoia_SVC_bisualizadorea.draw(screen)
        screen.blit(irudia, irudia_rect)
        pygame.display.flip()

    return zer_egin
    
def main():
    # Beharrezko aldagaiak:
    zer_egin = "menua_ireki"

    # Pygame hasi
    pygame.init()

    while zer_egin != "amaitu":

        if zer_egin == "menua_ireki":
            zer_egin = menua()

        elif zer_egin == "adibideak_ireki":
            zer_egin = adibideak_ikusi()

        elif zer_egin == "predezitu_ireki":
            zer_egin = predezitu()
        
        elif zer_egin == "SVM_bisuala":
            zer_egin = SVM_bisuala()

    # Pygame amaitu
    pygame.quit()















if __name__ == "__main__":
    main()