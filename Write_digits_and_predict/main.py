import pygame # Digituak idazteko
import numpy as np
import time
import pandas as pd
from sklearn.svm import SVC # Jada SVM inplementatutako pythoneko pakete bat
from random import randint
import pickle

# Parametroak (alda daitezke):
grid_size = 25  # Karratu bakoitzaren tamaina pixeletan

# Koloreak
WHITE = (238,238,210)
BLACK = (186,202,68)
GRID_COLOR = (200, 200, 200) # Karratuen arteko kolorea eta pantaila atzeko kolorea
TEXT_COLOR = (0,0,0)
BUTTON_COLOR = (200, 200, 200)

# Hemendik aurrerakoa ez aldatu













# -------------------------------------------------------------------------------
# -----------------------------DATU BASEA INPORTATU------------------------------
# -------------------------------------------------------------------------------
entrenamendu_datuak = pd.read_csv("mnist_train.csv")
testeatzeko_datuak = pd.read_csv("mnist_test.csv")
# print(entrenamendu_datuak.head())
X_entrenamendu = entrenamendu_datuak.iloc[:,1:]
Y_entrenamendu = entrenamendu_datuak["label"]

X_test = testeatzeko_datuak.iloc[:,1:]
Y_test = testeatzeko_datuak["label"]


# -------------------------------------------------------------------------------
# -------------------------------MODELOA INPORTATU-------------------------------
# -------------------------------------------------------------------------------

modeloa = pickle.load(open("SkLearn_SVC_model_C_4.pkl","rb"))

# Honek ez badu funtzionatzen, ondorengo kodea exekutatu: 
# modeloa = SVC(C = 4)
# modeloa.fit(X_entrenamendu,Y_entrenamendu)















# -------------------------------------------------------------------------------
# ----------------------------APLIKAZIO INTERAKTIBOA-----------------------------
# -------------------------------------------------------------------------------

# Pantailaren tamainiako aldagai orokorrak:
n = 28

grid_width, grid_height = n * grid_size, n * grid_size
width = grid_width + 300 # Instrukzioak / botoiak gehitzeko
height = grid_height


# Digituak marraztu:
def kuadrikula_marraztu(kuadrikula,screen):
    """
    grid = 28 x 28 tamainiako matrize bat: matrizearen elementuak 0 - 255 tarteko zenbakiak
    izango dira, 28 x 28 pixeleko pantaila baten gris-eskalak izanik
    """
    screen.fill(WHITE)
    for i in range(n):
        for j in range(n):
            balioa = kuadrikula[i][j]
            pygame.draw.rect(screen, (balioa,balioa,balioa), (j * grid_size, i * grid_size, grid_size, grid_size))
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
    botoia_itzuli_menura = Botoia(width - 230, 375, 170, 40, BUTTON_COLOR, "Menura itzuli")

    # Loop orokorra
    while running:
        for event in pygame.event.get():
            keys = pygame.key.get_pressed()
            if event.type == pygame.QUIT: # Exekutatzeaz bukatzeko
                running = False
                zer_egin = "amaitu"
            elif keys[pygame.K_RETURN]: # Enter botoia sakatzerakoan
                zenbagarren_adibidea += 1
            elif event.type == pygame.MOUSEBUTTONDOWN: # Xaguaren botoiren bat sakatzen bada
                if event.button == 1: # Ezkerreko botoia sakatzen bada
                    if botoia_itzuli_menura.rect.collidepoint(event.pos):
                        running = False
                        zer_egin = "menua_ireki"
                    elif botoia_hurrengo_adibidea.rect.collidepoint(event.pos):
                        zenbagarren_adibidea += 1

        # while buklearen amaieran, pantaila marraztu, textua idatzi eta aktualizatzeko
        kuadrikula_marraztu(np.array(X_entrenamendu.iloc[zenbagarren_adibidea,:]).reshape(n,n),screen)
        screen.blit(izenburua.render("MNIST datu-baseko adibideak", True, TEXT_COLOR),(width - 295, 30))
        screen.blit(digitua.render(str(Y_entrenamendu[zenbagarren_adibidea]), True, TEXT_COLOR),(width - 205, 130))
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
    running = True
    marrazten = False
    garbitzen = False
    zenbakia = np.zeros((n,n),dtype=int)
    predikzioa = "?"

    # Textuak idazteko beharrezkoa:
    izenburua = pygame.font.Font(None, 37)
    digitua = pygame.font.Font(None, 300)
    instrukzioak = pygame.font.Font(None,20)

    # Botoia:
    botoia_predezitu = Botoia(width - 230, 275, 170, 40, BUTTON_COLOR, "Zenbakia predezitu")
    botoia_garbitu = Botoia(width - 230, 325, 170, 40, BUTTON_COLOR, "Garbitu")
    botoia_itzuli_menura = Botoia(width - 230, 375, 170, 40, BUTTON_COLOR, "Menura itzuli")

    # Loop orokorra
    while running:
        for event in pygame.event.get():
            keys = pygame.key.get_pressed()
            if event.type == pygame.QUIT: # Exekutatzeaz bukatzeko
                running = False
                zer_egin = "amaitu"
            elif keys[pygame.K_RETURN]: # Enter botoia sakatzerakoan
                predikzioa = modeloa.predict(zenbakia.reshape((1,n**2)))[0]
            elif event.type == pygame.MOUSEBUTTONDOWN: # Xaguaren botoiren bat sakatzen bada
                if event.button == 1: # Ezkerreko botoia sakatzen bada
                    if botoia_itzuli_menura.rect.collidepoint(event.pos):
                        running = False
                        zer_egin = "menua_ireki"
                    elif botoia_predezitu.rect.collidepoint(event.pos):
                        predikzioa = modeloa.predict(zenbakia.reshape((1,n**2)))[0]
                    elif botoia_garbitu.rect.collidepoint(event.pos):
                        zenbakia = np.zeros((n,n))
                        predikzioa = "?"
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
        screen.blit(digitua.render(str(predikzioa), True, TEXT_COLOR),(width - 215, 80))
        screen.blit(instrukzioak.render("Sakatu 'enter' predezitzeko", True, TEXT_COLOR),(width - 295, 55))
        botoia_itzuli_menura.draw(screen)
        botoia_predezitu.draw(screen)
        botoia_garbitu.draw(screen)
        pygame.display.flip()
        
    return zer_egin


def menua():
    """
    Programa hau menua exekutatzeko da
    """
    # Pygame hasi
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("MNIST interaktiboa")

    # Beharrezko aldagaia
    running = True

    # Botoia:
    botoia_adibideak_ikusi = Botoia(width//2, 100, 160, 40, BUTTON_COLOR, "Ikusi MNIST-en adibideak")
    botoia_digituak_predezitu = Botoia(width//2, 200, 160, 40, BUTTON_COLOR, "Marraztu digituak modeloak aurresateko")


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

        # while buklearen amaieran, pantaila marraztu, textua idatzi eta aktualizatzeko
        screen.fill(WHITE)
        botoia_adibideak_ikusi.draw(screen)
        botoia_digituak_predezitu.draw(screen)
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



    # Pygame amaitu
    pygame.quit()















if __name__ == "__main__":
    main()