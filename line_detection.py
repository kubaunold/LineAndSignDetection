import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from moviepy.editor import VideoFileClip


# Funkcja ktora wycina region zainteresowan z obrazu
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Funkcja do rysowania linii na zwracanym filmie
def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    line_img = np.zeros( ( img.shape[0], img.shape[1], 3), dtype=np.uint8)
    img = np.copy(img)

    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return img


# Funkcja zwracajaca w ktora strone ma skrecic samochod
def if_out(img, lines):
    height = img.shape[0]
    width = img.shape[1]
    if lines is None:
        return
    else:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Wykrywamy czy ktoras z linii nie jest w srodku obrazu
                if x1 > 150 and x1 < width/2:
                    kontra = "w prawo"
                elif x1 < width-150 and x1 > width/2:
                    kontra = "w lewo"
                else:
                    kontra = ""
        return kontra


def pipeline(image):
    """Przetwarzanie obrazu uzyskanego z wideo."""
    height = image.shape[0]
    width = image.shape[1]

    # Fajny parametr do przebadania - region zainteresowan
    # Ustawiamy wierzcholki obszaru ktory nas interesuje i ma byc wyciety
    region_of_interest_vertices = [
        (0, height-50),                 # upper corner
        (width / 2, (height / 2)-50),   # left-down corner
        (width, height-50),             # right-down corner
    ]
    # Konwersja do skali szarosci
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Uzycie detektora Canny
    cannyed_image = cv2.Canny(gray_image, 100, 200)
    # Przyciecie obrazu do jego regionu zainteresowan
    cropped_image = region_of_interest(cannyed_image, np.array([region_of_interest_vertices], np.int32))
    # cropped_image = region_of_interest(image, np.array([region_of_interest_vertices], np.int32))

    # Transformacja Hougha do wykrywania linii na przycietym obrazie
    # Mozna pobawic sie w zmiane parametrow
    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )

    #Inicjowaie pustych tablic na wspolrzedne linii
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    if lines is None:
        return image
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if math.fabs(slope) < 0.5:
                continue
            if slope <= 0:
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])

    min_y = int(image.shape[0] * (3 / 5))
    max_y = int(image.shape[0])
    
    if len(left_line_y) != 0 and len(left_line_x) !=0 and len(right_line_y) != 0 and len(right_line_x) != 0:
        poly_left = np.poly1d(np.polyfit(
            left_line_y,
            left_line_x,
            deg=1
        ))
        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))
        poly_right = np.poly1d(np.polyfit(
            right_line_y,
            right_line_x,
            deg=1
        ))

        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))
        line_image = draw_lines(
            image,
            [[
                [left_x_start, max_y, left_x_end, min_y],
                [right_x_start, max_y, right_x_end, min_y],
                # adding middle line
                [
                    (int)((right_x_start + left_x_start)/2), max_y,
                    (int)((right_x_end + left_x_end)/2), min_y
                ]
            ]],
            thickness=5)
        tekst = if_out(image, [[
                [left_x_start, max_y, left_x_end, min_y],
                [right_x_start, max_y, right_x_end, min_y],
            ]])
        print(tekst)
    else:
        line_image = image

    return line_image   

def main():
    # Tu podajemy miejsce do zapisu
    white_output = 'test2_output.mp4'
    # Tutaj pilk z ktorego czyta funkcja
    clip1 = VideoFileClip("test2.mp4")
    white_clip = clip1.fl_image(pipeline)
    white_clip.write_videofile(white_output, audio=False)

if __name__ == "__main__":
    main()