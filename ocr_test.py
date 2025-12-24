from rapidocr import RapidOCR
import pyautogui

engine = RapidOCR()

screenshot = pyautogui.screenshot()

result = engine(screenshot)

if result:
    print("extracted text: ")
    for item in result:
        print(item[1])
else:
    print("No text was detected in the image")
