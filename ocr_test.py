from rapidocr import RapidOCR
import pyautogui

"""
Test for future agent tool, screenshot screen for more problem context
"""

engine = RapidOCR()

screenshot = pyautogui.screenshot()

result = engine(screenshot)

if result:
    print("extracted text: ")
    for item in result:
        print(item[1])
else:
    print("No text was detected in the image")
