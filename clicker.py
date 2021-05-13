import pynput.mouse as ms
import pynput.keyboard as bord

import time
keyboard = bord.Controller()


mouse = ms.Controller()

while True:
    time.sleep(30)
    mouse.position = (300, 5)
    mouse.click(ms.Button.left, 1)
    time.sleep(1)
    keyboard.press(bord.Key.space)
    keyboard.release(bord.Key.space)
    time.sleep(30)
    mouse.position = (100, 5)
    mouse.click(ms.Button.left, 1)
    time.sleep(1)
    keyboard.press(bord.Key.space)
    keyboard.release(bord.Key.space)


