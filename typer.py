from pynput.keyboard import Key, Controller
import time
keyboard = Controller()


while True:
    time.sleep(30)
    keyboard.press(Key.space)
    keyboard.release(Key.space)
