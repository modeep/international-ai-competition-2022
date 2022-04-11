import os
import matplotlib.pyplot as plt
import time
from PIL import Image

PATH = "result/"

dir_list = os.listdir(PATH)

for i in range((len(dir_list))):
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    img = Image.open(f'{PATH}{i}.png')
    plt.imshow(img)
    plt.show(block=False)
    plt.pause(1)
    plt.close()