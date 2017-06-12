"""Send JPEG image to tensorflow_model_server loaded with ip5wke model.
"""

import numpy
import requests
import base64
import json
import pickle

from time import time
import matplotlib.pyplot as plt

import pygame
import pygame.camera
import time
import sys

def json_numpy_obj_hook(dct):
    """Decodes a previously encoded numpy ndarray with proper shape and dtype.

    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct


def main():
    pygame.init()
    result_surf = pygame.display.set_mode((1728, 1152))

    pygame.camera.init()
    cam = pygame.camera.Camera(pygame.camera.list_cameras()[0], (288, 288), "RGB")
    cam.start()
    plt.ion()

    try:
        while True:
            print("Getting image %d"%time.time())
            img = cam.get_image()
            print("Got image %d"%time.time())
            cropped = pygame.Surface((288, 288))
            cropped.blit(img, (0, 0), (32, 0, 320, 288))
            data = pygame.image.tostring(cropped, 'RGB')
            cropped = pygame.transform.rotate(cropped, 90)
            enc = base64.b64encode(data)
            url = 'http://localhost:8888/'
            files = {'file': enc.decode('ascii')}
            print("Sending request %d"%time.time())
            result = requests.post(url, json=files)

            res = numpy.array(json.loads(result.json()["result"]))
            print("Got Reply %d"%time.time())

            result_surf.blit(img, (0, 0), (32, 0, 320, 288))

            res = res[0].transpose([2, 0, 1])

            for i in range(21):
                x = ((i+1) % 6) * 288
                y = ((i+1) // 6) * 288
                cur_img = pygame.surfarray.make_surface(numpy.tile(numpy.expand_dims(res[i], axis=2), [1, 1, 3])*255.0)
                cur_img = pygame.transform.rotate(cur_img, -90)
                cur_img = pygame.transform.flip(cur_img, True, False)
                result_surf.blit(cur_img, (x, y), (0, 0, 288, 288))

            pygame.display.flip()

            # plt.clf()
            # plt.figure(1, figsize=(15,15))
            # plt.gcf().canvas.set_window_title("Image")
            # plt.subplot(5, 5, 1)
            # img = pygame.surfarray.array3d(cropped)
            # plt.imshow(img/255.0)

            # res = res[0].transpose([2, 0, 1])
            # # for i in range(21):
            # #     plt.subplot(5, 5, i+2)
            # #     plt.imshow((res[i] > 0.6).astype(int),cmap='viridis', interpolation='nearest', vmin=0.0, vmax=1.0)

            # plt.subplot(5, 5, 2)
            # plt.imshow(res[0],cmap='viridis', interpolation='nearest', vmin=0.0, vmax=1.0)

            # plt.subplot(5, 5, 3)
            # plt.imshow(res[5],cmap='viridis', interpolation='nearest', vmin=0.0, vmax=1.0)

            # plt.pause(0.05)
            print("Done %d"%time.time())
    finally:
        pygame.camera.quit()
        pygame.display.quit()
        pygame.quit()
        sys.exit()

if __name__ == '__main__':
    main()