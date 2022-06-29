import numpy as np
import os

class BufferedImageSaver:
    """
    Stores incoming data in a Numpy ndarray and saves the array to disk once
    completely filled.
    """

    def __init__(self, filename: str, size: int,
                 rows: int, cols: int, depth:int, sensorname: str, lane_change_number:int):
        """An array of shape (size, rows, cols, depth) is created to hold
        incoming images (this is the buffer). `filename` is where the buffer
        will be stored once full.
        """
        self.filename = filename + sensorname + '/' + str(lane_change_number) + '/'
        self.size = size
        self.sensorname = sensorname
        dtype = np.float32 if self.sensorname == 'CameraDepth' else np.uint8
        self.buffer = np.empty(shape=(size, rows, cols, depth),
                               dtype=dtype)
        #self.steering_buffer = np.empty(shape=(size), dtype=np.float32)
        self.controls_buffer = np.empty(shape=(size, 6), dtype=np.float32)
        self.index = 0
        self.reset_count = 0  # how many times this object has been reset

    def is_full(self):
        """A BufferedImageSaver is full when `self.index` is one less than
        `self.size`.
        """
        return self.index == self.size

    def reset(self):
        self.buffer = np.empty_like(self.buffer)
        #self.steering_buffer = np.empty_like(self.steering_buffer)
        self.controls_buffer = np.empty_like(self.controls_buffer)
        self.index = 0
        self.reset_count += 1

    def save(self):
        save_name = self.filename + 'images_' +str(self.reset_count) + '.npy'
        #steering_save_name = self.filename + 'steering_'+str(self.reset_count) + '.npy'
        controls_save_name = self.filename + 'controls_'+str(self.reset_count) + '.npy'
        # make the enclosing directories if not already present
        folder = os.path.dirname(save_name)
        if not os.path.isdir(folder):
            os.makedirs(folder)

        # save the buffer
        np.save(save_name, self.buffer[:self.index + 1])
        #np.save(steering_save_name, self.steering_buffer[:self.index + 1])
        np.save(controls_save_name, self.controls_buffer[:self.index + 1])

    @staticmethod
    def process_by_type(raw_img, name):
        """Converts the raw image to a more efficient processed version
        useful for training. The processing to be applied depends on the
        sensor name, passed as the second argument.
        """
        if name == 'CameraRGB':
            return raw_img  # no need to do any processing

        elif name == 'CameraDepth':
            raw_img = raw_img.astype(np.float32)
            total = raw_img[:, :, 2:3] + 256*raw_img[:, :, 1:2] + 65536*raw_img[:, :, 0:1]
            total /= 16777215
            return total
        
        elif name == 'CameraSemSeg':
            return raw_img[:, :, 2: 3]  # only the red channel has information

    def add_image(self, img_bytes, steering_value, llc, rlc, lcsh, junk, distance, name):
        """Save the current buffer to disk and reset the current object
        if the buffer is full, otherwise store the bytes of an image in
        self.buffer.
        """
        if self.is_full():
            self.save()
            self.reset()
            self.add_image(img_bytes, steering_value, llc, rlc, lcsh, junk, distance, name)
        else:
            raw_image = np.frombuffer(img_bytes, dtype=np.uint8)
            raw_image = raw_image.reshape(
                            self.buffer.shape[1], self.buffer.shape[2], -1)
            raw_image = self.process_by_type(raw_image[:, :, :3], name)
            self.buffer[self.index] = raw_image
            #self.steering_buffer[self.index] = steering_value
            self.controls_buffer[self.index, 0] = steering_value
            self.controls_buffer[self.index, 1] = llc
            self.controls_buffer[self.index, 2] = rlc
            self.controls_buffer[self.index, 3] = lcsh
            self.controls_buffer[self.index, 4] = junk
            self.controls_buffer[self.index, 5] = distance
            self.index += 1
