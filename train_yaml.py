import torch
import numpy as np
import yaml
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
import cv2
import os
from utils.fake_objects import create_virtual_obj
import serial.tools.list_ports
import time
from config.cfg import yml_parse
from utils.cfg_from_gui import yml_parse2


# create nerf-like model
class model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(model, self).__init__()
        # input layer
        self.input = nn.Linear(input_dim, 128)
        # middle layers
        self.linear = nn.Linear(128, 128)
        # relu active function
        self.active = nn.ReLU()
        # self.dropout = nn.Dropout()
        # skip connection layer
        self.skip = nn.Linear(128 + input_dim, 128)
        # downsizing layer
        self.down = nn.Linear(128, 64)
        # output layer
        self.output = nn.Linear(64, output_dim)
        self.out_active = nn.Sigmoid()

    def forward(self, x):
        # store input for connection layer
        input = x
        x = self.input(x)
        x = self.active(x)
        for i in range(2):
            x = self.linear(x)
            x = self.active(x)
        x = torch.cat([x, input], dim=-1)
        x = self.skip(x)
        x = self.active(x)
        for i in range(2):
            x = self.linear(x)
            x = self.active(x)
        x = self.down(x)
        x = self.active(x)
        x = self.output(x)
        x = self.out_active(x)
        return x


# main training class
class Train:
    def __init__(self):
        # current file path
        self.file_path = os.path.dirname(__file__)
        self.cfg = yml_parse()
        self.cfg2 = yml_parse2()
        # device and serial config
        self.use_virtual_object = self.cfg['uvo']  # use a virtual object instead of sensor data, use "shape1" to "shape4"
        # serial_port = cfg[sport]  # what port is the device on
        self.sample_per_round = self.cfg['spr']  # how many samples per training
        self.sample_round = self.cfg['sr']  # how many rounds of sampling and training
        self.camera_dist = self.cfg['cd']  # distance between camera and axis
        self.device_name = self.cfg['device']  # serial device name
        self.rot_baud = self.cfg['baud']  # baud rate of rotation angle input
        # network config
        self.learning_rate = self.cfg['lr']  # learning rate of the network
        self.epoch = self.cfg['epoch']  # how many iterations per sample round
        self.batch_size = self.cfg['batch_size']  # how many samples being trained at once
        self.save_path = self.file_path + '\\model\\'  # where to save the model
        self.load_model = self.cfg['load']  # use pre-trained model
        # render config
        self.pix_per_unit = self.cfg['ppu']  # how many pixels per cm
        self.render_img_angle = self.cfg['render_angle']  # what angles to render for the output test image(0-1 where 0 is 0 degree and 1 360 degree)
        self.shade = self.cfg['shade']  # adjust this to models to get best render quality, normally 1 is fine
        self.tmp_path = self.file_path + '\\cache\\'  # path to store the rendered images for the video
        self.video_path = self.file_path + '\\output\\'  # path to store the generated video
        self.only_render = self.cfg['render_only']  # if there are existing trained model file, this should be true
        # select device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # initialize serial input
        if self.only_render is not True and self.use_virtual_object is None:
            self.ports = list(serial.tools.list_ports.comports())
            self.port = None
            for p in self.ports:
                p = str(p)
                if self.device_name in p:
                    port = p.split(' ')[0]
            assert self.port is not None, 'Arduino not found'
            print('found Arduino at: ', self.port)
            # rot input
            self.rot_uno = serial.Serial(
                port=self.port,
                baudrate=self.rot_baud,
                timeout=1,
            )
            self.rot_uno.flushInput()
            # dist input
            # dist_uno = serial.Serial(
            #     port=port,
            #     baudrate=dist_baud,
            #     timeout=1,
            # )
            # dist_uno.flushInput()
        # create model, loss function and optimizer objects
        self.model = model(input_dim=2, output_dim=1).to(self.device)
        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.use_virtual_object is None:
            self.min_height, self.max_height, self.min_angle, self.max_angle = self.position_config()
        else:
            self.min_height, self.max_height, self.min_angle, self.max_angle = 0, 20, 0, 1
        # if only_render or load_model:
        #     model.load_state_dict(torch.load(model_path))
        #     print("model loaded!\n")

    def position_config(self):
        """
        get basic parameters about the size of the object
        by communicating with arduino through serial
        """
        print('Starting configuration.')
        start_time = time.time()
        now_time = start_time
        interval = 60
        self.min_h = np.inf
        self.max_h = -1
        self.min_a = np.inf
        self.max_a = -1
        while now_time - start_time < interval:
            now_time = time.time()
            self.rot_angle, self.height, _ = self.fetch_serial()
            if self.rot_angle < 0 or self.rot_angle > 1023:
                continue
            if self.rot_angle < self.min_a:
                self.min_a = self.rot_angle
            if self.rot_angle > self.max_a:
                self.max_a = self.rot_angle
            if self.height < self.min_h:
                self.min_h = self.height
            if self.height > self.max_h:
                self.max_h = self.height
        assert self.min_h != np.inf and self.max_h > 0 and self.min_a != np.inf and self.max_a > 0, 'Cannot initialize configuration.'
        print('Configuration complete!', ' height range: ', self.min_h, '-', self.max_h, ' angle range: ', self.min_a, '-', self.max_a)
        return self.min_h, self.max_h, self.min_a, self.max_a

    def fetch_serial(self):
        """fetch a series of serial input"""
        if self.use_virtual_object is not None:
            rot_angle, height, dist = create_virtual_obj(self.use_virtual_object, self.camera_dist)
        else:
            dist = -1
            rot_angle = -1
            height = -1
            start_try = time.time()
            # while dist < 0 or dist > camera_dist or rot_angle < 0 or rot_angle > 1:
            while True:
                now_try = time.time()
                assert now_try - start_try <= 5, 'Connection Lost, please check Arduino.'
                if self.rot_uno.inWaiting():
                    # read line?
                    line = self.rot_uno.read(self.rot_uno.inWaiting())
                    if len(line) == 32:
                        line = line.decode('gbk').split()[1]
                        # print('hi', line, 'end')
                        if line[0] != '+' or line[-1] != '/':
                            continue
                        encode1 = line.split('+')[1]
                        raw_rot, encode2 = encode1.split('-')
                        raw_height, encode3 = encode2.split('*')
                        raw_dist = encode3.split('/')[0]
                        # process raw data
                        rot_angle = int(raw_rot)
                        height = int(raw_height) / 58.
                        dist = int(raw_dist) / 58.
                        break
            # height = min_height + np.random.rand() * (max_height - min_height)
            # dist = camera_dist / 2.
            # print("angle: ", rot_angle, " height: ", height, " dist: ", dist)
        return rot_angle, height, dist

    def write_yaml(self, percent, state):
        f = open(self.file_path + '\\utils\\progress.yaml', 'r')
        _ = yaml.safe_load(f)
        f.close()
        f = open(self.file_path + '\\utils\\progress.yaml', 'w')
        progress = {'percent': percent, 'state': state}
        yaml.dump(progress, f)
        f.close()
        return

    def main(self):
        """
        main function which will be executed automatically if the program starts
        will:
        1. read the values
        2. train the network
        3. render the center scene
        """
        self.__init__()
        # loss = []
        if self.only_render is False:
            print("start training!\n")
            for i in range(int(self.sample_round)):
                self.write_yaml(i * 100 / int(self.sample_round), 'sampling')
                sample_x = []
                sample_y = []
                for j in range(int(self.sample_per_round)):
                    [rot_angle, height, dist] = self.fetch_serial()
                    # normalize angle and height according their max-min values, results between 0 and 1
                    rot_angle = (rot_angle - self.min_angle) / (self.max_angle - self.min_angle)
                    height = (height - self.min_height) / (self.max_height - self.min_height)
                    # dist is between 0 and 1 with 0 being the camera plane and 1 being the center
                    dist /= self.camera_dist
                    # add the samples to x and y
                    sample_x.append([rot_angle, height])
                    sample_y.append([dist])
                if len(sample_y) == 0:
                    print("no data! check sensors\n")
                    continue
                print("sensor data fetched! count:", len(sample_x), "\n")
                x = torch.Tensor(sample_x)
                y = torch.Tensor(sample_y)
                # x = nn.functional.normalize(x, dim=0)
                self.dataset = TensorDataset(x, y)
                self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
                print(f"sample round: [{i + 1}/{int(self.sample_round)}]")
                print("-" * 80)
                # update progress
                self.write_yaml(i * 100 / int(self.sample_round), 'training')
                # train loop
                # x = range(i + 1)
                self.train_loop()
                # loss.append(train_loop(dataloader))
                img = self.render(self.render_img_angle)
                plt.imsave(self.file_path + '\\utils\\progress_pic.jpg', img, cmap='gray')
                # fig, ax = plt.subplots(1, 1)
                # ax.plot(x, loss)
                # ax.legend(['loss'])
                # fig.savefig(file_path + '\\utils\\loss_graph.jpg')
                torch.save(self.model.state_dict(), self.save_path + self.cfg2['train_name'] + ".pth")
            self.write_yaml(100, 'finished')
            print("model saved to: \n", self.save_path + "\n")
        return

    def train(self):
        """
        train the model with optimization
        :param model: the nn.Module subclass object
        :param dataloader: dataloader containing training data
        :param loss_fn: loss function object we just created
        :param optimizer: optimizer object we just created
        :return: none
        """
        dataset_size = len(self.dataloader.dataset)
        for batch, bunch in enumerate(self.dataloader):
            x = bunch[0]
            y = bunch[1]
            x = x.to(self.device)
            y = y.to(self.device)
            y_predict = self.model(x)
            loss = self.loss_fn(y_predict.T, y.T)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch % 100 == 0:
                loss_value = loss.item()
                print(f"current: [{batch * self.batch_size}/{dataset_size}] loss: {loss_value}")
        return loss_value

    def train_loop(self):
        """
        train for epoch times
        :param dataloader: training data
        :return: none
        """
        for i in range(int(self.epoch)):
            print(f"\nepoch[{i + 1}/{int(self.epoch)}]\n")
            loss = self.train()
        print("\n")
        return loss

    def run_model(self, x):
        """
        run the model for rendering, no backprop!
        :param model: the model
        :param x: the input
        :return: output data
        """
        with torch.no_grad():
            x = x.to(self.device)
            result = self.model(x)
        return result

    def render(self, theta):
        """
        render a scene
        :param theta: the position of the scene around z axis
        :return: hopefully a gray image of the scene
        """
        # calculate the size of the image
        height = int((self.max_height - self.min_height) * self.pix_per_unit)
        width = int(self.camera_dist * 2 * self.pix_per_unit)
        # initialize the image with black background
        img = np.zeros((height, width))
        img[:, :] = -self.camera_dist
        # calculate the angles needed for rendering the image
        unit_angle = 1. / (self.pix_per_unit * self.camera_dist) / (2 * np.pi)
        # calculate the left and right bound angle of the image, value between 0 and 1
        left_bound = (theta - 0.5) % 1
        right_bound = (theta + 0.5) % 1
        # loop through all the angles on the left and project them onto the image
        # go from -180 to 0 to keep the depth relationship right
        ttl_angle = 0
        while ttl_angle <= 0.5:
            x = torch.ones((height, 2))
            x[:, 0] = (left_bound + ttl_angle) % 1
            x[:, 1] = torch.arange(height - 1, -1, step=-1) / height
            result = self.run_model(x).to("cpu")
            dist = result[:, 0].numpy()
            # this is the real distance
            real_dist = dist * self.camera_dist
            # the angle between the plane and the result
            angle = (0.5 - ttl_angle) * np.pi * 2.
            # horizontal coordinate on the image
            h_value = ((self.camera_dist - real_dist) * np.sin(angle) * self.pix_per_unit).astype(int)
            h_value[h_value > (width // 2)] = width // 2
            h_value = width // 2 - h_value
            # real distance to the image plane
            dist_value = ((self.camera_dist - real_dist) * np.cos(angle) * self.pix_per_unit).astype(float)
            for i in range(height):
                img[i][h_value[i]] = dist_value[i]
            ttl_angle += unit_angle
        # go again on the right side
        ttl_angle = 0
        while ttl_angle <= 0.5:
            x = torch.ones((height, 2))
            x[:, 0] = (right_bound - ttl_angle) % 1
            x[:, 1] = torch.arange(height - 1, -1, step=-1) / height
            result = self.run_model(x).to("cpu")
            dist = result[:, 0].numpy()
            real_dist = dist * self.camera_dist
            angle = (0.5 - ttl_angle) * np.pi * 2.
            h_value = ((self.camera_dist - real_dist) * np.sin(angle) * self.pix_per_unit).astype(int)
            h_value[h_value > (width // 2)] = width // 2
            h_value = width // 2 + h_value
            dist_value = ((self.camera_dist - real_dist) * np.cos(angle) * self.pix_per_unit).astype(float)
            for i in range(height):
                img[i][h_value[i]] = dist_value[i]
            ttl_angle += unit_angle
        img += self.camera_dist
        img /= np.max(img)
        img **= self.shade
        return img

    def generate_vid(self):
        self.__init__()
        self.write_yaml(0, 'starting')
        self.model.load_state_dict(torch.load(self.file_path + '\\model\\' + self.cfg2['render_name'] + ".pth"))
        tmp_folder = self.tmp_path
        # if os.path.isdir(tmp_folder):
        #     shutil.rmtree(tmp_folder)
        # os.makedirs(tmp_folder, exist_ok=False)
        print("generating images...")
        for i in range(100):
            time.sleep(0.05)
            angle = i / 100.
            img = self.render(angle)
            img *= 255
            if i < 10:
                index = "00" + str(i)
            elif i < 100:
                index = "0" + str(i)
            else:
                index = str(i)
            cv2.imwrite(tmp_folder + "rendered_tmp_" + index + ".jpg", img)
            if (i + 1) % 10 == 0:
                print(f"generated [{i + 1}/{100}]")
            self.write_yaml(i, 'rendering')
        self.write_yaml(100, 'generating')
        print("images successfully generated!\n")
        height = int((self.max_height - self.min_height) * self.pix_per_unit)
        width = int(self.camera_dist * 2 * self.pix_per_unit)
        frame_size = [width, height]
        out = cv2.VideoWriter(self.video_path + 'result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 60, frame_size)
        name = os.listdir(tmp_folder)
        for file in name:
            filename = tmp_folder + file
            img = cv2.imread(filename)
            out.write(img)
        out.write(cv2.imread(tmp_folder + name[0]))
        for i in range(len(name)):
            filename = tmp_folder + name[len(name) - i - 1]
            img = cv2.imread(filename)
            out.write(img)
        out.release()
        self.write_yaml(100, 'finished')
