import torch
import argparse
import cv2
import KittiToYodaROIs

class SplitLabels():
    def __init__(self, dir, training=True, transform=None):
        self.dir = dir
        self.training = training
        self.mode = 'train'
        if self.training == False:
            self.mode = 'test'
        self.img_dir = os.path.join(dir, self.mode, 'image')
        self.label_dir = os.path.join(dir, self.mode, 'label')
        self.transform = transform
        self.num = 0
        self.img_files = []
        for file in os.listdir(self.img_dir):
            if fnmatch.fnmatch(file, '*.png'):
                self.img_files += [file]

        self.max = len(self)

        # print('break 12: ', self.img_dir)
        # print('break 12: ', self.label_dir)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        filename = os.path.splitext(self.img_files[idx])[0]
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        label_path = os.path.join(self.label_dir, filename+'.txt')
        labels_string = None

        with open(label_path) as label_file:
            labels_string = label_file.readlines()
        labels = []

        for i in range(len(labels_string)):
            lsplit = labels_string[i].split(' ')
            label = [lsplit[0], int(self.class_label[lsplit[0]]), float(lsplit[4]), float(lsplit[5]), float(lsplit[6]), float(lsplit[7])]
            labels += [label]
        return image, labels

    def __iter__(self):
        self.num = 0
        return self

    def __next__(self):
        if (self.num >= self.max):
            raise StopIteration
        else:
            self.num += 1
            return self.__getitem__(self.num-1)


def main():

    print('running showKitti ...')
    # freeze_support()

    argParser = argparse.ArgumentParser()
    argParser.add_argument('-i', metavar='input_dir', type=str, help='input dir (./)')
    argParser.add_argument('-m', metavar='mode', type=str, help='[train/test]')

    args = argParser.parse_args()

    input_dir = None
    if args.i != None:
        input_dir = args.i

    training = True
    if args.m == 'test':
        training = False


    min_dx = 10000
    max_dx = -1
    min_dy = 10000
    max_dy = -1

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('using device ', device)

    dataset = KittiDataset(dir=input_dir, training=training)

    ROI_shapes = []
    ROI_dx = []
    ROI_dy = []
    i = 0
    for item in enumerate(dataset):

        idx = item[0]
        image = item[1][0]
        label = item[1][1]
        print(i, idx, label)
        i += 1

        for j in range(len(label)):
            name = label[j][0]
            name_class = label[j][1]
            minx = int(label[j][2])
            miny = int(label[j][3])
            maxx = int(label[j][4])
            maxy = int(label[j][5])
            cv2.rectangle(image, (minx,miny), (maxx, maxy), (0,0,255))

            if name_class == 2:
                dx = maxx - minx + 1
                if dx > max_dx:
                    max_dx = dx
                if dx < min_dx:
                    min_dx = dx

                dy = maxy - miny + 1
                if dy > max_dy:
                    max_dy = dy
                if dy < min_dy:
                    min_dy = dy

                ROI_shapes += [(dx,dy)]
                ROI_dx += [dx]
                ROI_dy += [dy]

        if display == True:
            cv2.imshow('image', image)
            key = cv2.waitKey(0)
            if key == ord('x'):
                break

###################################################################

main()