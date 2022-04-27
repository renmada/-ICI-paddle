import utils.transforms as T
import cv2


def get_transforms(args):
    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform_test


def load_and_process(args):
    pic = cv2.imread(args.pic_path)
    transform_test = get_transforms(args)
    return transform_test(pic)
