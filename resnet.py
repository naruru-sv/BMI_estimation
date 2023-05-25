import os

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import read_image, image
import cv2
# Using pretrained weights:
resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Initialize the Weight Transforms
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()
# Initialize model
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# Set model to eval mode
model.eval()

imgs_path = "/home/nata/Documents/melissa"
imgs = os.listdir(imgs_path)
i = 0
for filename in imgs:
    # if i == 10:
    #     break
    try:
        filename = filename.replace("__", "_")
        a = filename.split("_")
        # height = float(a[3]) / 39.37
        # weight = float(a[2]) / 2.205
        # bmi = weight / (height**2)
        bmi = 43.3
    except:
        print(f"smth wrong with img split. filename = {filename}")
        continue

    try:
        img = read_image(imgs_path + "/" + filename,  mode=image.ImageReadMode.RGB)
        img_transformed = preprocess(img)
        batch = preprocess(img).unsqueeze(0)
        prediction = model(batch).squeeze(0)
    except:
        print(f"smth wrong with prediction. Index = {i}, filename = {filename}")
        continue


    try:
        with open("/home/nata/pythonProj/STRAPS/resnet_melissa.csv", "a") as file:
            a_list = prediction.tolist()
            pred_str = ",".join(map(str, a_list))
            file.write(filename + "," + str(bmi) + "," + pred_str)
            file.write("\n")
    except:
        print(f"smth wrong with result write. filename = {filename}")
        continue

    i += 1

print("result images = ", i)

# img_path = "/home/nata/pythonProj/STRAPS-3DHumanShapePose/demo/0001.png"
# Apply it to the input image




# prediction = model(batch).squeeze(0).softmax(0)
# print("prediction softmax = ", prediction)

# print("------------------------------------")
# prediction = model(batch)
# print("prediction = ", prediction)
# class_id = prediction.argmax().item()
# score = prediction[class_id].item()
# category_name = weights.meta["categories"][class_id]
# print(f"{category_name}: {100 * score}%")

# pred = model.forward(img_transformed)
# print(pred)