import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes=[0]
model.iou=0.2
# set image
img = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'

# perform inference
results = model(img)

# inference with larger input size
results = model(img, size=1280)

# inference with test time augmentation
results = model(img, augment=True)

# parse results
predictions = results.pred[0]
boxes = predictions[:, :4] # x1, x2, y1, y2
scores = predictions[:, 4]
#categories = predictions[:, 5]

print(predictions)

# show detection bounding boxes on image
results.show()

# save results into "results/" folder
results.save(save_dir='results/')
