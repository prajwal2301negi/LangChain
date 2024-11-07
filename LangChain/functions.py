# for Captions
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# for Object Detection
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch



def get_image_caption(image_path):
    # Generates caption for the image
    # Argument -> Path to image file.
    # Returns -> Caption for the image (string format).

    image = Image.open(image_path).convert('RGB')

    # model we are using
    model_name = "Salesforce/blip-image-captioning-large"

    # if GPU is use in comp -> 'cuda'
    device = "cpu"

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    # converting image to another representation
    inputs = processor(image, return_tensors = 'pt').to(device)

    # output as we need to generate a sentence only(caption)
    output = model.generate(**inputs, max_new_tokens = 20)

    caption = processor.decode(output[0], skip_special_tokens = True)

    return caption


def detect_objects(image_path):
    # Detects objects in the provided image
    # Argument -> Path to image file.
    # Return -> A string with all the detected objects. Each objects as '[x1, x2, y1, y2, class_name, confidence_score]'.

    image = Image.open(image_path).convert('RGB')

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    inputs = processor(images = image, return_tensors = "pt")
    outputs = model(**inputs)

    # convert outputs(bounding box and class logits) to COCO API
    # keeping detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    detections = ""

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        detections += '{}'.format(model.config.id2label[int(label)])
        detections += '{}\n'.format(float(score))

    return detections    
       



if __name__ == '__main__':
    image_path = r"C:\Users\prajwal\Downloads\jackyWeb\trainedDogdg.jpeg"
    # caption = get_image_caption(image_path)
    # print(caption)
    # arafed dog jumping over a hurdle in a field with a man  
    detections = detect_objects(image_path)
    print(detections)
   #[0, 7, 55, 199]person0.999016523361206
   #[66, 4, 209, 100]dog0.998309850692749