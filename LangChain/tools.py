# Using custom Tools form langchain
from langchain.tools import BaseTool

#  Basic Structure for a tool in langchain,
#  receives string 
#  returns string

# class ImageCaptionTool(BaseTool):
#     name = None
#     description = None

#     def _run(self, img_path):
#        pass

#     def _arun(self, query:str):
#         raise NotImplementedError("This tool does not support async")
    

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image  
import torch  


# class ImageCaptionTool(BaseTool):
#     name = "Image Captioner"
#     # Talking with agent what tool is about. It is very very imp as agent will know what tool to use from all the tools available.
#     description = "Use this tool when given the path to an image that you would like to be described."\
#     "It will return a simple caption describing the image."

#     # Creating tool using langchain
#     def _run(self, img_path):
#         image = Image.open(img_path).convert('RGB')

#         model_name = "Salesforce/blip-image-captioning-large"

#         device = "cpu"

#         processor = BlipProcessor.from_pretrained(model_name)
#         model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

#         inputs = processor(image, return_tensors = 'pt').to(device)

#         output = model.generate(**inputs, max_new_tokens = 20)

#         caption = processor.decode(output[0], skip_special_tokens = True)

#         return caption


#     # run on asynchronous executions
#     def _arun(self, query:str):
#         raise NotImplementedError("This tool does not support async")
        

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch


# class ObjectDetectionTool(BaseTool):
#     name = "Object detector"
#     description = "Use this tool when given the path to an image that you would like to detect objects. "\
#     "It will return a list of all objects detected in the image along with their confidence levels. Each element in the list in format; "\
#     "[x1, y1, x2, y2] class_names confidence score."

#     def _run(self, img_path):
#         image = Image.open(img_path).convert('RGB')

#         processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
#         model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

#         inputs = processor(images = image, return_tensors = "pt")
#         outputs = model(**inputs)

#         target_sizes = torch.tensor([image.size[::-1]])
#         results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

#         detections = ""

#         for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#             detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
#             detections += '{}'.format(model.config.id2label[int(label)])
#             detections += '{}\n'.format(float(score))

#         return detections   

#        # run on asynchronous executions
#     def _arun(self, query:str):
#         raise NotImplementedError("This tool does not support async") 


class ImageCaptionTool(BaseTool):
    name: str = "Image Captioner"  # Add type annotation
    description: str = (
        "Use this tool when given the path to an image that you would like to be described. "
        "It will return a simple caption describing the image."
    )

    def _run(self, img_path: str) -> str:
        image = Image.open(img_path).convert('RGB')
        model_name = "Salesforce/blip-image-captioning-large"
        device = "cpu"

        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        inputs = processor(image, return_tensors='pt').to(device)
        output = model.generate(**inputs, max_new_tokens=20)
        caption = processor.decode(output[0], skip_special_tokens=True)

        return caption

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

class ObjectDetectionTool(BaseTool):
    name: str = "Object detector"  # Add type annotation
    description: str = (
        "Use this tool when given the path to an image that you would like to detect objects. "
        "It will return a list of all objects detected in the image along with their confidence levels. Each element in the list in format; "
        "[x1, y1, x2, y2] class_names confidence score."
    )

    def _run(self, img_path: str) -> str:
        image = Image.open(img_path).convert('RGB')

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detections = ""

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            detections += '{}'.format(model.config.id2label[int(label)])
            detections += '{}\n'.format(float(score))

        return detections

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")