import gradio as gr
import sahi
import torch
from ultralyticsplus import YOLO, render_model_output

# Images
sahi.utils.file.download_from_url(
    "https://raw.githubusercontent.com/kadirnar/dethub/main/data/images/highway.jpg",
    "highway.jpg",
)
sahi.utils.file.download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/tests/data/small-vehicles1.jpeg",
    "small-vehicles1.jpeg",
)
sahi.utils.file.download_from_url(
    "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg",
    "zidane.jpg",
)


model_names = [
    "yolov8n-seg.pt",
    "yolov8s-seg.pt",
    "yolov8m-seg.pt",
    "yolov8l-seg.pt",
    "yolov8x-seg.pt",
]

current_model_name = "yolov8m-seg.pt"
model = YOLO(current_model_name)


def yolov8_inference(
    image: gr.inputs.Image = None,
    model_name: gr.inputs.Dropdown = None,
    image_size: gr.inputs.Slider = 640,
    conf_threshold: gr.inputs.Slider = 0.25,
    iou_threshold: gr.inputs.Slider = 0.45,
):
    """
    YOLOv8 inference function
    Args:
        image: Input image
        model_name: Name of the model
        image_size: Image size
        conf_threshold: Confidence threshold
        iou_threshold: IOU threshold
    Returns:
        Rendered image
    """
    global model
    global current_model_name
    if model_name != current_model_name:
        model = YOLO(model_name)
        current_model_name = model_name
    model.overrides["conf"] = conf_threshold
    model.overrides["iou"] = iou_threshold
    results = model.predict(image, imgsz=image_size, return_outputs=True)
    renders = []
    for image_results in model.predict(image, imgsz=image_size, return_outputs=True):
        render = render_model_output(
            model=model, image=image, model_output=image_results
        )
        renders.append(render)

    return renders[0]


inputs = [
    gr.Image(type="filepath", label="Input Image"),
    gr.Dropdown(
        model_names,
        value=current_model_name,
        label="Model type",
    ),
    gr.Slider(minimum=320, maximum=1280, value=640, step=32, label="Image Size"),
    gr.Slider(
        minimum=0.0, maximum=1.0, value=0.25, step=0.05, label="Confidence Threshold"
    ),
    gr.Slider(minimum=0.0, maximum=1.0, value=0.45, step=0.05, label="IOU Threshold"),
]

outputs = gr.Image(type="filepath", label="Output Image")
title = "Ultralytics YOLOv8 Segmentation Demo"

examples = [
    ["zidane.jpg", "yolov8m-seg.pt", 640, 0.6, 0.45],
    ["highway.jpg", "yolov8m-seg.pt", 640, 0.25, 0.45],
    ["small-vehicles1.jpeg", "yolov8m-seg.pt", 640, 0.25, 0.45],
]
demo_app = gr.Interface(
    fn=yolov8_inference,
    inputs=inputs,
    outputs=outputs,
    title=title,
    examples=examples,
    cache_examples=True,
    theme="default",
)
demo_app.launch(debug=True, enable_queue=True)
