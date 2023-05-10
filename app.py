import gradio as gr
import sahi
import torch
from ultralyticsplus import YOLO, render_result
import numpy as np
from PIL import Image

# Images1
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

def visualize_masks(masks):
    # 将 PyTorch 张量转换为 numpy 数组
    masks = masks.detach().cpu().numpy()

    # 计算有多少个 mask
    num_masks = masks.shape[0]

    # 创建一个空白图像，背景颜色为黑色
    height, width = masks.shape[1:]
    img = Image.new('RGB', (width, height),(0,0,0))
    #img.putpalette([0, 0, 0] * 256)
    img_array = np.array(img)

    # 将每个 mask 标记为不同的颜色
    for i in range(num_masks):
        color = np.random.randint(0, 256, size=3)
        #colorimg.paste
        #colorimg = Image.new('RGB', (width,height), color=tuple(np.random.randint(0, 256, size=3)))
        #mask_img_tmp = Image.fromarray(masks[i]).convert('RGB')
        #mask_array = Image.fromarray(masks[i])
        img_array[masks[i] != 0,:] = color
        #mask_img = mask_img.putpalette(color)
        #img.paste(mask_img,(0,0),mask_img_tmp)

        #img.putpalette(color + (0,) * 253)

    # 将 mask 根据颜色映射显示为 RGB 图像
    img_rgb = Image.fromarray(img_array)
    return img_rgb



def yolov8_inference(
    image = None,
    model_name = None,
    dest_width = 512,
    dest_height = 512,
    image_size = 640,
    conf_threshold = 0.25,
    iou_threshold = 0.45,
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
    model.overrides["classes"] = [0]
    results = model.predict(image)
    renders = []
    for image_results in model.predict(image):
        #print("predict results:  ",type(image_results.masks))
        #render = render_result(
        #    model=model, image=image, result=image_results
        #)
        render = visualize_masks(image_results.masks.data)
        render = render.resize((dest_width,dest_height))
        renders.append(render)

    return renders[0]

inputs = [
    gr.Image(type="filepath", label="Input Image"),
    gr.Dropdown(
        model_names,
        value=current_model_name,
        label="Model type",
    ),
    gr.inputs.Slider(minimum=128, maximum=2048, step=64, default=512, label="Width"),
    gr.inputs.Slider(minimum=128, maximum=2048, step=64, default=512, label="Height"),

    gr.Slider(minimum=320, maximum=1280, value=640, step=32, label="Image Size"),
    gr.Slider(
        minimum=0.0, maximum=1.0, value=0.25, step=0.05, label="Confidence Threshold"
    ),
    gr.Slider(minimum=0.0, maximum=1.0, value=0.45, step=0.05, label="IOU Threshold"),
]

outputs = gr.Image(type="filepath", label="Output Image")
title = "Ultralytics YOLOv8 Segmentation For Person Only Now"


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
    examples=None,
    cache_examples=False,
    theme="default",
)
demo_app.launch(debug=True, enable_queue=True)
