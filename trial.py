import cv2
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM

import numpy as np
import random
import copy

# Define colormap
colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

# Function to draw polygons
def draw_polygons(image, prediction, fill_mask=False):
    """
    Draws segmentation masks with polygons on an image and creates a binary mask.

    Parameters:
    - image: PIL Image object.
    - prediction: Dictionary containing 'polygons' and 'labels' keys.
                  'polygons' is a list of lists, each containing vertices of a polygon.
                  'labels' is a list of labels corresponding to each polygon.
    - fill_mask: Boolean indicating whether to fill the polygons with color.

    Returns:
    - Annotated PIL image.
    - Binary mask as a PIL Image (white for non-segmented regions, black for segmented regions).
    """
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    draw = ImageDraw.Draw(image)
    
    mask = Image.new('L', image.size, color=255)  # White background
    mask_draw = ImageDraw.Draw(mask)
    color_choices = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for polygons, label in zip(prediction['polygons'], prediction['labels']):
        color = random.choice(color_choices)
        for _polygon in polygons:
            _polygon = np.array(_polygon).reshape(-1, 2)
            if len(_polygon) < 3:
                continue
            _polygon = _polygon.reshape(-1).tolist()
            if fill_mask:
                draw.polygon(_polygon, outline=color, fill=color)
            else:
                draw.polygon(_polygon, outline=color)
            mask_draw.polygon(_polygon, fill=0)

    return image, mask

# Run example function
def run_example(task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image_pil, return_tensors="pt")

    # generated_ids = model.generate(
    #     input_ids=inputs["input_ids"].cuda(),
    #     pixel_values=inputs["pixel_values"].cuda(),
    #     max_new_tokens=1024,
    #     early_stopping=False,
    #     do_sample=False,
    #     num_beams=3,
    # )
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image_pil.width, image_pil.height)
    )

    return parsed_answer

# OpenCV to PIL conversion function
def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

# PIL to OpenCV conversion function
def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGR)

if __name__=="__main__":

    model_id = 'microsoft/Florence-2-large'
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval() #.cuda()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Load OpenCV image
    image_cv2 = cv2.imread('img1.jpeg')
    image_cv2 = cv2.resize(image_cv2, (640, 480))


    # Convert OpenCV image to PIL
    image_pil = cv2_to_pil(image_cv2)

    # Run Florence2 model
    task_prompt = '<REGION_TO_SEGMENTATION>'
    results = run_example(task_prompt, text_input="detect all sky")

    # Copy and process the image
    output_image_pil = copy.deepcopy(image_pil)
    image_sky, mask_sky = draw_polygons(output_image_pil, results['<REGION_TO_SEGMENTATION>'], fill_mask=True)

    # Convert processed PIL image back to OpenCV format
    output_image_cv2 = pil_to_cv2(image_sky)

    # Display the result using OpenCV
    cv2.imshow("Segmented Image", output_image_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
